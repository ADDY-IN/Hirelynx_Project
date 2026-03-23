"""
Smart Candidate Search Service
================================
Two modes for the POST /v1/admin/candidates/search endpoint:

  1. FILTER MODE ("mode": "filter")
     - Structured filters: category, locations, skills, jobType,
       experience range, salary range, minMatchScore
     - Optional text `query` for name / email / skill keyword matching
     - minMatchScore uses pre-computed jobMatchScore from the matches table
       (NULL = unknown score, included unless minMatchScore > 0)

  2. AI MODE ("mode": "ai")
     - Natural language `query`: "experienced plumber in Toronto with 5 years"
     - BERT semantic search against resumeParsedJson text + skill matching
     - Returns candidates ranked by semantic relevance

Both modes:
  - Join users ← candidate_profiles ← matches (latest per candidate)
  - Return a uniform candidate card response shape
  - Are admin-only (auth handled at the route level)
"""
import re
import json
import logging
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Any, Optional
from app.scoring import ScoringEngine
from app.config import settings
from app.utils import encode_id

logger = logging.getLogger(__name__)
scoring_engine = ScoringEngine(weight=settings.SCORING_WEIGHT)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_years(work_exp: Any) -> float:
    """Sum years of work experience from workExperience JSON."""
    total = 0.0
    if not isinstance(work_exp, list):
        return total
    current_year = 2026
    for exp in work_exp:
        if not isinstance(exp, dict):
            continue
        start = str(exp.get("startDate", "") or "")
        end   = str(exp.get("endDate", "") or "")
        curr  = bool(exp.get("currentlyWorking", False))
        sy = re.search(r"\b(20\d{2}|19\d{2})\b", start)
        ey = re.search(r"\b(20\d{2}|19\d{2})\b", end)
        if sy:
            s_yr = int(sy.group(1))
            e_yr = current_year if curr else (int(ey.group(1)) if ey else s_yr + 1)
            total += max(0.0, float(e_yr - s_yr))
    return round(total, 1)


def _skill_names(skills_json: Any) -> List[str]:
    """Extract lowercase skill name strings from a skills JSON column."""
    names = []
    if not isinstance(skills_json, list):
        return names
    for s in skills_json:
        if isinstance(s, dict) and s.get("name"):
            names.append(str(s["name"]).strip())
        elif isinstance(s, str) and s.strip():
            names.append(s.strip())
    return names


def _get_personal_details(parsed_json: Any) -> dict:
    """
    Get personalDetails from resumeParsedJson.
    Handles two storage shapes:
      - flat:    { "personalDetails": {...}, ... }
      - nested:  { "structuredData": { "personalDetails": {...} }, ... }
    """
    if not isinstance(parsed_json, dict):
        return {}
    # Try top-level first
    pd = parsed_json.get("personalDetails")
    if not pd:
        # Fall back to structuredData nesting
        structured = parsed_json.get("structuredData") or {}
        pd = structured.get("personalDetails") if isinstance(structured, dict) else None
    return pd if isinstance(pd, dict) else {}


def _extract_location_phone(parsed_json: Any):
    """Extract location string and phone from resumeParsedJson.personalDetails."""
    pd = _get_personal_details(parsed_json)
    if not pd:
        return None, None
    city     = (pd.get("city") or "").strip()
    province = (pd.get("province") or "").strip()
    loc_gen  = (pd.get("location") or "").strip()
    location = ", ".join(filter(None, [city, province])) or loc_gen or None
    phone    = (pd.get("phone") or "").strip() or None
    return location, phone


def _build_card(row, extra: Dict) -> Dict[str, Any]:
    """Build the uniform candidate response card from a DB row."""
    (user_id, email, first_name, last_name, avatar, profile_pic,
     is_verified, status, created_at,
     profile_id, skills, education, work_exp,
     parse_status, parsed_json, job_match_score, matched_skills_list) = row

    full_name            = f"{first_name or ''} {last_name or ''}".strip() or None
    skill_list           = _skill_names(skills)
    years_exp            = _extract_years(work_exp)
    location, phone      = _extract_location_phone(parsed_json)

    card = {
        "userId":           user_id,
        "candidateId":      profile_id,
        "token":            encode_id("USER", user_id),
        "email":            email,
        "firstName":        first_name,
        "lastName":         last_name,
        "fullName":         full_name,
        "avatar":           avatar or profile_pic,
        "isEmailVerified":  is_verified,
        "status":           status,
        "createdAt":        str(created_at) if created_at else None,
        "location":         location,
        "phone":            phone,
        "skills":           skill_list,
        "education":        education or [],
        "workExperience":   work_exp or [],
        "experienceYears":  years_exp,
        "resumeParseStatus": parse_status,
        "jobMatchScore":    float(job_match_score) if job_match_score is not None else None,
        "matchedSkills":    matched_skills_list if isinstance(matched_skills_list, list) else [],
    }
    card.update(extra)
    return card


# ---------------------------------------------------------------------------
# Main query — fetches all candidates with their latest match score
# ---------------------------------------------------------------------------

_CANDIDATE_SQL = text("""
    SELECT
        u.id            AS user_id,
        u.email,
        u."firstName",
        u."lastName",
        u."avatar",
        u."profilePicture",
        u."isEmailVerified",
        u.status,
        u."createdAt",
        cp.id           AS profile_id,
        cp."skills",
        cp."education",
        cp."workExperience",
        cp."resumeParseStatus",
        cp."resumeParsedJson",
        m."jobMatchScore",
        m."matchedSkillsList"
    FROM users u
    LEFT JOIN candidate_profiles cp ON cp."userId" = u.id
    LEFT JOIN LATERAL (
        SELECT "jobMatchScore", "matchedSkillsList"
        FROM   matches
        WHERE  "candidateId" = cp.id
          AND  "jobMatchScore" IS NOT NULL
        ORDER BY id DESC
        LIMIT 1
    ) m ON TRUE
    WHERE u.role = 'CANDIDATE'
      AND u."deletedAt" IS NULL
    ORDER BY u.id DESC
""")


class SearchService:

    # ----------------------------------------------------------------
    # PUBLIC entry point
    # ----------------------------------------------------------------

    @staticmethod
    def smart_search(
        db:            Session,
        mode:          str             = "filter",
        query:         Optional[str]   = None,
        filters:       Optional[Any]   = None,   # CandidateSearchFilters or dict
        limit:         int             = 20,
    ) -> List[Dict[str, Any]]:
        """
        Unified smart search. Delegates to filter_search or ai_search
        based on `mode`.
        """
        if mode == "ai":
            return SearchService._ai_search(db, query or "", limit)
        return SearchService._filter_search(db, query, filters, limit)

    # ----------------------------------------------------------------
    # MODE 1: FILTER SEARCH
    # ----------------------------------------------------------------

    @staticmethod
    def _filter_search(
        db:      Session,
        query:   Optional[str],
        filters: Optional[Any],
        limit:   int,
    ) -> List[Dict[str, Any]]:

        rows = db.execute(_CANDIDATE_SQL).fetchall()

        # Normalise filters
        f = filters
        if f is None:
            f = {}
        if hasattr(f, "model_dump"):
            f = f.model_dump(exclude_none=False)
        elif hasattr(f, "dict"):
            f = f.dict(exclude_none=False)

        category       = (f.get("category") or "").strip().lower()
        locations      = [l.strip().lower() for l in (f.get("locations") or []) if l]
        req_skills     = [s.strip().lower() for s in (f.get("skills") or []) if s]
        job_types      = [j.strip().upper() for j in (f.get("jobType") or []) if j]
        exp_min        = f.get("experienceMin")
        exp_max        = f.get("experienceMax")
        min_score      = float(f.get("minMatchScore") or 0.0)
        query_low      = (query or "").strip().lower()

        results = []
        for row in rows:
            (user_id, email, first_name, last_name, avatar, profile_pic,
             is_verified, status, created_at,
             profile_id, skills, education, work_exp,
             parse_status, parsed_json, job_match_score, matched_skills_list) = row

            skill_names = _skill_names(skills)
            skill_lower = [s.lower() for s in skill_names]
            years_exp   = _extract_years(work_exp)
            full_name   = f"{first_name or ''} {last_name or ''}".strip().lower()
            email_str   = (email or "").lower()

            # ── minMatchScore filter ─────────────────────────────────
            # NULL = no score yet → include (don't penalise un-scored candidates)
            if min_score > 0 and job_match_score is not None:
                if float(job_match_score) < min_score:
                    continue

            # ── Category filter ──────────────────────────────────────
            # Category is matched against the candidate's skills / parsed role
            if category:
                category_hit = any(category in s for s in skill_lower)
                if not category_hit and parsed_json and isinstance(parsed_json, dict):
                    role_text = ""
                    for exp in (parsed_json.get("workExperience") or []):
                        if isinstance(exp, dict):
                            role_text += f" {exp.get('role','')} {exp.get('jobTitle','')}".lower()
                    category_hit = category in role_text
                if not category_hit:
                    continue

            # ── Location filter ───────────────────────────────────!
            if locations:
                pd_loc = _get_personal_details(parsed_json)
                candidate_location = (
                    f"{pd_loc.get('city','')} {pd_loc.get('province','')} "
                    f"{pd_loc.get('location','')}"
                ).lower().strip()
                if not any(loc in candidate_location or loc in email_str for loc in locations):
                    remote_wanted = any("remote" in l for l in locations)
                    if not remote_wanted:
                        continue

            # ── Skills filter (candidate must have ALL required skills) ───
            if req_skills:
                if not all(
                    any(rs in sl or sl in rs for sl in skill_lower)
                    for rs in req_skills
                ):
                    continue

            # ── Experience range filter ──────────────────────────────
            if exp_min is not None and years_exp < exp_min:
                continue
            if exp_max is not None and years_exp > exp_max:
                continue

            # ── jobType / workType filter ────────────────────────────
            # "REMOTE" and "HYBRID" are checked against resume/parsed data
            if job_types:
                # Check if any of the candidate's work experience matches
                candidate_types = set()
                if work_exp and isinstance(work_exp, list):
                    for exp in work_exp:
                        if isinstance(exp, dict):
                            et = str(exp.get("employmentType", "") or "").upper()
                            if et:
                                candidate_types.add(et)
                # Also check workType from parsed profile personalDetails
                if parsed_json and isinstance(parsed_json, dict):
                    wt = str(parsed_json.get("workType") or "").upper()
                    if wt:
                        candidate_types.add(wt)
                if candidate_types and not any(jt in candidate_types for jt in job_types):
                    continue
                # If no employment type data at all, include the candidate

            # ── Text query filter (name / email / skill / location / phone) ─
            relevance = 1.0
            if query_low:
                # Get personalDetails using the helper (handles flat + nested)
                pd_info          = _get_personal_details(parsed_json)
                candidate_loc_str = (
                    f"{pd_info.get('city','')} {pd_info.get('province','')} "
                    f"{pd_info.get('location','')}"
                ).lower().strip()
                candidate_phone = (pd_info.get("phone") or "").lower()

                text_blob = (
                    f"{full_name} {email_str} {' '.join(skill_lower)} "
                    f"{candidate_loc_str} {candidate_phone}"
                )

                # Match any word of the query against the blob (OR logic per word)
                words = [w for w in query_low.split() if len(w) > 1]
                if not any(word in text_blob for word in words):
                    continue

                # Boost relevance for closer matches
                if query_low in full_name or query_low in email_str:
                    relevance = 2.0
                elif any(query_low in field for field in [candidate_loc_str, candidate_phone]):
                    relevance = 1.5

            card = _build_card(row, {"relevance": relevance})
            results.append(card)

        # Sort: by relevance (text query boost), then by jobMatchScore
        results.sort(
            key=lambda c: (c.get("relevance", 1.0), c.get("jobMatchScore") or 0.0),
            reverse=True,
        )
        return results[:limit]

    # ----------------------------------------------------------------
    # MODE 2: AI SEMANTIC SEARCH
    # ----------------------------------------------------------------

    @staticmethod
    def _ai_search(db: Session, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Natural language semantic search.
        Encodes the query with BERT and compares against each candidate's
        resume text + skills. Returns ranked results.
        """
        if not query.strip():
            return []

        rows = db.execute(_CANDIDATE_SQL).fetchall()

        # Pre-compute query embedding once
        query_embedding = None
        keywords = [k for k in re.split(r"[\s,]+", query.lower()) if len(k) > 2]
        try:
            if scoring_engine.encoder:
                query_embedding = scoring_engine.encoder.encode([query])[0]
        except Exception as e:
            logger.warning(f"Could not encode query: {e}")

        scored = []
        for row in rows:
            (user_id, email, first_name, last_name, avatar, profile_pic,
             is_verified, status, created_at,
             profile_id, skills, education, work_exp,
             parse_status, parsed_json, job_match_score, matched_skills_list) = row

            skill_names = _skill_names(skills)
            skill_lower = [s.lower() for s in skill_names]

            # Build resume text corpus for semantic scoring
            resume_text = ""
            if parsed_json and isinstance(parsed_json, dict):
                resume_text = parsed_json.get("_raw_text") or parsed_json.get("text", "") or ""

            if not resume_text:
                # Fall back to skills only if no resume text
                resume_text = " ".join(skill_names)

            if not resume_text:
                continue

            try:
                res = scoring_engine.score_with_embedding(
                    resume_text    = resume_text,
                    jd_description = query,
                    query_embedding= query_embedding,
                    keywords       = keywords,
                    candidate_skills = skill_lower,
                )
                ai_score = float(res.get("score", 0.0))
                matched  = res.get("matched_skills", [])
            except Exception as e:
                logger.warning(f"Scoring error for candidate {user_id}: {e}")
                continue

            if ai_score < 10.0:   # skip very poor matches
                continue

            card = _build_card(row, {
                "aiScore":      round(ai_score, 2),
                "matchedSkills": matched,
                "relevance":    ai_score,
            })
            scored.append(card)

        scored.sort(key=lambda c: c.get("aiScore", 0.0), reverse=True)
        return scored[:limit]

    # ----------------------------------------------------------------
    # LEGACY: kept for backward compatibility (old admin search endpoint)
    # ----------------------------------------------------------------

    @staticmethod
    def search_candidates(db: Session, query: str, user_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Legacy method — routes to AI search for backward compatibility."""
        return SearchService._ai_search(db, query, limit)
