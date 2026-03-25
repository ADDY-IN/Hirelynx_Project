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
    """Sum years of work experience from workExperience JSON.
    Handles both full ISO dates (YYYY-MM-DD) and year-only values.
    """
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

        def _parse_decimal_year(s: str) -> float:
            """Extract year as decimal from YYYY-MM-DD or YYYY string."""
            s = s.strip()
            # Full ISO date: YYYY-MM-DD
            iso = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
            if iso:
                return int(iso.group(1)) + (int(iso.group(2)) - 1) / 12.0
            # Year only
            yr = re.search(r"\b(20\d{2}|19\d{2})\b", s)
            if yr:
                return float(yr.group(1))
            return 0.0

        s_dec = _parse_decimal_year(start)
        if s_dec == 0.0:
            continue  # no parseable start date
        if curr:
            e_dec = current_year
        else:
            e_dec = _parse_decimal_year(end)
            if e_dec == 0.0:
                e_dec = s_dec + 1.0  # assume 1-year role if no end date
        total += max(0.0, e_dec - s_dec)
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


def _resolve_personal_details(parsed_json: Any, personal_details_col: Any = None) -> dict:
    """
    Best-effort resolution of personalDetails.
    Priority:
      1. resumeParsedJson (flat or nested under structuredData)
      2. candidate_profiles.personalDetails column (direct JSON column)
    This ensures phone / location are returned even when resumeParsedJson is NULL
    (e.g. candidates whose resumes were parsed by the Node.js backend).
    """
    pd = _get_personal_details(parsed_json)
    if not pd and isinstance(personal_details_col, dict):
        pd = personal_details_col
    return pd if isinstance(pd, dict) else {}


def _extract_location_phone(parsed_json: Any, personal_details_col: Any = None):
    """Extract location string and phone — checks resumeParsedJson then personalDetails column."""
    pd = _resolve_personal_details(parsed_json, personal_details_col)
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
     parse_status, personal_details_col, parsed_json,
     job_match_score, matched_skills_list) = row

    full_name            = f"{first_name or ''} {last_name or ''}".strip() or None
    skill_list           = _skill_names(skills)
    years_exp            = _extract_years(work_exp)
    location, phone      = _extract_location_phone(parsed_json, personal_details_col)

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
        cp."personalDetails",
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
        job_types      = [j.strip().upper().replace("-", "_") for j in (f.get("jobType") or []) if j]
        exp_min        = f.get("experienceMin")
        exp_max        = f.get("experienceMax")
        salary_min     = f.get("salaryMin")
        salary_max     = f.get("salaryMax")
        min_score      = float(f.get("minMatchScore") or 0.0)
        query_low      = (query or "").strip().lower()

        results = []
        for row in rows:
            (user_id, email, first_name, last_name, avatar, profile_pic,
             is_verified, status, created_at,
             profile_id, skills, education, work_exp,
             parse_status, personal_details_col, parsed_json,
             job_match_score, matched_skills_list) = row

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

            # ── Category filter ─────────────────────────────────────────────
            # Uses a keyword map so "IT" matches developers, engineers, etc.
            # Falls back to work experience roles, then resume text.
            if category:
                _CATEGORY_KEYWORDS = {
                    "it": [
                        "developer", "engineer", "software", "web", "frontend", "backend",
                        "fullstack", "full stack", "devops", "cloud", "data", "machine learning",
                        "javascript", "python", "java", "react", "node", "angular", "vue",
                        "typescript", "sql", "database", "api", "mobile", "ios", "android",
                        "qa", "testing", "cyber", "security", "network", "sysadmin", "linux",
                        "aws", "azure", "gcp", "docker", "kubernetes", "git", "programming",
                        "technology", "computer science", "information technology",
                    ],
                    "chef": ["chef", "cook", "culinary", "kitchen", "food", "pastry", "baker"],
                    "driver": ["driver", "cdl", "truck", "delivery", "logistics", "transportation"],
                    "healthcare": ["nurse", "doctor", "medical", "healthcare", "clinical", "hospital", "pharmacy"],
                    "finance": ["accountant", "finance", "banking", "cpa", "audit", "tax", "financial"],
                    "hr": ["human resources", "hr", "recruiter", "talent", "payroll"],
                }
                kw_list = _CATEGORY_KEYWORDS.get(category, [category])

                # Check 1: structured skill names
                category_hit = any(
                    any(kw in skill for kw in kw_list)
                    for skill in skill_lower
                )

                # Check 2: work experience roles / job titles
                if not category_hit and work_exp and isinstance(work_exp, list):
                    role_text = " ".join(
                        f"{exp.get('role','')} {exp.get('jobTitle','')}".lower()
                        for exp in work_exp if isinstance(exp, dict)
                    )
                    category_hit = any(kw in role_text for kw in kw_list)

                # Check 3: resume text snippet
                if not category_hit and parsed_json and isinstance(parsed_json, dict):
                    resume_head = (
                        parsed_json.get("text") or parsed_json.get("_raw_text") or ""
                    )[:1000].lower()
                    category_hit = any(kw in resume_head for kw in kw_list)

                if not category_hit:
                    continue

            # ── Location filter ───────────────────────────────────
            if locations:
                pd_loc = _resolve_personal_details(parsed_json, personal_details_col)
                candidate_location = (
                    f"{pd_loc.get('city','')} {pd_loc.get('province','')} "
                    f"{pd_loc.get('location','')}"
                ).lower().strip()
                remote_wanted = any("remote" in l for l in locations)
                city_locations = [l for l in locations if "remote" not in l]
                city_matched = bool(city_locations) and any(
                    loc in candidate_location for loc in city_locations
                )
                if not city_matched and not remote_wanted:
                    continue
                # If remote_wanted → pass through regardless; if city_matched → also pass

            # ── Skills filter (candidate must have ALL required skills) ───
            if req_skills:
                # Build a broad text blob for skill matching:
                # structured skills list + resume text (catches skills buried in bullet points)
                resume_text_blob = ""
                if parsed_json and isinstance(parsed_json, dict):
                    resume_text_blob = (
                        parsed_json.get("text") or
                        parsed_json.get("_raw_text") or
                        parsed_json.get("structuredData", {}).get("text", "")
                        if isinstance(parsed_json.get("structuredData"), dict) else ""
                    ) or ""
                resume_text_lower = resume_text_blob.lower()
                if not all(
                    any(rs in sl or sl in rs for sl in skill_lower)   # structured skills
                    or rs in resume_text_lower                          # resume text fallback
                    for rs in req_skills
                ):
                    continue

            # ── Experience range filter ──────────────────────────────
            if exp_min is not None and years_exp < exp_min:
                continue
            if exp_max is not None and years_exp > exp_max:
                continue

            # ── Salary filter (based on candidate's resume parsed salary expectation) ─
            # NOTE: candidate_profiles doesn't store salary expectations directly.
            # We skip this filter silently — salary is a job-side field.
            # (salary_min / salary_max are accepted in the request for forward-compat)

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

            # ── Text query filter (name / email / skill / location / resume) ─
            relevance = 1.0
            if query_low:
                # Resolve personalDetails: resumeParsedJson first, then personalDetails column
                pd_info           = _resolve_personal_details(parsed_json, personal_details_col)
                candidate_loc_str = (
                    f"{pd_info.get('city','')} {pd_info.get('province','')} "
                    f"{pd_info.get('location','')}"
                ).lower().strip()
                candidate_phone = (pd_info.get("phone") or "").lower()

                # Build resume text for query matching
                resume_snippet = ""
                if parsed_json and isinstance(parsed_json, dict):
                    resume_snippet = (
                        parsed_json.get("text") or
                        parsed_json.get("_raw_text") or ""
                    )[:3000].lower()

                # Also pull work experience roles/titles
                role_text = ""
                if work_exp and isinstance(work_exp, list):
                    for exp in work_exp:
                        if isinstance(exp, dict):
                            role_text += f" {exp.get('role','')} {exp.get('jobTitle','')}".lower()

                text_blob = (
                    f"{full_name} {email_str} {' '.join(skill_lower)} "
                    f"{candidate_loc_str} {candidate_phone} "
                    f"{role_text} {resume_snippet}"
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

        Also extracts location hints (e.g. 'from Toronto', 'in Alberta') from the
        query text and applies a hard location filter on top of the semantic score.
        """
        if not query.strip():
            return []

        rows = db.execute(_CANDIDATE_SQL).fetchall()

        # ── Extract location filter from natural-language query ───────────
        # Matches: "from Toronto", "in British Columbia", "near Vancouver", etc.
        query_lower = query.lower()
        loc_match = re.search(
            r"\b(?:from|in|near|based in|located in|living in)\s+([a-z][a-z\s]{2,30}?)(?:\s+(?:with|who|that|and|,)|$)",
            query_lower,
        )
        location_filter: Optional[str] = None
        if loc_match:
            location_filter = loc_match.group(1).strip()
            logger.info(f"AI search: extracted location filter = '{location_filter}'")

        # Pre-compute query embedding once
        query_embedding = None
        keywords = [k for k in re.split(r"[\s,]+", query_lower) if len(k) > 2]
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
             parse_status, personal_details_col, parsed_json,
             job_match_score, matched_skills_list) = row

            skill_names_list = _skill_names(skills)
            skill_lower = [s.lower() for s in skill_names_list]

            # ── Location hard-filter ──────────────────────────────────────
            if location_filter:
                pd_info = _resolve_personal_details(parsed_json, personal_details_col)
                cand_loc = (
                    f"{pd_info.get('city','')} {pd_info.get('province','')} "
                    f"{pd_info.get('location','')}"
                ).lower().strip()
                if location_filter not in cand_loc:
                    continue  # skip candidates not in specified location

            # Build resume text corpus for semantic scoring
            resume_text = ""
            if parsed_json and isinstance(parsed_json, dict):
                resume_text = parsed_json.get("_raw_text") or parsed_json.get("text", "") or ""

            if not resume_text:
                # Fall back to skills only if no resume text
                resume_text = " ".join(skill_names_list)

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
    # SUGGESTIONS: dynamic AI-search prompt chips
    # ----------------------------------------------------------------

    @staticmethod
    def get_suggestions(db: Session, count: int = 6) -> List[str]:
        """
        Build dynamic suggestion strings from live DB data.
        Pulls top skills and top cities from candidate_profiles,
        generates varied natural-language prompt strings, and
        returns `count` randomly-sampled results.
        """
        import random

        _SKILLS_SQL = text("""
            SELECT elem->>'name' AS skill_name, COUNT(*) AS cnt
            FROM candidate_profiles,
                 jsonb_array_elements(
                     CASE
                         WHEN skills IS NOT NULL
                          AND jsonb_typeof(skills::jsonb) = 'array'
                         THEN skills::jsonb
                         ELSE '[]'::jsonb
                     END
                 ) AS elem
            WHERE skills IS NOT NULL
              AND (elem->>'name') IS NOT NULL
              AND (elem->>'name') <> ''
            GROUP BY skill_name
            ORDER BY cnt DESC
            LIMIT 15
        """)

        _CITIES_SQL = text("""
            SELECT
                COALESCE(
                    NULLIF(TRIM("personalDetails"->>'city'), ''),
                    NULLIF(TRIM("personalDetails"->>'province'), '')
                ) AS city,
                COUNT(*) AS cnt
            FROM candidate_profiles
            WHERE "personalDetails" IS NOT NULL
              AND COALESCE(
                    NULLIF(TRIM("personalDetails"->>'city'), ''),
                    NULLIF(TRIM("personalDetails"->>'province'), '')
                  ) IS NOT NULL
            GROUP BY city
            ORDER BY cnt DESC
            LIMIT 10
        """)

        try:
            skill_rows = db.execute(_SKILLS_SQL).fetchall()
            city_rows  = db.execute(_CITIES_SQL).fetchall()
        except Exception as e:
            logger.warning(f"Suggestions query failed: {e}")
            return []

        skills = [r[0] for r in skill_rows if r[0]]
        cities  = [r[0] for r in city_rows  if r[0]]

        if not skills:
            return []

        templates: List[str] = []

        if cities:
            for skill in skills[:8]:
                city = random.choice(cities)
                templates += [
                    f"Find {skill} developers in {city}",
                    f"Show me {skill} candidates from {city}",
                    f"Find candidates with {skill} skills in {city}",
                    f"Top {skill} professionals in {city}",
                ]

        for skill in skills[:10]:
            exp = random.choice(["2+", "3+", "5+"])
            templates += [
                f"{skill} specialists with {exp} years experience",
                f"Find experienced {skill} professionals",
                f"Show me {skill} candidates available for remote work",
                f"Find {skill} engineers open to relocation",
            ]

        if len(skills) >= 2:
            pairs = list(zip(skills[:5], skills[5:10] or skills[:5]))
            for s1, s2 in pairs:
                templates.append(f"Candidates with both {s1} and {s2} experience")

        for city in cities[:5]:
            templates.append(f"Show all candidates located in {city}")

        seen: set = set()
        unique: List[str] = []
        for t in templates:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique.append(t)

        random.shuffle(unique)
        return unique[:count]

    # ----------------------------------------------------------------
    # LEGACY: kept for backward compatibility (old admin search endpoint)
    # ----------------------------------------------------------------

    @staticmethod
    def search_candidates(db: Session, query: str, user_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Legacy method — routes to AI search for backward compatibility."""
        return SearchService._ai_search(db, query, limit)
