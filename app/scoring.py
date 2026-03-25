"""
Scoring Engine
==============
Multi-factor resume-to-job scoring combining:
  1. Skills Match           (50%) — required job skills vs candidate skills
  2. Experience Match       (20%) — candidate years vs job's min/max requirement
  3. Education Match        (10%) — degree tier vs implied job education level
  4. Responsibilities Match (20%) — job responsibilities vs candidate work responsibilities
                                     (semantic cosine similarity via SentenceTransformer)

All four sub-scores are 0–100 and combined into a single jobMatchScore (float 0–100).

The original `score` / `score_with_embedding` methods are retained for backward
compatibility with existing batch-matching endpoints.
"""
import re
import logging
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Education tier: higher tier = higher qualification
# ---------------------------------------------------------------------------
_EDU_TIERS: Dict[str, int] = {
    "high school": 1, "secondary": 1, "ged": 1, "diploma": 2,
    "certificate": 2, "trade": 2, "red seal": 3, "journeyman": 3,
    "associate": 3, "college diploma": 3,
    "bachelor": 4, "b.sc": 4, "b.tech": 4, "b.eng": 4,
    "b.com": 4, "b.a": 4, "undergraduate": 4,
    "master": 5, "m.sc": 5, "m.eng": 5, "mba": 5, "graduate": 5,
    "phd": 6, "doctorate": 6, "md": 6, "llb": 4, "llm": 5,
    "postgraduate": 5,
}

def _degree_tier(degree: str) -> int:
    """Return the education tier for a degree string (higher = more qualified)."""
    d = degree.lower()
    best = 0
    for kw, tier in _EDU_TIERS.items():
        if kw in d:
            best = max(best, tier)
    return best or 1  # default to 1 if unknown


def _infer_required_edu_tier(job_description: str, responsibilities: List[str]) -> int:
    """
    Infer the minimum education tier the job implicitly requires based on
    keywords in its description or responsibilities.
    Returns 0 if no education requirement is detectable (score will be 100).
    """
    text = (job_description + " " + " ".join(responsibilities)).lower()
    if any(kw in text for kw in ["phd", "doctorate", "doctoral"]):
        return 6
    if any(kw in text for kw in ["master's", "masters degree", "mba", "m.sc", "m.eng"]):
        return 5
    if any(kw in text for kw in ["bachelor", "degree required", "undergraduate",
                                   "b.sc", "b.tech", "b.eng", "b.com"]):
        return 4
    if any(kw in text for kw in ["diploma", "college diploma", "certificate required",
                                   "trade certificate", "red seal", "journeyman"]):
        return 2
    return 0  # no education requirement detectable


def _extract_candidate_years(work_experience: List[Dict]) -> float:
    """Calculate total years of professional experience from structured work history."""
    total = 0.0
    current_year = 2026
    for exp in work_experience:
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
    return total


def _extract_candidate_responsibilities(work_experience: List[Dict]) -> List[str]:
    """
    Extract all responsibility bullet points from a candidate's work experience.
    Flattens the list of lists into a single deduplicated list of non-empty strings.
    """
    seen: set = set()
    result: List[str] = []
    for exp in work_experience:
        if not isinstance(exp, dict):
            continue
        for r in exp.get("responsibilities", []):
            r = str(r).strip()
            if r and r.lower() not in seen:
                seen.add(r.lower())
                result.append(r)
    return result


def _extract_candidate_skills(parsed_json: Dict, resume_text: str) -> List[str]:
    """Extract candidate skills from parsed JSON or fall back to raw text parsing."""
    skills: List[str] = []
    raw_skills = parsed_json.get("skills", [])
    if isinstance(raw_skills, list):
        for s in raw_skills:
            if isinstance(s, dict):
                skills.append(s.get("name", ""))
            elif isinstance(s, str):
                skills.append(s)
    return [s.strip() for s in skills if s.strip()]


def _match_skills(required: List[str], candidate_skills: List[str],
                  resume_text: str) -> Tuple[List[str], List[str]]:
    """
    Match required skills against candidate's profile.
    Uses exact match → fuzzy match (threshold 85) → substring in resume text.
    Returns (matched, missing).
    """
    resume_lower = resume_text.lower()
    c_lower = [s.lower() for s in candidate_skills]
    matched, missing = [], []

    for req in required:
        req_low = req.lower().strip()
        found = False

        # 1. Exact match in structured skills
        if req_low in c_lower:
            found = True

        # 2. Fuzzy match in structured skills (handles slight variations)
        if not found:
            for cs in c_lower:
                if fuzz.ratio(req_low, cs) >= 88:
                    found = True
                    break

        # 3. Substring in full resume text (catches skills mentioned in responsibilities)
        if not found and req_low in resume_lower:
            found = True

        (matched if found else missing).append(req)

    return matched, missing


class ScoringEngine:
    """
    Advanced Resume-to-Job Scoring Engine.

    Multi-factor scoring for the new `/v1/scoring/match-resume` endpoint.
    Backward-compatible methods retained for batch matching.
    """

    # Weights for the four factors (must sum to 1.0)
    WEIGHT_SKILLS  = 0.50
    WEIGHT_EXP     = 0.15
    WEIGHT_EDU     = 0.05
    WEIGHT_RESP    = 0.30

    def __init__(self, weight: float = 0.5):
        self.weight = weight   # legacy: used by score_with_embedding
        try:
            logger.info("Loading SentenceTransformer model...")
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.encoder = None

    # ----------------------------------------------------------------
    # NEW: Multi-factor scoring from parsed resume data
    # ----------------------------------------------------------------

    def _match_responsibilities(
        self,
        job_responsibilities: List[str],
        candidate_responsibilities: List[str],
    ) -> float:
        """
        Semantic similarity between job responsibilities and candidate responsibilities.
        Uses the SentenceTransformer encoder (already loaded) to compute sentence embeddings
        and then computes mean max-cosine-similarity: for each job responsibility, find the
        best-matching candidate responsibility, then average across all job responsibilities.
        Returns a score 0–100. Falls back to 0 if encoder unavailable or lists are empty.
        """
        if not job_responsibilities or not candidate_responsibilities or self.encoder is None:
            return 0.0
        try:
            job_embs  = self.encoder.encode(job_responsibilities,  convert_to_numpy=True, show_progress_bar=False)
            cand_embs = self.encoder.encode(candidate_responsibilities, convert_to_numpy=True, show_progress_bar=False)
            # sim matrix: (n_job, n_cand)
            sim_matrix = cosine_similarity(job_embs, cand_embs)
            # For each job responsibility, pick the best candidate match
            best_per_job = sim_matrix.max(axis=1)
            score = float(np.mean(best_per_job)) * 100
            return round(max(0.0, min(100.0, score)), 2)
        except Exception as e:
            logger.warning(f"Responsibilities semantic match error: {e}")
            return 0.0

    def score_resume_against_job(
        self,
        parsed_json: Dict,
        resume_text: str,
        required_skills: List[str],
        exp_min: Optional[float],
        exp_max: Optional[float],
        job_description: str,
        job_responsibilities: List[str],
        candidate_responsibilities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Four-factor scoring:
          - Skills Match           (50%): required skills vs candidate skills
          - Experience             (20%): candidate years vs job requirement
          - Education              (10%): candidate's highest degree vs job's implied minimum
          - Responsibilities Match (20%): semantic similarity between job responsibilities
                                          and candidate's work history responsibilities

        Returns a dict with: score, jobMatchScore, matchedSkillsList,
        missingSkills, totalRequired, breakdown, recommendation.
        """

        # ── Candidate data extraction ──────────────────────────────────────
        c_skills      = _extract_candidate_skills(parsed_json, resume_text)
        c_years       = _extract_candidate_years(parsed_json.get("workExperience", []))
        edu_list      = parsed_json.get("education", [])
        c_degrees     = [
            e.get("degree", "") for e in edu_list
            if isinstance(e, dict) and e.get("degree")
        ]
        c_edu_tier = max((_degree_tier(d) for d in c_degrees), default=0)

        # Build candidate responsibilities if not provided — extract from workExperience
        if candidate_responsibilities is None:
            candidate_responsibilities = _extract_candidate_responsibilities(
                parsed_json.get("workExperience", [])
            )

        # ── 1. Skills Score ────────────────────────────────────────────────
        if required_skills:
            matched, missing = _match_skills(required_skills, c_skills, resume_text)
            skills_score = round(len(matched) / len(required_skills) * 100, 2)
        else:
            matched, missing = [], []
            skills_score = 100.0   # no requirements = full score

        # ── 2. Experience Score ────────────────────────────────────────────
        if exp_min is not None and exp_min > 0:
            exp_score = min(100.0, round((c_years / exp_min) * 100, 2))
        elif exp_max is not None and exp_max > 0:
            exp_score = min(100.0, round((c_years / exp_max) * 100, 2))
        else:
            exp_score = 100.0   # no experience requirement stated

        # ── 3. Education Score ─────────────────────────────────────────────
        req_edu_tier = _infer_required_edu_tier(job_description, job_responsibilities)
        if req_edu_tier == 0:
            edu_score = 100.0    # job has no implicit education requirement
        elif c_edu_tier == 0:
            edu_score = 50.0     # no degree found in resume — partial credit
        elif c_edu_tier >= req_edu_tier:
            edu_score = 100.0
        else:
            edu_score = round((c_edu_tier / req_edu_tier) * 100, 2)

        # ── 4. Responsibilities Score ──────────────────────────────────────
        if job_responsibilities and candidate_responsibilities:
            resp_score = self._match_responsibilities(job_responsibilities, candidate_responsibilities)
        elif not job_responsibilities:
            resp_score = 100.0   # job posted no responsibilities — no penalty
        else:
            resp_score = 0.0     # job has responsibilities but candidate has none listed

        # ── Composite Score ────────────────────────────────────────────────
        final = round(
            skills_score * self.WEIGHT_SKILLS +
            exp_score    * self.WEIGHT_EXP    +
            edu_score    * self.WEIGHT_EDU    +
            resp_score   * self.WEIGHT_RESP,
            2
        )

        recommendation = (
            "Excellent Match" if final >= 85 else
            "Strong Match"    if final >= 70 else
            "Good Match"      if final >= 55 else
            "Potential Match" if final >= 40 else
            "Low Match"
        )

        return {
            "score":             final,
            "jobMatchScore":     final,
            "matchedSkillsList": matched,
            "missingSkills":     missing,
            "totalRequired":     len(required_skills),
            "recommendation":    recommendation,
            "breakdown": {
                "skillsScore":              skills_score,
                "experienceScore":          exp_score,
                "educationScore":           edu_score,
                "responsibilitiesScore":    resp_score,
                "candidateYears":           c_years,
                "requiredYearsMin":         exp_min,
                "requiredYearsMax":         exp_max,
                "candidateEduTier":         c_edu_tier,
                "requiredEduTier":          req_edu_tier,
                "jobResponsibilitiesCount": len(job_responsibilities),
                "candidateRespCount":       len(candidate_responsibilities),
            },
        }

    # ----------------------------------------------------------------
    # AI: Groq LLM scoring (primary) with rule-based fallback
    # ----------------------------------------------------------------

    def _groq_score(
        self,
        resume_text: str,
        job_description: str,
        required_skills: List[str],
        exp_min: Optional[float],
        exp_max: Optional[float],
        job_responsibilities: List[str],
        job_title: str = "",
        candidate_responsibilities: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Call Groq LLM to score a resume against a job.
        Returns parsed result dict on success, None on failure.
        """
        try:
            from groq import Groq
            from app.config import settings
            import json as _json

            api_key = getattr(settings, "GROQ_API_KEY", None)
            if not api_key:
                logger.warning("GROQ_API_KEY not set — skipping AI scoring")
                return None

            # Truncate resume/JD to avoid token limits (~6000 chars each)
            resume_snippet = resume_text[:6000].strip()
            jd_snippet = job_description[:3000].strip()
            skills_str = ", ".join(required_skills) if required_skills else "Not specified"
            exp_str = (
                f"{exp_min}–{exp_max} years" if exp_min and exp_max else
                f"{exp_min}+ years" if exp_min else
                f"up to {exp_max} years" if exp_max else
                "Not specified"
            )
            resp_str = "\n".join(f"- {r}" for r in job_responsibilities[:15]) or "Not specified"

            # Extract candidate responsibilities from their work experience for the prompt
            cand_resp_list = candidate_responsibilities or []
            cand_resp_str = (
                "\n".join(f"- {r}" for r in cand_resp_list[:20])
                if cand_resp_list else "Not specified"
            )

            prompt = f"""You are an expert recruiter and HR analyst. Your job is to score how well this candidate's resume matches the job posting.

Be STRICT and REALISTIC. Differentiate candidates based on actual skill alignment, experience depth, and role fit.
Do NOT give scores above 80 unless the candidate is a near-perfect match.

JOB TITLE: {job_title or "Not specified"}
JOB DESCRIPTION:
{jd_snippet}

REQUIRED SKILLS: {skills_str}
EXPERIENCE REQUIRED: {exp_str}
KEY JOB RESPONSIBILITIES:
{resp_str}

CANDIDATE RESUME:
{resume_snippet}

CANDIDATE WORK RESPONSIBILITIES (extracted from their past roles):
{cand_resp_str}

Return ONLY valid JSON with no markdown, no extra text — just the raw JSON object:
{{
  "jobMatchScore": <integer 0-100>,
  "matchedSkills": ["skill1", "skill2"],
  "missingSkills": ["skill3", "skill4"],
  "breakdown": {{
    "skillsScore": <integer 0-100>,
    "experienceScore": <integer 0-100>,
    "educationScore": <integer 0-100>,
    "responsibilitiesScore": <integer 0-100>,
    "overallReasoning": "<1-2 sentence explanation of the score>"
  }},
  "recommendation": "<one of: Excellent Match, Strong Match, Good Match, Potential Match, Low Match>"
}}"""

            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,   # low temp = consistent, deterministic scoring
                max_tokens=512,
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if model adds them despite instructions
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            data = _json.loads(raw)

            matched = data.get("matchedSkills", [])
            missing = data.get("missingSkills", [])
            breakdown = data.get("breakdown", {})

            # ── Recompute final score from sub-scores using OUR weights ──────
            # Groq generates jobMatchScore independently from the sub-scores,
            # making them inconsistent. We discard Groq's raw jobMatchScore and
            # recompute it deterministically so the weights actually take effect.
            s_score = float(breakdown.get("skillsScore", 0))
            e_score = float(breakdown.get("experienceScore", 0))
            edu_score = float(breakdown.get("educationScore", 0))
            r_score = float(breakdown.get("responsibilitiesScore", 0))

            score = round(
                s_score   * self.WEIGHT_SKILLS +
                r_score   * self.WEIGHT_RESP   +
                e_score   * self.WEIGHT_EXP    +
                edu_score * self.WEIGHT_EDU,
                2
            )

            recommendation = data.get("recommendation", "")
            # Validate recommendation label against the RECOMPUTED score
            valid_labels = {"Excellent Match", "Strong Match", "Good Match", "Potential Match", "Low Match"}
            if recommendation not in valid_labels:
                recommendation = (
                    "Excellent Match" if score >= 85 else
                    "Strong Match"    if score >= 70 else
                    "Good Match"      if score >= 55 else
                    "Potential Match" if score >= 40 else
                    "Low Match"
                )

            logger.info(f"Groq scoring complete (recomputed from sub-scores): score={score}, recommendation={recommendation}")

            return {
                "score":             score,
                "jobMatchScore":     score,
                "matchedSkillsList": matched,
                "missingSkills":     missing,
                "totalRequired":     len(required_skills),
                "recommendation":    recommendation,
                "breakdown": {
                    "skillsScore":           breakdown.get("skillsScore", 0),
                    "experienceScore":       breakdown.get("experienceScore", 0),
                    "educationScore":        breakdown.get("educationScore", 0),
                    "responsibilitiesScore": breakdown.get("responsibilitiesScore", 0),
                    "overallReasoning":      breakdown.get("overallReasoning", ""),
                    "scoredBy":              "groq-llm",
                },
            }

        except Exception as e:
            logger.warning(f"Groq scoring error: {e}")
            return None

    def score_resume_against_job_ai(
        self,
        parsed_json: Dict,
        resume_text: str,
        required_skills: List[str],
        exp_min: Optional[float],
        exp_max: Optional[float],
        job_description: str,
        job_responsibilities: List[str],
        job_title: str = "",
        candidate_responsibilities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Primary entry-point for on-demand resume scoring.
        Uses Groq LLM for intelligent, context-aware scoring.
        Automatically falls back to rule-based scoring if Groq is unavailable.
        """
        # Pre-extract candidate responsibilities once — used by both Groq and fallback
        if candidate_responsibilities is None:
            candidate_responsibilities = _extract_candidate_responsibilities(
                parsed_json.get("workExperience", [])
            )

        result = self._groq_score(
            resume_text=resume_text,
            job_description=job_description,
            required_skills=required_skills,
            exp_min=exp_min,
            exp_max=exp_max,
            job_responsibilities=job_responsibilities,
            job_title=job_title,
            candidate_responsibilities=candidate_responsibilities,
        )
        if result:
            return result

        # Fallback: rule-based multi-factor scoring
        logger.info("Falling back to rule-based scoring")
        fallback = self.score_resume_against_job(
            parsed_json=parsed_json,
            resume_text=resume_text,
            required_skills=required_skills,
            exp_min=exp_min,
            exp_max=exp_max,
            job_description=job_description,
            job_responsibilities=job_responsibilities,
            candidate_responsibilities=candidate_responsibilities,
        )
        # Tag the fallback so it's identifiable in the response
        fallback.setdefault("breakdown", {})["scoredBy"] = "rule-based-fallback"
        return fallback

    # ----------------------------------------------------------------
    # LEGACY: kept for backward compatibility with batch-matching
    # ----------------------------------------------------------------

    def score(self, resume_text: str, jd_description: str, keywords: List[str]) -> Dict[str, Any]:
        return self.score_with_embedding(resume_text, jd_description, None, keywords, [])

    def score_with_embedding(
        self,
        resume_text: str,
        jd_description: str,
        query_embedding: Any,
        keywords: List[str],
        candidate_skills: List[str] = [],
    ) -> Dict[str, Any]:
        from app.utils import clean_keywords

        resume_lower = resume_text.lower()
        matched: List[str] = []

        if keywords:
            for k in keywords:
                k_low = k.lower()
                found = False
                if candidate_skills and any(k_low == cs.lower() for cs in candidate_skills):
                    found = True
                if not found and fuzz.partial_ratio(k_low, resume_lower) > 85:
                    found = True
                if found:
                    matched.append(k)

        matched = clean_keywords(matched)
        kw_score = (len(matched) / len(keywords) * 100) if keywords else 0

        sem_score = 0.0
        if self.encoder and resume_text:
            try:
                resume_embedding = self.encoder.encode([resume_text])[0]
                if query_embedding is None and jd_description:
                    target_embedding = self.encoder.encode([jd_description])[0]
                else:
                    target_embedding = query_embedding

                if target_embedding is not None:
                    sim = cosine_similarity(
                        np.array([resume_embedding]), np.array([target_embedding])
                    )[0][0]
                    sem_score = max(0.0, float(sim) * 100)
            except Exception as e:
                logger.error(f"Semantic scoring error: {e}")

        if not keywords:
            kw_score = 0.0

        final = round((kw_score * self.weight) + (sem_score * (1 - self.weight)), 2)

        return {
            "score": final,
            "matched_skills": matched,
            "recommendation": (
                "Strong Match"    if final > 80 else
                "Good Match"      if final > 60 else
                "Potential Match" if final > 40 else
                "Low Match"
            ),
        }
