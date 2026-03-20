"""
Scoring Engine
==============
Multi-factor resume-to-job scoring combining:
  1. Skills Match    (60%) — required job skills vs candidate skills
  2. Experience Match (25%) — candidate years vs job's min/max requirement
  3. Education Match  (15%) — degree tier vs implied job education level

All three sub-scores are 0–100 and combined into a single jobMatchScore (float 0–100).

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

    # Weights for the three factors (must sum to 1.0)
    WEIGHT_SKILLS  = 0.60
    WEIGHT_EXP     = 0.25
    WEIGHT_EDU     = 0.15

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

    def score_resume_against_job(
        self,
        parsed_json: Dict,
        resume_text: str,
        required_skills: List[str],
        exp_min: Optional[float],
        exp_max: Optional[float],
        job_description: str,
        job_responsibilities: List[str],
    ) -> Dict[str, Any]:
        """
        Three-factor scoring:
          - Skills Match   (60%): required skills vs candidate skills
          - Experience     (25%): candidate years vs job requirement
          - Education      (15%): candidate's highest degree vs job's implied minimum

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

        # ── 1. Skills Score ────────────────────────────────────────────────
        if required_skills:
            matched, missing = _match_skills(required_skills, c_skills, resume_text)
            skills_score = round(len(matched) / len(required_skills) * 100, 2)
        else:
            matched, missing = [], []
            skills_score = 100.0   # no requirements = full score

        # ── 2. Experience Score ────────────────────────────────────────────
        if exp_min is not None and exp_min > 0:
            # Score ramps from 0% at 0 years to 100% at exp_min
            # Bonus above exp_max is ignored — capped at 100
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
            # Partial credit proportional to how close they are
            edu_score = round((c_edu_tier / req_edu_tier) * 100, 2)

        # ── Composite Score ────────────────────────────────────────────────
        final = round(
            skills_score  * self.WEIGHT_SKILLS +
            exp_score     * self.WEIGHT_EXP    +
            edu_score     * self.WEIGHT_EDU,
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
                "skillsScore":      skills_score,
                "experienceScore":  exp_score,
                "educationScore":   edu_score,
                "candidateYears":   c_years,
                "requiredYearsMin": exp_min,
                "requiredYearsMax": exp_max,
                "candidateEduTier": c_edu_tier,
                "requiredEduTier":  req_edu_tier,
            },
        }

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
