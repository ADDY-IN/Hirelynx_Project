import re
import logging
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Any, Optional
from app.scoring import ScoringEngine
from app.config import settings
from app.utils import encode_id

logger = logging.getLogger(__name__)
engine = ScoringEngine(weight=settings.SCORING_WEIGHT)

class SearchService:
    @staticmethod
    def search_candidates(db: Session, query: str, user_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search candidates by name, email, or skills.
        Joins candidate_profiles with users table to get real name/email data.
        """
        query_low = query.lower().strip()

        # -- SQL Join: candidate_profiles + users --
        # We pull the key fields needed for search and display.
        sql = text("""
            SELECT
                cp.id,
                cp."userId",
                u."email",
                u."firstName",
                u."lastName",
                cp."personalDetails",
                cp."skills",
                cp."education",
                cp."workExperience",
                cp."projects",
                cp."resumeParseStatus",
                cp."resumeS3Key",
                cp."resumeParsedJson"
            FROM candidate_profiles cp
            LEFT JOIN users u ON cp."userId" = u.id
            ORDER BY cp.id DESC
        """)

        rows = db.execute(sql).fetchall()

        results = []
        for row in rows:
            cid, user_id_val, email, first_name, last_name, personal_details, skills, education, work_exp, projects, parse_status, s3_key, parsed_json = row

            # Build searchable text from available data
            searchable_parts = []

            # Name from users table
            if first_name:
                searchable_parts.append(str(first_name).lower())
            if last_name:
                searchable_parts.append(str(last_name).lower())
            # Email
            if email:
                searchable_parts.append(str(email).lower())
            # Personal details JSON
            if personal_details and isinstance(personal_details, dict):
                searchable_parts.extend([str(v).lower() for v in personal_details.values() if v])
            # Skills JSON
            skill_names = []
            if skills and isinstance(skills, list):
                for s in skills:
                    if isinstance(s, dict) and "name" in s:
                        skill_names.append(str(s["name"]).lower())
                    elif isinstance(s, str):
                        skill_names.append(s.lower())
            searchable_parts.extend(skill_names)
            # Resume text
            resume_text = ""
            if parsed_json and isinstance(parsed_json, dict):
                resume_text = parsed_json.get("text", "") or ""

            searchable_str = " ".join(searchable_parts)

            # -- Matching logic --
            if not query_low:
                # Empty query: return everyone
                results.append({
                    "candidateId": cid,
                    "token": encode_id("CAND", cid),
                    "email": email,
                    "firstName": first_name,
                    "lastName": last_name,
                    "personalDetails": personal_details,
                    "skills": skills,
                    "education": education,
                    "workExperience": work_exp,
                    "projects": projects,
                    "resumeParseStatus": parse_status,
                    "resumeS3Key": s3_key,
                    "score": 0.0,
                    "matchedSkills": [],
                    "recommendation": "All Candidates"
                })
                continue

            # Check direct text match
            score = 0.0
            matched_skills = []
            query_words = query_low.split()

            # Name / email / skill match boost
            if query_low in searchable_str:
                score += 60.0
            else:
                # partial word matching
                for word in query_words:
                    if word in searchable_str:
                        score += 20.0

            # Skill-specific matching
            for sn in skill_names:
                if query_low in sn or any(w in sn for w in query_words):
                    score += 15.0
                    matched_skills.append(sn)

            # Semantic / keyword match against resume text (if available)
            if resume_text and score < 10.0:
                try:
                    keywords = [k for k in query_low.replace(',', ' ').split() if len(k) > 2]
                    query_embedding = None
                    if engine.encoder:
                        query_embedding = engine.encoder.encode([query])[0]
                    res = engine.score_with_embedding(
                        resume_text=resume_text,
                        jd_description=query,
                        query_embedding=query_embedding,
                        keywords=keywords,
                        candidate_skills=[s for s in skill_names]
                    )
                    score = max(score, float(res["score"]))
                    matched_skills = matched_skills or res["matched_skills"]
                except Exception:
                    pass

            if score > 0:
                full_name = f"{first_name or ''} {last_name or ''}".strip()
                results.append({
                    "candidateId": cid,
                    "token": encode_id("CAND", cid),
                    "email": email,
                    "firstName": first_name,
                    "lastName": last_name,
                    "fullName": full_name if full_name else None,
                    "personalDetails": personal_details,
                    "skills": skills,
                    "education": education,
                    "workExperience": work_exp,
                    "projects": projects,
                    "resumeParseStatus": parse_status,
                    "resumeS3Key": s3_key,
                    "score": round(min(1.0, score / 100.0), 4),
                    "matchedSkills": matched_skills,
                    "recommendation": "Strong Match" if score >= 60 else "Partial Match"
                })

        # Sort by score descending, limit results
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:limit]
        return results
