import logging
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Any, Optional
from app.utils import encode_id

logger = logging.getLogger(__name__)

class SearchService:
    @staticmethod
    def search_candidates(db: Session, query: str, user_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search candidates from the users table (role=CANDIDATE).
        Also pulls skills from candidate_profiles for skill-based search.
        All IDs are from the users table.
        """
        query_low = query.lower().strip()

        sql = text("""
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
                cp."resumeParseStatus"
            FROM users u
            LEFT JOIN candidate_profiles cp ON cp."userId" = u.id
            WHERE u.role = 'CANDIDATE'
              AND u."deletedAt" IS NULL
            ORDER BY u.id DESC
        """)

        rows = db.execute(sql).fetchall()

        results = []
        for row in rows:
            (user_id_val, email, first_name, last_name, avatar,
             profile_pic, is_verified, status, created_at,
             profile_id, skills, parse_status) = row

            full_name = f"{first_name or ''} {last_name or ''}".strip()

            # Extract skill names
            skill_names = []
            if skills and isinstance(skills, list):
                for s in skills:
                    if isinstance(s, dict) and "name" in s:
                        skill_names.append(str(s["name"]).lower())
                    elif isinstance(s, str):
                        skill_names.append(s.lower())

            # Searchable string: name + email + skills
            searchable = f"{full_name} {email or ''} {' '.join(skill_names)}".lower()

            # Build result dict
            candidate_data = {
                "userId": user_id_val,
                "candidateId": user_id_val,          # Use userId as the primary identifier
                "profileId": profile_id,              # candidate_profiles.id (may be None)
                "token": encode_id("USER", user_id_val),
                "email": email,
                "firstName": first_name,
                "lastName": last_name,
                "fullName": full_name or None,
                "avatar": avatar or profile_pic,
                "isEmailVerified": is_verified,
                "status": status,
                "createdAt": str(created_at) if created_at else None,
                "skills": skills,
                "resumeParseStatus": parse_status,
            }

            if not query_low:
                candidate_data.update({"score": 0.0, "matchedSkills": [], "recommendation": "All Candidates"})
                results.append(candidate_data)
                continue

            # -- Scoring --
            score = 0.0
            matched_skills = []
            query_words = query_low.split()

            # Word-level match
            for word in query_words:
                if word in searchable:
                    score += 25.0

            # Full phrase bonus
            if query_low in searchable:
                score += 40.0

            # Skill match
            for sn in skill_names:
                for word in query_words:
                    if word in sn or sn in query_low:
                        score += 15.0
                        if sn not in matched_skills:
                            matched_skills.append(sn)

            if score > 0:
                candidate_data.update({
                    "score": round(min(1.0, score / 100.0), 4),
                    "matchedSkills": matched_skills,
                    "recommendation": "Strong Match" if score >= 60 else "Partial Match"
                })
                results.append(candidate_data)

        results = sorted(results, key=lambda x: x["score"], reverse=True)[:limit]
        return results
