import re
import logging
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from app.models import DBCandidate
from app.scoring import ScoringEngine
from app.config import settings

logger = logging.getLogger(__name__)
engine = ScoringEngine(weight=settings.SCORING_WEIGHT)

class SearchService:
    @staticmethod
    def parse_experience_from_query(query: str) -> float:
        """Extracts numerical years of experience from a query string."""
        match = re.search(r'(\d+)(?:\+|-|\s*to\s*|\s*years?)', query, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r'(\d+)\s*(?:years?|yrs?)', query, re.IGNORECASE)
        return float(match.group(1)) if match else 0.0

    @staticmethod
    def search_candidates(db: Session, query: str, user_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Optimized two-step semantic search for candidates.
        Step 1: Fast ranking using minimal fields.
        Step 2: Enrichment of top candidates with full profile data.
        """
        from app.utils import encode_id
        from sqlalchemy.orm import load_only
        
        query_low = query.lower().strip()
        keywords = [k.strip() for k in query.replace(',', ' ').split() if len(k.strip()) > 3]

        # Pre-encode query ONCE
        query_embedding = None
        if engine.encoder and len(query_low) > 3:
            try:
                query_embedding = engine.encoder.encode([query])[0]
            except Exception:
                pass

        # Step 1: Fast Ranking (Fetch only needed fields)
        search_query = db.query(DBCandidate).options(load_only(
            DBCandidate.id, 
            DBCandidate.userId, 
            DBCandidate.personalDetails, 
            DBCandidate.skills, 
            DBCandidate.resumeParsedJson,
            DBCandidate.resumeS3Key
        ))
        
        if user_id:
            candidates = search_query.filter(DBCandidate.userId == user_id).all()
        else:
            candidates = search_query.all()
        
        scored_results = []
        for c in candidates:
            # Fast in-memory name matching
            name_boost = 0.0
            if isinstance(c.personalDetails, dict):
                first = (c.personalDetails.get("firstName") or "").lower()
                last = (c.personalDetails.get("lastName") or "").lower()
                if query_low in first or query_low in last:
                    name_boost = 0.5

            # Extract candidate skills
            c_skills = []
            if isinstance(c.skills, list):
                for s in c.skills:
                    if isinstance(s, dict) and "name" in s:
                        c_skills.append(str(s["name"]))
                    elif isinstance(s, str):
                        c_skills.append(s)

            resume_text = ""
            if c.resumeParsedJson and isinstance(c.resumeParsedJson, dict):
                resume_text = c.resumeParsedJson.get("text", "") or ""

            # Skip if no content to score
            if not resume_text and not name_boost:
                continue

            try:
                res = engine.score_with_embedding(
                    resume_text=resume_text,
                    jd_description=query,
                    query_embedding=query_embedding,
                    keywords=keywords,
                    candidate_skills=c_skills
                )
                final_score = float(res["score"]) + name_boost
            except Exception:
                final_score = name_boost
                res = {"matched_skills": [], "recommendation": "Unknown"}
            
            scored_results.append({
                "id": c.id,
                "score": round(min(1.0, final_score / 100.0), 4),
                "recommendation": res["recommendation"],
                "matchedSkills": res["matched_skills"]
            })
            
        # Sort and take top K for enrichment
        ranked_results = sorted(scored_results, key=lambda x: x["score"], reverse=True)[:limit]
        top_ids = [r["id"] for r in ranked_results]

        if not top_ids:
            return []

        # Step 2: Enrichment (Fetch full data for top candidates)
        full_profiles = db.query(DBCandidate).filter(DBCandidate.id.in_(top_ids)).all()
        profile_map = {p.id: p for p in full_profiles}

        # Step 3: Data Mapping
        enriched_results = []
        for r in ranked_results:
            p = profile_map.get(r["id"])
            if not p:
                continue
                
            enriched_results.append({
                "candidateId": p.id,
                "userId": p.userId,
                "personalDetails": p.personalDetails,
                "education": p.education,
                "workExperience": p.workExperience,
                "skills": p.skills,
                "projects": p.projects,
                "candidateToken": encode_id("CAND", p.id),
                "score": r["score"],
                "recommendation": r["recommendation"],
                "matchedSkills": r["matchedSkills"],
                "resumeS3Key": p.resumeS3Key,
                "resumeParseStatus": p.resumeParseStatus
            })

        return enriched_results
