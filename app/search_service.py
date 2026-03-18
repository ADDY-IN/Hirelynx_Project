import re
import logging
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from app.models import DBCandidate, DBJob
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
    def search_candidates(db: Session, query: str) -> List[Dict[str, Any]]:
        """Semantic search for candidates based on a text query."""
        from app.utils import encode_id
        
        query_low = query.lower().strip()

        # 2. Semantic query prep
        query_embedding = None
        if engine.encoder and len(query_low) > 3:
            try:
                query_embedding = engine.encoder.encode([query])[0]
            except Exception:
                pass
            
        keywords = [k.strip() for k in query.replace(',', ' ').split() if len(k) > 3]
        
        # Fetch all candidates via ORM (safe, no transaction issues)
        candidates = db.query(DBCandidate).all()
        
        results = []
        for c in candidates:
            # 1. In-memory name matching (avoids DB transaction issues)
            name_boost = 0.0
            if isinstance(c.personalDetails, dict):
                first = (c.personalDetails.get("firstName") or "").lower()
                last = (c.personalDetails.get("lastName") or "").lower()
                if query_low in first or query_low in last:
                    name_boost = 0.5

            
            c_skills = []
            if isinstance(c.skills, list):
                for s in c.skills:
                    if isinstance(s, dict) and "name" in s:
                        c_skills.append(str(s["name"]))
                    elif isinstance(s, str):
                        c_skills.append(s)

            resume_text = ""
            if c.resumeParsedJson and isinstance(c.resumeParsedJson, dict):
                resume_text = c.resumeParsedJson.get("text", "")

            res = engine.score_with_embedding(
                resume_text=resume_text,
                jd_description=query,
                query_embedding=query_embedding,
                keywords=keywords,
                candidate_skills=c_skills
            )
            
            final_score = float(res["score"]) + name_boost
            
            results.append({
                "candidateId": c.id,
                "candidateToken": encode_id("CAND", c.id),
                "score": round(float(min(1.0, final_score)), 4),
                "recommendation": res["recommendation"],
                "matchedSkills": res["matched_skills"],
                "resumeS3Key": c.resumeS3Key
            })
            
        return sorted(results, key=lambda x: x["score"], reverse=True)
