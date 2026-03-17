import re
import logging
from sqlalchemy.orm import Session
from app.models import DBCandidate
from app.scoring import ScoringEngine
from app.config import settings

logger = logging.getLogger(__name__)
engine = ScoringEngine(weight=settings.SCORING_WEIGHT)

class AdminSearchService:
    @staticmethod
    def parse_experience_from_query(query: str) -> float:
        """Extracts numerical years of experience from a query string (handles '2+', '2-5', etc)."""
        # Improved regex to handle "2+", "2-3", "2 years", "over 5"
        match = re.search(r'(\d+)(?:\+|-|\s*to\s*|\s*years?)', query, re.IGNORECASE)
        if match:
            return float(match.group(1))
        # Fallback for just a number near the word "year"
        match = re.search(r'(\d+)\s*(?:years?|yrs?)', query, re.IGNORECASE)
        return float(match.group(1)) if match else 0.0

    @staticmethod
    def search_candidates(db: Session, query: str):
        target_experience = AdminSearchService.parse_experience_from_query(query)
        
        # PRODUCTION OPTIMIZATION: Encode the query once, not for every candidate
        query_embedding = None
        if engine.encoder:
            query_embedding = engine.encoder.encode([query])[0]
            
        # Extract keywords once
        keywords = [k.strip() for k in query.replace(',', ' ').split() if len(k) > 3]
        
        candidates = db.query(DBCandidate).filter(DBCandidate.resume_parse_status == "PARSED").all()
        
        results = []
        for c in candidates:
            # Enhanced scoring using the pre-computed query embedding
            res = engine.score_with_embedding(
                resume_text=c.resume_parsed_json.get("text", ""),
                jd_description=query,
                query_embedding=query_embedding,
                keywords=keywords
            )
            
            # Simple bonus for matching experience if we had it parsed
            # For now, we contribute to the recommendation
            
            results.append({
                "candidate_id": c.id,
                "score": res["score"],
                "recommendation": res["recommendation"],
                "matched_skills": res["matched_skills"],
                "resume_key": c.resume_s3_key
            })
            
        return sorted(results, key=lambda x: x["score"], reverse=True)
