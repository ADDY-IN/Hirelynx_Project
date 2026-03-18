import re
import logging
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
        query_embedding = None
        if engine.encoder:
            query_embedding = engine.encoder.encode([query])[0]
            
        keywords = [k.strip() for k in query.replace(',', ' ').split() if len(k) > 3]
        candidates = db.query(DBCandidate).all()
        
        results = []
        for c in candidates:
            # Extract skills for structured matching
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
            
            results.append({
                "candidateId": c.id,
                "score": res["score"],
                "recommendation": res["recommendation"],
                "matchedSkills": res["matched_skills"],
                "resumeS3Key": c.resumeS3Key
            })
            
        return sorted(results, key=lambda x: x["score"], reverse=True)

    @staticmethod
    def search_jobs(db: Session, query: str) -> List[Dict[str, Any]]:
        """Semantic search for jobs based on a text query (for candidates)."""
        query_embedding = None
        if engine.encoder:
            query_embedding = engine.encoder.encode([query])[0]
            
        keywords = [k.strip() for k in query.replace(',', ' ').split() if len(k) > 3]
        jobs = db.query(DBJob).all()
        
        results = []
        for j in jobs:
            # JD description or title
            jd_text = str(j.description) if j.description else str(j.title)
            
            # Extract job skills for structured matching if available
            j_skills = []
            if hasattr(j, 'requiredSkills') and isinstance(j.requiredSkills, list):
                j_skills = [str(s) for s in j.requiredSkills]
            
            # Use ScoringEngine to match JD against the query
            res = engine.score_with_embedding(
                resume_text=jd_text, # The text being searched
                jd_description=query, # The search query
                query_embedding=query_embedding,
                keywords=keywords,
                candidate_skills=j_skills # Using job skills as "candidate skills" for structured matching
            )
            
            results.append({
                "jobId": j.id,
                "title": j.title,
                "score": res["score"],
                "matchedSkills": res["matched_skills"],
                "jobS3Key": j.job_s3_key
            })
            
        return sorted(results, key=lambda x: x["score"], reverse=True)
