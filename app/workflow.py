import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, cast
from sqlalchemy.orm import Session
from app.models import DBCandidate, DBJob, DBMatch, CandidateProfile, JobProfile, ResumeParseStatus
from app.utils import extract_text, clean_text, extract_jd_keywords
from app.scoring import ScoringEngine
from app.config import settings

logger = logging.getLogger(__name__)
engine = ScoringEngine(weight=settings.SCORING_WEIGHT)

# --- Service Layer ---

def parse_and_store_resume(db: Session, s3_key: str) -> DBCandidate:
    from app.s3_service import s3_service
    
    local_path = s3_service.download_file(s3_key)
    try:
        raw = extract_text(local_path)
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)
    
    db_candidate = DBCandidate(
        resume_s3_key=s3_key,
        resume_parse_status=ResumeParseStatus.PARSED,
        resume_last_parsed_at=datetime.utcnow(),
        resume_parsed_json={"text": clean_text(raw)}
    )
    db.add(db_candidate)
    db.commit()
    db.refresh(db_candidate)
    return db_candidate

def parse_and_store_jd(db: Session, s3_key: str) -> DBJob:
    from app.s3_service import s3_service
    
    local_path = s3_service.download_file(s3_key)
    try:
        raw = extract_text(local_path)
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)
    
    db_job = DBJob(
        title=os.path.basename(s3_key),
        skills=extract_jd_keywords(raw),
        job_s3_key=s3_key
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job

def run_matching() -> int:
    """
    Runs batch matching for all candidates vs all jobs. 
    Thread-safe implementation: creates its own session.
    """
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        candidates = db.query(DBCandidate).all()
        jobs = db.query(DBJob).all()
        
        # Clear old matches
        db.query(DBMatch).delete()
        
        count = 0
        c: DBCandidate
        for c in candidates:
            parsed_json = c.resume_parsed_json
            if not isinstance(parsed_json, dict) or "text" not in parsed_json:
                continue
                
            resume_text = str(parsed_json["text"])
            
            j: DBJob
            for j in jobs:
                # Explicit casting to satisfy strict IDE linters
                jd_description = str(j.title) if j.title is not None else ""
                
                # Ensure keywords is always a list of strings
                raw_skills = j.skills
                keywords: List[str] = []
                if isinstance(raw_skills, list):
                    keywords = [str(s) for s in raw_skills]
                elif isinstance(raw_skills, str) and len(str(raw_skills)) > 0:
                    keywords = [s.strip() for s in str(raw_skills).split(",")]
                    
                res = engine.score(resume_text, jd_description, keywords)
                
                # Extract IDs safely for match creation
                c_id = int(cast(Any, c.id))
                j_id = int(cast(Any, j.id))
                
                match = DBMatch(
                    candidate_id=c_id, 
                    job_id=j_id,
                    overall_score=float(res["score"]), 
                    matched_skills=res["matched_skills"],
                    recommendation=str(res["recommendation"])
                )
                db.add(match)
                count += 1
        db.commit()
        return count
    except Exception as e:
        db.rollback()
        logger.error(f"Batch matching failed: {e}")
        return 0
    finally:
        db.close()

def get_recommendations(db: Session, target_id: int, for_candidate: bool = True) -> List[DBMatch]:
    if for_candidate:
        return db.query(DBMatch).filter(cast(Any, DBMatch.candidate_id) == target_id).order_by(DBMatch.overall_score.desc()).all()
    else:
        return db.query(DBMatch).filter(cast(Any, DBMatch.job_id) == target_id).order_by(DBMatch.overall_score.desc()).all()
