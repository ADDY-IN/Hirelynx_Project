import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, cast
from sqlalchemy.orm import Session
from app.models import DBCandidate, DBJob, DBMatch, CandidateProfile, JobProfile, ResumeParseStatus
from app.config import settings
from app.database import SessionLocal
from app.s3_service import s3_service
from app.parser import ResumeParser
from app.scoring import ScoringEngine
from app.utils import extract_text, clean_text, extract_jd_keywords, sanitize_for_db

logger = logging.getLogger(__name__)
engine = ScoringEngine(weight=settings.SCORING_WEIGHT)

# --- Service Layer ---

def parse_and_store_resume(db: Session, s3_key: str, user_id: int) -> DBCandidate:
    
    local_path = s3_service.download_file(s3_key)
    try:
        raw = extract_text(local_path)
        raw = sanitize_for_db(raw) # NEW: Remove null bytes and weird unicode
        parsed_data = ResumeParser.parse(raw) # NEW: Extract structured data
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)
    
    # UPSERT Logic: Check if profile already exists for this user
    db_candidate = db.query(DBCandidate).filter(DBCandidate.userId == user_id).first()
    
    if db_candidate:
        # Update existing record
        db_candidate.personalDetails = parsed_data["personalDetails"]
        db_candidate.education = parsed_data["education"]
        db_candidate.workExperience = parsed_data["workExperience"]
        db_candidate.skills = parsed_data["skills"]
        db_candidate.projects = parsed_data["projects"]
        db_candidate.resumeS3Key = s3_key
        db_candidate.resumeParseStatus = ResumeParseStatus.PARSED
        db_candidate.resumeLastParsedAt = datetime.utcnow()
        db_candidate.resumeParsedJson = {"text": clean_text(raw), "email": parsed_data.get("email")}
        logger.info(f"Updated existing profile for userId: {user_id}")
    else:
        # Create new record
        db_candidate = DBCandidate(
            userId=user_id,
            personalDetails=parsed_data["personalDetails"],
            education=parsed_data["education"],
            workExperience=parsed_data["workExperience"],
            skills=parsed_data["skills"],
            projects=parsed_data["projects"],
            resumeS3Key=s3_key,
            resumeParseStatus=ResumeParseStatus.PARSED,
            resumeLastParsedAt=datetime.utcnow(),
            resumeParsedJson={"text": clean_text(raw), "email": parsed_data.get("email")}
        )
        db.add(db_candidate)
        logger.info(f"Created new profile for userId: {user_id}")
    
    db.commit()
    db.refresh(db_candidate)
    return db_candidate

def parse_and_store_jd(db: Session, s3_key: str) -> DBJob:
    
    local_path = s3_service.download_file(s3_key)
    try:
        raw = extract_text(local_path)
        raw = sanitize_for_db(raw) # NEW: Remove null bytes
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)
    
    db_job = DBJob(
        title=os.path.basename(s3_key),
        requiredSkills=extract_jd_keywords(raw),
        description=raw, # Store full text in description
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
    db = SessionLocal()
    try:
        candidates = db.query(DBCandidate).all()
        jobs = db.query(DBJob).all()
        
        # Clear old matches
        db.query(DBMatch).delete()
        
        logger.info(f"Starting batch matching: {len(candidates)} candidates, {len(jobs)} jobs")
        
        count = 0
        c: DBCandidate
        for c in candidates:
            # Flexible JSON parsing as requested
            raw_pj = c.resumeParsedJson
            parsed_json = {}
            
            if isinstance(raw_pj, str):
                try:
                    parsed_json = json.loads(raw_pj)
                except Exception as e:
                    logger.warning(f"Failed to parse resumeParsedJson string for candidate {c.id}: {e}")
                    continue
            elif isinstance(raw_pj, dict):
                parsed_json = raw_pj
            else:
                logger.debug(f"Candidate {c.id} skipped: resumeParsedJson is {type(raw_pj)}")
                continue

            if not parsed_json or "text" not in parsed_json:
                logger.debug(f"Candidate {c.id} skipped: No 'text' in parsed JSON")
                continue
                
            resume_text = str(parsed_json["text"])
            logger.debug(f"Processing candidate {c.id}, text length: {len(resume_text)}")
            
            j: DBJob
            for j in jobs:
                # Use description for more accurate matching
                jd_description = str(j.description) if j.description else str(j.title)
                
                # Ensure keywords is always a list of strings
                raw_skills = j.requiredSkills
                keywords: List[str] = []
                if isinstance(raw_skills, list):
                    keywords = [str(s) for s in raw_skills]
                elif isinstance(raw_skills, str) and len(str(raw_skills)) > 0:
                    keywords = [s.strip() for s in str(raw_skills).split(",")]
                
                # Extract candidate skills for direct comparison
                c_skills = []
                if isinstance(c.skills, list):
                    c_skills = [str(s) for s in c.skills]
                    
                res = engine.score_with_embedding(
                    resume_text=resume_text, 
                    jd_description=jd_description, 
                    query_embedding=None, 
                    keywords=keywords,
                    candidate_skills=c_skills
                )
                
                # Extract IDs safely for match creation
                c_id = int(c.id)
                j_id = int(j.id)
                
                match = DBMatch(
                    candidateId=c_id, 
                    jobId=j_id,
                    overallScore=float(res["score"]), 
                    matchedSkills=res["matched_skills"],
                    recommendation=str(res["recommendation"])
                )
                db.add(match)
                count += 1
        
        db.commit()
        logger.info(f"Batch matching completed: {count} matches generated")
        return count
    except Exception as e:
        db.rollback()
        logger.error(f"Batch matching failed: {e}")
        return 0
    finally:
        db.close()

def get_recommendations(db: Session, target_id: int, for_candidate: bool = True) -> List[DBMatch]:
    if for_candidate:
        return db.query(DBMatch).filter(cast(Any, DBMatch.candidateId) == target_id).order_by(DBMatch.overallScore.desc()).all()
    else:
        return db.query(DBMatch).filter(cast(Any, DBMatch.jobId) == target_id).order_by(DBMatch.overallScore.desc()).all()
