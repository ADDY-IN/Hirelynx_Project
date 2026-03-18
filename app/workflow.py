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

def parse_only(s3_key: str) -> Dict[str, Any]:
    """Downloads and parses a resume from S3, returning the structured data."""
    local_path = s3_service.download_file(s3_key)
    try:
        raw = extract_text(local_path)
        raw = sanitize_for_db(raw)
        parsed_data = ResumeParser.parse(raw)
        parsed_data["_raw_text"] = clean_text(raw) # Keep for summary/indexing later
        return parsed_data
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)

def parse_and_store_resume(db: Session, s3_key: str, user_id: int) -> DBCandidate:
    parsed_data = parse_only(s3_key)
    raw_text = parsed_data.pop("_raw_text", "")
    
    # Store full structured data in resumeParsedJson for auto-fill
    json_data = {
        "text": raw_text,
        "email": parsed_data.get("email"),
        "structuredData": parsed_data
    }
    
    # Delegate to the indexing function
    return index_candidate(db, user_id, parsed_data, json_data, s3_key)

def index_candidate(db: Session, user_id: int, parsed_data: Dict[str, Any], json_data: Dict[str, Any], s3_key: Optional[str] = None) -> DBCandidate:
    """Explicitly indexes/updates a candidate profile in the AI database."""
    db_candidate = db.query(DBCandidate).filter(DBCandidate.userId == user_id).first()
    
    if db_candidate:
        db_candidate.personalDetails = parsed_data.get("personalDetails")
        db_candidate.education = parsed_data.get("education", [])
        db_candidate.workExperience = parsed_data.get("workExperience", [])
        db_candidate.skills = parsed_data.get("skills", [])
        db_candidate.projects = parsed_data.get("projects", [])
        if s3_key:
            db_candidate.resumeS3Key = s3_key
        db_candidate.resumeParseStatus = ResumeParseStatus.PARSED
        db_candidate.resumeLastParsedAt = datetime.utcnow()
        db_candidate.resumeParsedJson = json_data
        logger.info(f"Indexed existing profile for userId: {user_id}")
    else:
        db_candidate = DBCandidate(
            userId=user_id,
            personalDetails=parsed_data.get("personalDetails"),
            education=parsed_data.get("education", []),
            workExperience=parsed_data.get("workExperience", []),
            skills=parsed_data.get("skills", []),
            projects=parsed_data.get("projects", []),
            resumeS3Key=s3_key,
            resumeParseStatus=ResumeParseStatus.PARSED,
            resumeLastParsedAt=datetime.utcnow(),
            resumeParsedJson=json_data
        )
        db.add(db_candidate)
        logger.info(f"Indexed new profile for userId: {user_id}")
    
    db.commit()
    db.refresh(db_candidate)
    
    # Trigger auto-matching for this user
    try:
        match_candidate_against_all_jobs(db, db_candidate)
    except Exception as e:
        logger.error(f"Auto-matching failed for userId {user_id}: {e}")
        
    return db_candidate

def parse_and_store_jd(db: Session, s3_key: str) -> DBJob:
    local_path = s3_service.download_file(s3_key)
    try:
        raw = extract_text(local_path)
        raw = sanitize_for_db(raw)
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)
    
    # Create or update job entry - simplified for now
    db_job = DBJob(
        title=os.path.basename(s3_key),
        requiredSkills=extract_jd_keywords(raw),
        description=raw,
        job_s3_key=s3_key
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job

def index_job(db: Session, job_data: JobProfile) -> DBJob:
    """Explicitly indexes/updates a job profile in the AI database, including auto-summarizing its payload."""
    from app.workflow import generate_job_summary_from_text
    
    db_job = None
    if job_data.id:
        db_job = db.query(DBJob).filter(DBJob.id == job_data.id).first()
    
    # 1. Prepare raw text for summarization
    raw_text_parts = []
    if job_data.description:
        raw_text_parts.append(job_data.description)
    if job_data.responsibilities:
        raw_text_parts.append("Responsibilities: " + ", ".join(job_data.responsibilities))
    
    # 2. Generate summary if not explicitly provided
    generated_summary = job_data.summary
    if not generated_summary and raw_text_parts:
        generated_summary = generate_job_summary_from_text("\n\n".join(raw_text_parts))

    # 3. Determine skills
    skills = job_data.requiredSkills if job_data.requiredSkills else job_data.skills
    
    if db_job:
        db_job.title = job_data.title
        db_job.description = job_data.description or ""
        db_job.summary = generated_summary
        db_job.requiredSkills = skills
        db_job.responsibilities = job_data.responsibilities or []
        db_job.location = job_data.location or "Unknown"
        db_job.employmentType = job_data.employmentType or "FULL_TIME"
        db_job.experienceMin = job_data.experienceMin or 0
        db_job.experienceMax = job_data.experienceMax
        db_job.isRemote = job_data.isRemote or False
        db_job.currency = job_data.currency or "USD"
        db_job.requiresWorkAuthorization = job_data.requiresWorkAuthorization or False
        db_job.openToInternationalCandidates = job_data.openToInternationalCandidates or False
        if job_data.jobS3Key:
            db_job.job_s3_key = job_data.jobS3Key
        logger.info(f"Indexed existing job: {job_data.id}")
    else:
        db_job = DBJob(
            id=job_data.id,
            title=job_data.title,
            description=job_data.description or "",
            summary=generated_summary,
            requiredSkills=skills,
            responsibilities=job_data.responsibilities or [],
            location=job_data.location or "Unknown",
            employmentType=job_data.employmentType or "FULL_TIME",
            experienceMin=job_data.experienceMin or 0,
            experienceMax=job_data.experienceMax,
            isRemote=job_data.isRemote or False,
            currency=job_data.currency or "USD",
            requiresWorkAuthorization=job_data.requiresWorkAuthorization or False,
            openToInternationalCandidates=job_data.openToInternationalCandidates or False,
            job_s3_key=job_data.jobS3Key
        )
        db.add(db_job)
        logger.info(f"Indexed new job: {job_data.id}")
        
    db.commit()
    db.refresh(db_job)
    return db_job

def match_candidate_against_all_jobs(db: Session, candidate: DBCandidate):
    """Matches a single candidate against all available jobs on-demand."""
    jobs = db.query(DBJob).all()
    
    # Clear existing matches for this candidate
    db.query(DBMatch).filter(DBMatch.candidateId == candidate.id).delete()
    
    # Extract parsed JSON safely
    raw_pj = candidate.resumeParsedJson
    parsed_json = {}
    if isinstance(raw_pj, str):
        try:
            parsed_json = json.loads(raw_pj)
        except:
            return
    elif isinstance(raw_pj, dict):
        parsed_json = raw_pj
    
    resume_text = str(parsed_json.get("text", ""))
    if not resume_text:
        return

    for j in jobs:
        jd_description = str(j.description) if j.description else str(j.title)
        
        # Extract skills
        raw_skills = j.requiredSkills
        keywords: List[str] = []
        if isinstance(raw_skills, list):
            keywords = [str(s) for s in raw_skills]
        elif isinstance(raw_skills, str) and len(str(raw_skills)) > 0:
            keywords = [s.strip() for s in str(raw_skills).split(",")]
            
        c_skills = []
        if isinstance(candidate.skills, list):
            c_skills = [str(s) for s in candidate.skills]
            
        res = engine.score_with_embedding(
            resume_text=resume_text, 
            jd_description=jd_description, 
            query_embedding=None, 
            keywords=keywords,
            candidate_skills=c_skills
        )
        
        match = DBMatch(
            candidateId=int(candidate.id), 
            jobId=int(j.id),
            overallScore=float(res["score"]), 
            matchedSkills=res["matched_skills"],
            recommendation=str(res["recommendation"])
        )
        db.add(match)
    db.commit()

def match_job_against_all_candidates(db: Session, job: DBJob):
    """Matches a single job against all available candidates on-demand."""
    candidates = db.query(DBCandidate).all()
    
    # Clear existing matches for this job (optional, but keeps it clean)
    db.query(DBMatch).filter(DBMatch.jobId == job.id).filter(DBMatch.applicationId == None).delete()
    
    jd_description = str(job.description) if job.description else str(job.title)
    raw_skills = job.requiredSkills
    keywords: List[str] = []
    if isinstance(raw_skills, list):
        keywords = [str(s) for s in raw_skills]
    
    for c in candidates:
        # Extract skills and text safely
        c_skills = [str(s) for s in c.skills] if isinstance(c.skills, list) else []
        
        raw_pj = c.resumeParsedJson
        parsed_json = {}
        if isinstance(raw_pj, dict):
            parsed_json = raw_pj
        elif isinstance(raw_pj, str):
            try: parsed_json = json.loads(raw_pj)
            except: continue
        
        resume_text = str(parsed_json.get("text", ""))
        if not resume_text:
            continue
            
        res = engine.score_with_embedding(
            resume_text=resume_text, 
            jd_description=jd_description, 
            keywords=keywords,
            candidate_skills=c_skills
        )
        
        match = DBMatch(
            candidateId=c.id, 
            jobId=job.id,
            overallScore=float(res["score"]), 
            matchedSkills=res["matched_skills"],
            recommendation=str(res["recommendation"]),
            status="COMPLETED"
        )
        db.add(match)
    
    db.commit()


def score_pair(db: Session, candidate: DBCandidate, job: DBJob, application_id: Optional[int] = None) -> DBMatch:
    """Scores a single candidate-job pair specifically for an application."""
    
    # Extract parsed JSON safely
    raw_pj = candidate.resumeParsedJson
    parsed_json = {}
    if isinstance(raw_pj, str):
        try:
            parsed_json = json.loads(raw_pj)
        except:
            pass
    elif isinstance(raw_pj, dict):
        parsed_json = raw_pj
    
    resume_text = str(parsed_json.get("text", ""))
    jd_description = str(job.description) if job.description else str(job.title)
    
    # Skills logic
    raw_skills = job.requiredSkills
    keywords = [str(s) for s in raw_skills] if isinstance(raw_skills, list) else []
    c_skills = [str(s) for s in candidate.skills] if isinstance(candidate.skills, list) else []
    
    res = engine.score_with_embedding(
        resume_text=resume_text, 
        jd_description=jd_description, 
        keywords=keywords,
        candidate_skills=c_skills
    )
    
    # Check for existing match for this application
    match = None
    if application_id:
        match = db.query(DBMatch).filter(DBMatch.applicationId == application_id).first()
        
    if not match:
        match = DBMatch(
            candidateId=candidate.id,
            jobId=job.id,
            applicationId=application_id
        )
        db.add(match)
        
    match.overallScore = float(res["score"])
    match.matchedSkills = res["matched_skills"]
    match.recommendation = str(res["recommendation"])
    match.status = "COMPLETED"
    
    db.commit()
    db.refresh(match)
    return match

def generate_job_summary_from_text(text: str) -> str:
    """
    Summarizes raw JD text for the 'Summarize with AI' form feature.
    """
    from app.summarizer_service import summarizer_service
    if not text:
        return ""
    # Use the existing extractive summarizer
    return summarizer_service.summarize(text, max_length=150)

def run_matching() -> int:
    """
    Runs batch matching for all candidates vs all jobs. 
    Thread-safe implementation: creates its own session.
    """
    db = SessionLocal()
    try:
        candidates = db.query(DBCandidate).all()
        
        # Clear old matches (this is now handled per candidate in match_candidate_against_all_jobs)
        # db.query(DBMatch).delete() 
        
        logger.info(f"Starting batch matching for {len(candidates)} candidates")
        
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
            
            # Use the new helper function
            match_candidate_against_all_jobs(db, c)
            count += 1 # Count candidates processed, not individual matches
        
        # db.commit() # commit is now handled inside match_candidate_against_all_jobs
        logger.info(f"Batch matching completed: {count} candidates processed")
        return count
    except Exception as e:
        db.rollback()
        logger.error(f"Batch matching failed: {e}")
        return 0
    finally:
        db.close()

def get_recommendations(db: Session, target_id: int, for_candidate: bool = True) -> List[DBMatch]:
    from sqlalchemy import text as sql_text
    try:
        if for_candidate:
            rows = db.execute(
                sql_text("SELECT id, \"candidateId\", \"jobId\", \"applicationId\", \"overallScore\", \"matchedSkills\", recommendation, status FROM matches WHERE \"candidateId\" = :tid ORDER BY \"overallScore\" DESC"),
                {"tid": target_id}
            ).fetchall()
        else:
            rows = db.execute(
                sql_text("SELECT id, \"candidateId\", \"jobId\", \"applicationId\", \"overallScore\", \"matchedSkills\", recommendation, status FROM matches WHERE \"jobId\" = :tid ORDER BY \"overallScore\" DESC"),
                {"tid": target_id}
            ).fetchall()
        
        # Build lightweight objects matching DBMatch attribute access
        result = []
        for r in rows:
            m = DBMatch()
            m.id = r[0]
            m.candidateId = r[1]
            m.jobId = r[2]
            m.applicationId = r[3]
            m.overallScore = r[4]
            m.matchedSkills = r[5] or []
            m.recommendation = r[6]
            m.status = r[7] or "COMPLETED"
            result.append(m)
        return result
    except Exception:
        # Fallback to ORM if raw SQL fails
        if for_candidate:
            return db.query(DBMatch).filter(cast(Any, DBMatch.candidateId) == target_id).order_by(DBMatch.overallScore.desc()).all()
        else:
            return db.query(DBMatch).filter(cast(Any, DBMatch.jobId) == target_id).order_by(DBMatch.overallScore.desc()).all()
