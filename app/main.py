from fastapi import FastAPI, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import text as sql_text
from typing import List, Optional
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.models import CandidateProfile, JobProfile, MatchScore, DBCandidate, DBJob
from app.workflow import parse_only, get_recommendations, score_pair, generate_job_summary_from_text, match_job_against_all_candidates
from app.database import get_db
from app.config import settings
from app.utils import encode_id, decode_id, extract_user_id_from_token
from app.summarizer_service import summarizer_service
from app.search_service import SearchService

app = FastAPI(title=settings.PROJECT_NAME, version="1.0.0")
security = HTTPBearer()

@app.get("/")
async def root():
    return {"message": "Hirelynx Resume Scorer API", "docs": "/docs"}

# --- 1. RESUME PARSER ---

@app.post("/v1/parser/index-resume", response_model=CandidateProfile)
async def upload_resume(s3_key: str, db: Session = Depends(get_db)):
    """
    Parses a resume from S3 and returns the structured profile data.
    Note: In this version, saving to DB is handled by other specialized endpoints.
    """
    try:
        parsed_data = parse_only(s3_key)
        parsed_data.pop("_raw_text", None)
        return CandidateProfile(
            personalDetails=parsed_data.get("personalDetails"), 
            skills=parsed_data.get("skills"), 
            education=parsed_data.get("education", []), 
            workExperience=parsed_data.get("workExperience", [])
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- 2. MATCHING (SCORING) ---

@app.post("/v1/recruiter/jobs/{job_id}/match-all", response_model=List[MatchScore])
async def match_all_candidates(job_id: int, db: Session = Depends(get_db)):
    """Matches a specific job against all available candidates and returns scores."""
    try:
        job = db.query(DBJob).filter(DBJob.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
            
        match_job_against_all_candidates(db, job)
        recs = get_recommendations(db, job_id, for_candidate=False)
        
        return [MatchScore(
            id=r.id, candidateId=r.candidateId, jobId=r.jobId,
            applicationId=r.applicationId,
            candidateToken=encode_id("CAND", r.candidateId),
            jobToken=encode_id("JOB", r.jobId),
            overallScore=float(r.overallScore),
            matchPercentage=float(r.overallScore),
            matchedSkills=r.matchedSkills or [],
            recommendation=r.recommendation,
            status=r.status or "COMPLETED"
        ) for r in recs]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/v1/scoring/match-application", response_model=MatchScore)
async def match_application(candidate_token: str, job_id: int, application_id: Optional[int] = None, db: Session = Depends(get_db)):
    """Scores a specific application between a candidate and a job."""
    try:
        candidate_id = decode_id(candidate_token)
        
        candidate = db.query(DBCandidate).filter(DBCandidate.id == candidate_id).first()
        job = db.query(DBJob).filter(DBJob.id == job_id).first()
        
        if not candidate or not job:
            raise HTTPException(status_code=404, detail="Data not found")
            
        match = score_pair(db, candidate, job, application_id)
        return MatchScore(
            id=match.id, candidateId=match.candidateId, jobId=match.jobId,
            applicationId=match.applicationId, 
            candidateToken=candidate_token,
            jobToken=encode_id("JOB", job_id), 
            overallScore=float(match.overallScore),
            matchPercentage=float(match.overallScore), 
            matchedSkills=match.matchedSkills,
            recommendation=match.recommendation, 
            status=match.status
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- 3. RECOMMENDATION (CANDIDATE) ---

@app.get("/v1/candidate/recommended-jobs", response_model=List[MatchScore])
async def candidate_recs(user_token: str, db: Session = Depends(get_db)):
    """Returns top matching jobs for the candidate."""
    try:
        user_id = extract_user_id_from_token(user_token)
        candidate = db.query(DBCandidate).filter(DBCandidate.userId == user_id).first()
        if not candidate:
            raise HTTPException(status_code=404, detail="Profile not found")
            
        recs = get_recommendations(db, candidate.id, for_candidate=True)
        return [MatchScore(
            id=r.id, candidateId=r.candidateId, jobId=r.jobId,
            candidateToken=encode_id("CAND", r.candidateId),
            jobToken=encode_id("JOB", r.jobId),
            overallScore=float(r.overallScore),
            matchPercentage=float(r.overallScore),
            matchedSkills=r.matchedSkills or [],
            recommendation=r.recommendation,
            status=r.status or "COMPLETED"
        ) for r in recs]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- 4. RECOMMENDATION (JOBS/ADMIN) ---

@app.get("/v1/admin/jobs/{job_token}/applications", response_model=List[MatchScore])
async def job_recs(
    job_token: str,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Returns ranked applicants for a specific job (Admin only). Pass admin JWT as Bearer token."""
    try:
        _ = extract_user_id_from_token(credentials.credentials)
        job_id = decode_id(job_token)
        recs = get_recommendations(db, job_id, for_candidate=False)
        return [MatchScore(
            id=r.id, candidateId=r.candidateId, jobId=r.jobId,
            applicationId=r.applicationId,
            candidateToken=encode_id("CAND", r.candidateId),
            jobToken=encode_id("JOB", r.jobId),
            overallScore=float(r.overallScore),
            matchPercentage=float(r.overallScore),
            matchedSkills=r.matchedSkills or [],
            recommendation=r.recommendation,
            status=r.status or "COMPLETED"
        ) for r in recs]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- 6. CANDIDATE SEARCH (ADMIN) ---

@app.get("/v1/admin/candidates/search")
@app.post("/v1/admin/candidates/search")
async def search_candidates(
    query: str = "",
    limit: int = Query(50, ge=1, le=100),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Semantic search for candidates by name, skill, or natural language query. Supports GET/POST. Pass admin JWT as Bearer token."""
    try:
        _ = extract_user_id_from_token(credentials.credentials)
        # Note: user_id is removed as the 'candidates' table schema doesn't have it.
        results = SearchService.search_candidates(db, query, limit=limit)
        return {"results": results, "count": len(results)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/v1/candidate/summarize")
async def candidate_summary(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Generates a professional 'About Me' summary for the candidate. Pass user JWT as Bearer token."""
    try:
        user_token = credentials.credentials
        user_id = extract_user_id_from_token(user_token)
        candidate = db.query(DBCandidate).filter(DBCandidate.userId == user_id).first()
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate profile not found.")
        return {"summary": summarizer_service.summarize_candidate_profile(candidate)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/v1/recruiter/jobs/summarize")
async def summarize_and_update_job(
    job_data: JobProfile,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Accepts job details in the request body and a job token in the Bearer header.
    Generates an AI summary based on the provided payload, saves it to the DB 
    for the job identified by the token, and returns the summary.
    """
    try:
        token = credentials.credentials
        try:
            job_id = decode_id(token)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid token provided in the Bearer header.")

        # Build raw text for summarization from the provided payload
        raw_text_parts = []
        if job_data.description:
            raw_text_parts.append(str(job_data.description))
        
        if job_data.responsibilities:
            raw_text_parts.append("Responsibilities: " + ", ".join([str(r) for r in job_data.responsibilities]))
            
        if not raw_text_parts:
            raw_text_parts.append(str(job_data.title))

        generated_summary = generate_job_summary_from_text("\n\n".join(raw_text_parts))

        # Persist summary to DB using raw SQL to ensure stability across schema variations
        db.execute(
            sql_text("UPDATE jobs SET summary = :summary WHERE id = :jid"),
            {"summary": generated_summary, "jid": job_id}
        )
        db.commit()

        return {"job_id": job_id, "summary": generated_summary}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
