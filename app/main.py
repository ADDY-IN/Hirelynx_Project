from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from app.models import CandidateProfile, JobProfile, MatchScore, DBCandidate, DBJob
from app.workflow import parse_and_store_resume, parse_and_store_jd, run_matching, get_recommendations
from app.database import get_db
from app.config import settings

app = FastAPI(title=settings.PROJECT_NAME, version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Hirelynx Resume Scorer API", "docs": "/docs"}

@app.post("/v1/parser/resume", response_model=CandidateProfile)
async def upload_resume(s3_key: str, user_id: int, db: Session = Depends(get_db)):
    try:
        return parse_and_store_resume(db, s3_key, user_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/v1/parser/jd", response_model=JobProfile)
async def upload_jd(s3_key: str, db: Session = Depends(get_db)):
    try:
        return parse_and_store_jd(db, s3_key)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/v1/scoring/match-all")
async def trigger_matching(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_matching)
    return {"message": "Batch matching initiated in background"}

from app.search_service import SearchService
from app.summarizer_service import summarizer_service

@app.get("/v1/candidate/{candidate_id}/recommended-jobs", response_model=List[MatchScore])
async def jobs_for_candidate(candidate_id: int, db: Session = Depends(get_db)):
    """Get jobs matched for a candidate with match percentage"""
    recs = get_recommendations(db, candidate_id, for_candidate=True)
    if not recs:
        raise HTTPException(status_code=404, detail="No matches found")
    
    # Map overallScore to matchPercentage for clarity
    results = []
    for r in recs:
        # Create a dictionary of all fields from the database record
        data = {
            "id": r.id,
            "candidateId": r.candidateId,
            "jobId": r.jobId,
            "overallScore": float(r.overallScore),
            "matchPercentage": float(r.overallScore),
            "matchedSkills": r.matchedSkills or [],
            "recommendation": r.recommendation,
            "status": r.status or "PENDING",
            "breakdown": r.breakdown or {}
        }
        results.append(MatchScore(**data))
    return results

@app.get("/v1/recruiter/jobs/{job_id}/top-candidates", response_model=List[MatchScore])
async def candidates_for_job(job_id: int, db: Session = Depends(get_db)):
    """Get top candidates for a job with suitability score"""
    recs = get_recommendations(db, job_id, for_candidate=False)
    if not recs:
        raise HTTPException(status_code=404, detail="No matches found")
    
    # Map overallScore to suitabilityScore for recruiters
    results = []
    for r in recs:
        # Create a dictionary of all fields from the database record
        data = {
            "id": r.id,
            "candidateId": r.candidateId,
            "jobId": r.jobId,
            "overallScore": float(r.overallScore),
            "suitabilityScore": float(r.overallScore),
            "matchedSkills": r.matchedSkills or [],
            "recommendation": r.recommendation,
            "status": r.status or "PENDING",
            "breakdown": r.breakdown or {}
        }
        results.append(MatchScore(**data))
    return results

@app.post("/v1/recruiter/search-candidates")
async def search_candidates(query: str, db: Session = Depends(get_db)):
    """Semantic search for candidates (Recruiter side)"""
    try:
        return SearchService.search_candidates(db, query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/candidate/search-jobs")
async def search_jobs(query: str, db: Session = Depends(get_db)):
    """Semantic search for jobs (Candidate side)"""
    try:
        return SearchService.search_jobs(db, query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/admin/search", deprecated=True)
async def admin_search_legacy(query: str, db: Session = Depends(get_db)):
    """Deprecated: Use /v1/search/candidates instead"""
    try:
        return SearchService.search_candidates(db, query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/candidate/{candidate_id}/summarize")
async def summarize_candidate(candidate_id: int, db: Session = Depends(get_db)):
    """Generates a professional summary for a candidate using AI"""
    candidate = db.query(DBCandidate).filter(DBCandidate.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    # Use structured profile generator instead of raw text chunking
    summary = summarizer_service.summarize_candidate_profile(candidate)
    
    return {"candidateId": candidate_id, "summary": summary}

@app.get("/v1/recruiter/jobs/{job_id}/summarize")
async def summarize_job(job_id: int, db: Session = Depends(get_db)):
    """Summarizes a job description using AI"""
    job = db.query(DBJob).filter(DBJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    summary = summarizer_service.summarize(job.description, max_length=120)
    return {"jobId": job_id, "summary": summary}
