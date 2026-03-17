from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from app.models import CandidateProfile, JobProfile, MatchScore
from app.workflow import parse_and_store_resume, parse_and_store_jd, run_matching, get_recommendations
from app.admin_service import AdminSearchService
from app.database import get_db
from app.config import settings

app = FastAPI(title=settings.PROJECT_NAME, version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Hirelynx Resume Scorer API", "docs": "/docs"}

@app.post("/v1/parser/resume", response_model=CandidateProfile)
async def upload_resume(s3_key: str, db: Session = Depends(get_db)):
    try:
        return parse_and_store_resume(db, s3_key)
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

@app.get("/v1/recommendations/jobs/{candidate_id}", response_model=List[MatchScore])
async def jobs_for_candidate(candidate_id: int, db: Session = Depends(get_db)):
    recs = get_recommendations(db, candidate_id, for_candidate=True)
    if not recs:
        raise HTTPException(status_code=404, detail="No matches found")
    return recs

@app.get("/v1/recommendations/candidates/{job_id}", response_model=List[MatchScore])
async def candidates_for_job(job_id: int, db: Session = Depends(get_db)):
    recs = get_recommendations(db, job_id, for_candidate=False)
    if not recs:
        raise HTTPException(status_code=404, detail="No matches found")
    return recs

@app.post("/v1/admin/search")
async def admin_search(query: str, db: Session = Depends(get_db)):
    """
    Search candidates using AI 
    """
    try:
        return AdminSearchService.search_candidates(db, query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
