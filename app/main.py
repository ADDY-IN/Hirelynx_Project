from fastapi import FastAPI, HTTPException, BackgroundTasks
from typing import List, Dict, Any
from app.models import CandidateProfile, JobProfile, MatchScore
from app.workflow import parse_and_store_resume, parse_and_store_jd, run_matching, get_recommendations
from app.config import settings

app = FastAPI(title=settings.PROJECT_NAME, version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Hirelynx Resume Scorer API", "docs": "/docs"}

@app.post("/v1/parser/resume", response_model=CandidateProfile)
async def upload_resume(file_path: str):
    """
    Simulates a file upload/processing from a path. 
    In production, this would accept a file or an S3 key.
    """
    try:
        return parse_and_store_resume(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/v1/parser/jd", response_model=JobProfile)
async def upload_jd(file_path: str):
    try:
        return parse_and_store_jd(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/v1/scoring/match-all")
async def trigger_matching(background_tasks: BackgroundTasks):
    """
    Triggers batch matching for all candidates and jobs.
    Useful for heavy ML workloads.
    """
    background_tasks.add_task(run_matching)
    return {"message": "Batch matching transition started in background"}

@app.get("/v1/recommendations/jobs/{candidate_id}")
async def jobs_for_candidate(candidate_id: int):
    recs = get_recommendations(candidate_id, for_candidate=True)
    if not recs:
        raise HTTPException(status_code=404, detail="No matches found for this candidate")
    return recs

@app.get("/v1/recommendations/candidates/{job_id}")
async def candidates_for_job(job_id: int):
    recs = get_recommendations(job_id, for_candidate=False)
    if not recs:
        raise HTTPException(status_code=404, detail="No matches found for this job")
    return recs
