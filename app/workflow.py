import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from app.models import CandidateProfile, JobProfile, MatchScore, ResumeParseStatus
from app.utils import extract_text, clean_text, extract_jd_keywords
from app.scoring import ScoringEngine
from app.config import settings

logger = logging.getLogger(__name__)
engine = ScoringEngine(weight=settings.SCORING_WEIGHT)

# --- DB Layer (Robust Simulation) ---
def _load_db() -> Dict[str, List[Any]]:
    if os.path.exists(settings.DB_PATH):
        try:
            with open(settings.DB_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load DB: {e}")
    return {"candidates": [], "jobs": [], "matches": []}

def _save_db(data: Dict[str, Any]):
    try:
        db_dir = os.path.dirname(settings.DB_PATH)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        with open(settings.DB_PATH, "w") as f:
            json.dump(data, f, indent=4, default=str)
    except Exception as e:
        logger.error(f"Failed to save DB: {e}")
        logger.error(f"Failed to save DB: {e}")

# --- Service Layer ---

def parse_and_store_resume(s3_key: str) -> CandidateProfile:
    from app.s3_service import s3_service
    db = _load_db()
    
    local_path = s3_service.download_file(s3_key)
    try:
        raw = extract_text(local_path)
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)
    
    profile = CandidateProfile(
        id=len(db["candidates"]) + 1,
        resumeS3Key=s3_key,
        resumeParseStatus=ResumeParseStatus.PARSED,
        resumeLastParsedAt=datetime.now(),
        resumeParsedJson={"text": clean_text(raw)}
    )
    db["candidates"].append(profile.model_dump())
    _save_db(db)
    return profile

def parse_and_store_jd(s3_key: str) -> JobProfile:
    from app.s3_service import s3_service
    db = _load_db()
    
    local_path = s3_service.download_file(s3_key)
    try:
        raw = extract_text(local_path)
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)
    
    job = JobProfile(
        id=len(db["jobs"]) + 1,
        title=os.path.basename(s3_key),
        skills=extract_jd_keywords(raw),
        jobS3Key=s3_key
    )
    db["jobs"].append(job.model_dump())
    _save_db(db)
    return job


def run_matching() -> int:
    """Runs batch matching for all candidates vs all jobs. Returns count of matches."""
    db = _load_db()
    db["matches"] = []
    count = 0
    for c in db["candidates"]:
        for j in db["jobs"]:
            res = engine.score(c["resumeParsedJson"]["text"], j["title"], j["skills"])
            match = MatchScore(
                candidateId=c["id"], 
                jobId=j["id"],
                overallScore=res["score"], 
                matchedSkills=res["matched_skills"],
                recommendation=res["recommendation"]
            )
            db["matches"].append(match.model_dump())
            count += 1
    _save_db(db)
    return count

def get_recommendations(target_id: int, for_candidate: bool = True) -> List[Dict[str, Any]]:
    db = _load_db()
    key = "candidateId" if for_candidate else "jobId"
    matches = [m for m in db["matches"] if m[key] == target_id]
    return sorted(matches, key=lambda x: x["overallScore"], reverse=True)



