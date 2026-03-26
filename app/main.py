from fastapi import FastAPI, HTTPException, Depends, Request
from sqlalchemy.orm import Session
from sqlalchemy import text as sql_text
from typing import List, Optional
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.models import (CandidateProfile, JobProfile, MatchScore, DBCandidate, DBJob,
                        ResumeMatchRequest, CandidateSearchRequest, EmployerProfile)
from app.workflow import (parse_only,
                          generate_job_summary_from_profile,
                          score_from_s3_and_job)
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


# --- 2. MATCHING: Score resume from S3 against a job ---

@app.post("/v1/scoring/match-resume")
async def match_resume(
    body: ResumeMatchRequest,
    db: Session = Depends(get_db)
):
    """
    Score a resume from S3 against a job. Safe for concurrent use.

    Body: { "s3_key": "...", "job_id": 123 }
    """
    try:
        return score_from_s3_and_job(
            db=db,
            s3_key=body.s3_key,
            job_id=body.job_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- 3. ADMIN: Smart Candidate Search ---

@app.post("/v1/admin/candidates/search")
async def smart_search_candidates(
    body: CandidateSearchRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Three-mode admin candidate search. Requires admin Bearer token.

    MODE: 'filter' (default)
      - Structured filters: category, locations, skills, jobType,
        experienceMin/Max, salaryMin/Max, minMatchScore
      - Optional 'query' for name / email / skill text matching

    MODE: 'ai'
      - Natural language 'query': 'python developer with 5 years in Toronto'
      - BERT semantic search ranked by relevance

    MODE: 'suggestions'
      - Returns dynamic AI-prompt chip strings built from live DB data
      - Use 'limit' to control how many suggestions are returned (default 6)
      - Response: { "suggestions": ["...", ...] }

    Body: { "mode": "filter"|"ai"|"suggestions", "query": "...", "filters": {...}, "limit": 6 }
    """
    try:
        _ = extract_user_id_from_token(credentials.credentials)

        if body.mode == "suggestions":
            from app.search_service import SearchService as _SS
            suggestions = _SS.get_suggestions(db, count=min(body.limit, 20))
            return {"suggestions": suggestions}

        results = SearchService.smart_search(
            db      = db,
            mode    = body.mode,
            query   = body.query,
            filters = body.filters,
            limit   = min(body.limit, 100),
        )
        return {"results": results, "count": len(results)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- 4. CANDIDATE: Generate About Me Summary ---

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


# --- 5. RECRUITER: Summarize Job Description ---

@app.post("/v1/recruiter/jobs/summarize")
async def summarize_and_update_job(
    job_data: JobProfile,
):
    """
    Accepts a full JobProfile JSON payload from the recruiter frontend.
    Generates an AI summary using all structured fields (title, description,
    responsibilities, salary, experience level, location, skills, etc.)
    and returns it. No auth required — Node.js backend calls this directly.
    """
    try:
        generated_summary = generate_job_summary_from_profile(job_data)
        return {"summary": generated_summary}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- 6. EMPLOYER: Company Profile Summary ---

@app.post("/v1/employer/company-profile")
async def create_employer_company_profile(
    request: Request,
):
    """
    Accepts either:
      - The full login response body (with a nested "employerProfile" key), OR
      - Just the employerProfile object directly

    Steps:
      1. Extracts employer/company data from whichever shape is sent.
      2. Scrapes the company website (4s timeout) for real company content.
      3. Sends scraped content + employer data to Groq (1s timeout).
      4. Returns a personalized company summary.

    No auth required — Node.js backend calls this directly after employer login.

    Response: { "summary": "..." }
    """
    try:
        from app.summarizer_service import summarize_employer_profile

        body = await request.json()

        # Handle both shapes: full login response OR just the employerProfile object
        employer_dict = body.get("employerProfile") or body.get("data", {}).get("employerProfile") or body

        # Merge top-level firstName/lastName if present (from the user object)
        if "firstName" not in employer_dict or not employer_dict.get("firstName"):
            employer_dict = {**employer_dict}
            employer_dict.setdefault("firstName", body.get("firstName") or body.get("data", {}).get("firstName"))
            employer_dict.setdefault("lastName",  body.get("lastName")  or body.get("data", {}).get("lastName"))

        # Validate with Pydantic
        profile = EmployerProfile(**employer_dict)
        summary = await summarize_employer_profile(profile.model_dump())
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

