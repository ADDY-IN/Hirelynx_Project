from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from sqlalchemy.orm import Session
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.models import JobProfile, DBNocOccupation
from app.services.workflow import generate_job_summary_from_profile
from app.services.summarizer import (
    generate_personalized_responsibilities,
    generate_responsibilities_from_scratch,
)
from app.db.session import get_db
from app.core.utils import extract_role_from_token

router = APIRouter(prefix="/recruiter", tags=["Recruiter"])
security = HTTPBearer()

@router.post("/jobs/summarize")
async def summarize_and_update_job(
    job_data: JobProfile,
):
    """
    Accepts a full JobProfile JSON payload from the recruiter frontend and
    generates a fully AI-written job summary via Groq LLM.

    Gate: the following fields must be present before generation is attempted
      - title         (non-empty)
      - category      (non-empty — the user must select a category)
      - responsibilities OR requiredSkills  (at least one item in either list)

    Fields that are pre-selected by default (employmentType, experienceLevel,
    workSchedule, compensationType, salary slider, country) are intentionally
    excluded from the gate — they carry no signal about form completion.

    Returns 400 with a descriptive message if the gate fails or if Groq is down.
    No auth required — Node.js backend calls this directly.
    """
    try:
        generated_summary = generate_job_summary_from_profile(job_data)
        return {"summary": generated_summary}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/noc/responsibilities")
async def get_job_responsibilities(
    jobTitle: str,
    companyName: Optional[str] = None,
    category: Optional[str] = None,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Fetches personalized responsibilities for a job title.
    1. Searches for a NOC match in the database by title.
    2. If found, uses NOC duties + AI to personalize.
    3. If NOT found, uses AI to generate responsibilities from scratch.
    Requires ADMIN or EMPLOYER token.
    """
    try:
        # 1. Authorize: Only ADMIN or EMPLOYER allowed
        token = credentials.credentials
        role = extract_role_from_token(token)
        if role not in ["ADMIN", "EMPLOYER"]:
             if not (str(token).isdigit() and int(token) > 0):
                 raise HTTPException(status_code=403, detail="Access denied.")

        # 2. Search for NOC match by Title
        noc = db.query(DBNocOccupation).filter(DBNocOccupation.title.ilike(f"%{jobTitle}%")).first()

        if noc and noc.mainDuties:
            # Match found — use NOC duties + AI personalization
            responsibilities = await generate_personalized_responsibilities(
                job_title    = jobTitle,
                noc_title    = noc.title,
                base_duties  = noc.mainDuties,
                company_name = companyName,
                category     = category
            )
        else:
            # No match found — generate from scratch via AI
            responsibilities = await generate_responsibilities_from_scratch(
                job_title    = jobTitle,
                company_name = companyName,
                category     = category
            )

        return {"responsibilities": responsibilities}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
