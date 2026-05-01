from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models import CandidateProfile
from app.services.workflow import parse_only

router = APIRouter(prefix="/parser", tags=["Parser"])

@router.post("/index-resume", response_model=CandidateProfile)
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
            workExperience=parsed_data.get("workExperience", []),
            projects=parsed_data.get("projects", []),
            certifications=parsed_data.get("certifications", []),
            summary=parsed_data.get("summary", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
