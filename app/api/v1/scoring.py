from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models import ResumeMatchRequest
from app.services.workflow import score_from_s3_and_job

router = APIRouter(prefix="/scoring", tags=["Scoring"])

@router.post("/match-resume")
async def match_resume(
    body: ResumeMatchRequest,
    db: Session = Depends(get_db)
):
    """
    Score a resume from S3 against a job. Safe for concurrent use.
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
