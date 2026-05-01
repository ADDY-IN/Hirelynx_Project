from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.db.session import get_db
from app.models import DBCandidate
from app.services.summarizer import summarizer_service
from app.core.utils import extract_user_id_from_token

router = APIRouter(prefix="/candidate", tags=["Candidate"])
security = HTTPBearer()

@router.get("/summarize")
async def candidate_summary(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Generates a professional 'About Me' summary for the candidate."""
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
