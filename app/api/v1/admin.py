from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.db.session import get_db
from app.models import CandidateSearchRequest
from app.services.search_service import SearchService
from app.core.utils import extract_user_id_from_token

router = APIRouter(prefix="/admin", tags=["Admin"])
security = HTTPBearer()

@router.post("/candidates/search")
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
            from app.services.search_service import SearchService as _SS
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
