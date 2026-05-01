from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def root():
    return {"message": "Hirelynx Resume Scorer API", "docs": "/docs"}
