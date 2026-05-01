from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class ResumeMatchRequest(BaseModel):
    s3_key: str
    job_id: int

class MatchScore(BaseModel):
    id: Optional[int] = None
    candidateId: Optional[int] = None
    jobId: int
    candidateToken: Optional[str] = None
    jobToken: Optional[str] = None
    applicationId: Optional[int] = None
    overallScore: float = 0.0
    matchPercentage: Optional[float] = None
    suitabilityScore: Optional[float] = None
    breakdown: Dict[str, Any] = {}
    matchedSkills: List[str] = []
    recommendation: str
    status: str = 'PENDING'
    jobMatchScore: Optional[float] = None
    matchedSkillsList: List[str] = []
    missingSkills: List[str] = []
    totalRequiredSkills: Optional[int] = None

    class Config:
        from_attributes = True
        validate_assignment = True
        extra = "allow"
