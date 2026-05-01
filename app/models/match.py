from sqlalchemy import Column, Integer, Float, String, JSON, ForeignKey
from app.db.session import Base

class DBMatch(Base):
    __tablename__ = "matches"
    id               = Column(Integer, primary_key=True, index=True)
    candidateId      = Column(Integer, ForeignKey("candidate_profiles.id"), nullable=True)
    jobId            = Column(Integer, ForeignKey("jobs.id"))
    overallScore     = Column(Float, default=0.0)
    jobMatchScore    = Column(Float, nullable=True)
    breakdown        = Column(JSON, default={})
    matchedSkills = Column(JSON, default=[])           # legacy
    matchedSkillsList = Column(JSON, default=[])       # NEW
    recommendation = Column(String)
    status = Column(String, default="PENDING")
