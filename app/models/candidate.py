from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON
from app.db.session import Base

class DBCandidate(Base):
    __tablename__ = "candidate_profiles"
    id = Column(Integer, primary_key=True, index=True)
    userId = Column(Integer, nullable=True)
    personalDetails = Column(JSON, nullable=True) 
    education = Column(JSON, default=[]) 
    workExperience = Column(JSON, default=[]) 
    skills = Column(JSON, default=[]) 
    projects = Column(JSON, default=[]) 
    resumeS3Key = Column(String, nullable=True)
    resumeParseStatus = Column(String, default="PENDING")
    resumeParsedJson = Column(JSON, nullable=True)
    resumeLastParsedAt = Column(DateTime, nullable=True)
    isProfileCompleted = Column(Boolean, default=False)
    createdAt = Column(DateTime, default=datetime.utcnow)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
