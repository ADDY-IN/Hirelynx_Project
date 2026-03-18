from datetime import datetime
from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from app.database import Base

# --- SQLAlchemy Models (Database) ---

class DBCandidate(Base):
    __tablename__ = "candidate_profiles"
    id = Column(Integer, primary_key=True, index=True)
    userId = Column(Integer, nullable=False) # Link to users table, required by DB
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

class DBJob(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True, nullable=False)
    requiredSkills = Column(JSON, default=[]) 
    description = Column(String, nullable=False) 
    companyName = Column(String, default="Unknown", nullable=False)
    location = Column(String, default="Unknown", nullable=False)
    employmentType = Column(String, default="FULL_TIME", nullable=False)
    status = Column(String, default="PENDING", nullable=False)
    experienceMin = Column(Float, default=0)
    experienceMax = Column(Float, nullable=True)
    education = Column(JSON, default=[])
    job_s3_key = Column(String, nullable=True)
    applicationsCount = Column(Integer, default=0, nullable=False)
    isRemote = Column(Boolean, default=False, nullable=False)
    currency = Column(String, default="USD", nullable=False)
    requiresWorkAuthorization = Column(Boolean, default=False, nullable=False)
    openToInternationalCandidates = Column(Boolean, default=False, nullable=False)
    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

class DBMatch(Base):
    __tablename__ = "matches"
    id = Column(Integer, primary_key=True, index=True)
    candidateId = Column(Integer, ForeignKey("candidate_profiles.id"))
    jobId = Column(Integer, ForeignKey("jobs.id"))
    overallScore = Column(Float)
    breakdown = Column(JSON, default={})
    matchedSkills = Column(JSON, default=[])
    recommendation = Column(String)
    status = Column(String, default="PENDING")

# --- Pydantic Models (API) ---

class ResumeParseStatus(str, Enum):
    PENDING = "PENDING"
    PARSED = "PARSED"
    FAILED = "FAILED"

class Gender(str, Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    OTHER = "OTHER"
    PREFER_NOT_TO_SAY = "PREFER_NOT_TO_SAY"

class EmploymentType(str, Enum):
    FULL_TIME = "FULL_TIME"
    PART_TIME = "PART_TIME"
    CONTRACT = "CONTRACT"
    INTERNSHIP = "INTERNSHIP"
    TEMPORARY = "TEMPORARY"
    FREELANCE = "FREELANCE"

class PersonalDetails(BaseModel):
    phone: Optional[str] = None
    location: Optional[str] = None
    dateOfBirth: Optional[str] = None
    gender: Optional[Gender] = None
    openToRelocate: Optional[bool] = None

class Education(BaseModel):
    degree: str
    institution: str
    fieldOfStudy: Optional[str] = None
    country: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    currentlyStudying: Optional[bool] = None

class WorkExperience(BaseModel):
    companyName: Optional[str] = None
    jobTitle: Optional[str] = None
    location: Optional[str] = None
    employmentType: Optional[EmploymentType] = None
    startDate: str
    endDate: Optional[str] = None
    currentlyWorking: Optional[bool] = None
    responsibilities: List[str] = []

class Skill(BaseModel):
    name: str
    level: Optional[str] = None
    yearsOfExperience: Optional[float] = None

class Project(BaseModel):
    title: str
    domain: Optional[str] = None
    tools: List[str] = []
    projectLink: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    summary: Optional[str] = None

class Certificate(BaseModel):
    name: str
    issuer: Optional[str] = None
    issueDate: Optional[str] = None
    certificateS3Key: Optional[str] = None
    certificateLink: Optional[str] = None

class Capabilities(BaseModel):
    professionalSummary: Optional[str] = Field(None, max_length=150)
    strengths: Optional[str] = Field(None, max_length=150)
    additionalDetailsS3Key: Optional[str] = None

class CandidateProfile(BaseModel):
    id: Optional[int] = None
    token: Optional[str] = None
    userId: int
    personalDetails: Optional[PersonalDetails] = None
    education: Optional[List[Education]] = []
    workExperience: Optional[List[WorkExperience]] = []
    skills: Optional[List[Skill]] = []
    projects: Optional[List[Project]] = []
    certifications: Optional[List[Certificate]] = []
    capabilities: Optional[Capabilities] = None
    resumeS3Key: Optional[str] = None
    resumeParseStatus: Optional[ResumeParseStatus] = None
    resumeParsedJson: Optional[Any] = None
    resumeLastParsedAt: Optional[datetime] = None
    isProfileCompleted: bool = False

    class Config:
        from_attributes = True

class JobProfile(BaseModel):
    id: Optional[int] = None
    token: Optional[str] = None
    title: str
    skills: Optional[List[str]] = []
    requirements: Optional[Any] = None
    experienceYears: Optional[float] = 0
    education: Optional[List[str]] = []
    jobS3Key: Optional[str] = None

    class Config:
        from_attributes = True

class MatchScore(BaseModel):
    id: Optional[int] = None
    candidateId: int
    jobId: int
    candidateToken: Optional[str] = None
    jobToken: Optional[str] = None
    overallScore: float
    matchPercentage: Optional[float] = None
    suitabilityScore: Optional[float] = None
    breakdown: Dict[str, Any] = {}
    matchedSkills: List[str] = []
    recommendation: str
    status: str = 'PENDING'

    class Config:
        from_attributes = True
        validate_assignment = True
        extra = "allow"

