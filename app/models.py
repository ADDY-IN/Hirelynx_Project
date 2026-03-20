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

class DBJob(Base):
    __tablename__ = "jobs"
    id                           = Column(Integer, primary_key=True, index=True)
    title                        = Column(String, nullable=False, index=True)
    description                  = Column(String, nullable=True)
    summary                      = Column(String, nullable=True)
    companyName                  = Column(String, nullable=True)
    location                     = Column(String, nullable=True)
    category                     = Column(String, nullable=True)          # USER-DEFINED enum in DB
    employmentType               = Column(String, nullable=True)
    experienceLevel              = Column(String, nullable=True)          # USER-DEFINED enum
    experienceMin                = Column(Float,  nullable=True)
    experienceMax                = Column(Float,  nullable=True)
    status                       = Column(String, nullable=True)          # USER-DEFINED enum
    applicationsCount            = Column(Integer, default=0)
    isRemote                     = Column(Boolean, default=False)
    currency                     = Column(String, nullable=True)
    compensationType             = Column(String, nullable=True)          # USER-DEFINED enum
    salaryMin                    = Column(Integer, nullable=True)
    salaryMax                    = Column(Integer, nullable=True)
    responsibilities             = Column(JSON, default=[])
    requiredSkills               = Column(JSON, default=[])
    reportingTo                  = Column(String, nullable=True)
    workSchedule                 = Column(String, nullable=True)          # USER-DEFINED enum
    screeningQuestions           = Column(JSON, default=[])
    requiresWorkAuthorization    = Column(Boolean, default=False)
    openToInternationalCandidates= Column(Boolean, default=False)
    employerId                   = Column(Integer, nullable=True)
    approvedById                 = Column(Integer, nullable=True)
    rejectedById                 = Column(Integer, nullable=True)
    activatedById                = Column(Integer, nullable=True)
    reOpenById                   = Column(Integer, nullable=True)
    rejectReason                 = Column(String, nullable=True)
    closedReason                 = Column(String, nullable=True)
    approvedAt                   = Column(DateTime, nullable=True)
    rejectedAt                   = Column(DateTime, nullable=True)
    closedAt                     = Column(DateTime, nullable=True)
    expiresAt                    = Column(DateTime, nullable=True)
    activatedAt                  = Column(DateTime, nullable=True)
    reOpenAt                     = Column(DateTime, nullable=True)
    deletedAt                    = Column(DateTime, nullable=True)
    createdAt                    = Column(DateTime, default=datetime.utcnow)
    updatedAt                    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DBMatch(Base):
    __tablename__ = "matches"
    id               = Column(Integer, primary_key=True, index=True)
    candidateId      = Column(Integer, ForeignKey("candidate_profiles.id"), nullable=True)
    jobId            = Column(Integer, ForeignKey("jobs.id"))
    overallScore     = Column(Float, default=0.0)
    jobMatchScore    = Column(Float, nullable=True)
    breakdown        = Column(JSON, default={})
    matchedSkills = Column(JSON, default=[])           # legacy
    matchedSkillsList = Column(JSON, default=[])       # NEW: skills matched in this request
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
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None # Legacy/General
    city: Optional[str] = None
    province: Optional[str] = None
    dateOfBirth: Optional[str] = None
    gender: Optional[str] = None # Changed from Enum to str for flexible parsing
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
    role: Optional[str] = None
    jobTitle: Optional[str] = None
    location: Optional[str] = None
    employmentType: Optional[EmploymentType] = None
    startDate: Optional[str] = None
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
    userId: Optional[int] = None
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

class ScreeningQuestion(BaseModel):
    id: str
    question: str
    type: str

class JobProfile(BaseModel):
    id: Optional[int] = None
    title: str
    description: Optional[str] = None
    summary: Optional[str] = None
    location: Optional[str] = None
    category: Optional[str] = None
    employmentType: Optional[str] = None
    experienceLevel: Optional[str] = None
    experienceMin: Optional[float] = None
    experienceMax: Optional[float] = None
    compensationType: Optional[str] = None
    salaryMin: Optional[float] = None
    salaryMax: Optional[float] = None
    currency: Optional[str] = None
    responsibilities: Optional[List[str]] = []
    reportingTo: Optional[str] = None
    workSchedule: Optional[str] = None
    requiredSkills: Optional[List[str]] = []
    requiresWorkAuthorization: Optional[bool] = None
    openToInternationalCandidates: Optional[bool] = None
    screeningQuestions: Optional[List[ScreeningQuestion]] = []
    isRemote: Optional[bool] = None
    expiresAt: Optional[str] = None
    
    # legacy fields just in case
    token: Optional[str] = None
    skills: Optional[List[str]] = []
    jobS3Key: Optional[str] = None

    class Config:
        from_attributes = True
        extra = "allow"

class CandidateSearchFilters(BaseModel):
    """Structured filters for the 'Search by Filter' tab."""
    category:      Optional[str]       = None   # e.g. "IT", "Chef", "Plumber"
    locations:     Optional[List[str]] = None   # e.g. ["Toronto", "Remote"]
    skills:        Optional[List[str]] = None   # e.g. ["React", "Python"]
    jobType:       Optional[List[str]] = None   # FULL_TIME, PART_TIME, CONTRACT, REMOTE, HYBRID
    experienceMin: Optional[float]     = None   # slider min years
    experienceMax: Optional[float]     = None   # slider max years
    salaryMin:     Optional[float]     = None   # slider min monthly salary
    salaryMax:     Optional[float]     = None   # slider max monthly salary
    minMatchScore: Optional[float]     = 0.0    # filter by pre-computed jobMatchScore (0–100)

    class Config:
        extra = "allow"


class CandidateSearchRequest(BaseModel):
    """Request body for POST /v1/admin/candidates/search."""
    mode:    str            = "filter"   # "filter" | "ai"
    query:   Optional[str] = None        # text box (name, title, skills) OR AI natural language
    filters: Optional[CandidateSearchFilters] = None
    limit:   int            = 20

    class Config:
        extra = "allow"


class ResumeMatchRequest(BaseModel):
    """Request body for POST /v1/scoring/match-resume"""
    s3_key: str
    job_id: int


class MatchScore(BaseModel):
    id: Optional[int] = None
    candidateId: Optional[int] = None
    jobId: int
    candidateToken: Optional[str] = None
    jobToken: Optional[str] = None
    applicationId: Optional[int] = None
    # Legacy
    overallScore: float = 0.0
    matchPercentage: Optional[float] = None
    suitabilityScore: Optional[float] = None
    breakdown: Dict[str, Any] = {}
    matchedSkills: List[str] = []
    recommendation: str
    status: str = 'PENDING'
    # New rich fields
    jobMatchScore: Optional[float] = None
    matchedSkillsList: List[str] = []
    missingSkills: List[str] = []
    totalRequiredSkills: Optional[int] = None

    class Config:
        from_attributes = True
        validate_assignment = True
        extra = "allow"

