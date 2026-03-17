from datetime import datetime
from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from app.database import Base

# --- SQLAlchemy Models (Database) ---

class DBCandidate(Base):
    __tablename__ = "candidates"
    id = Column(Integer, primary_key=True, index=True)
    personal_details = Column(JSON, nullable=True) # Map to PersonalDetails
    education = Column(JSON, default=[]) # List of Education
    work_experience = Column(JSON, default=[]) # List of WorkExperience
    skills = Column(JSON, default=[]) # List of Skill objects
    projects = Column(JSON, default=[]) # List of Project
    resume_s3_key = Column(String, nullable=True)
    resume_parse_status = Column(String, default="PENDING")
    resume_parsed_json = Column(JSON, nullable=True)
    resume_last_parsed_at = Column(DateTime, nullable=True)
    is_profile_completed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DBJob(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    skills = Column(JSON, default=[]) # List[str]
    requirements = Column(JSON, nullable=True)
    experience_years = Column(Float, default=0)
    education = Column(JSON, default=[])
    job_s3_key = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class DBMatch(Base):
    __tablename__ = "matches"
    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"))
    job_id = Column(Integer, ForeignKey("jobs.id"))
    overall_score = Column(Float)
    breakdown = Column(JSON, default={})
    matched_skills = Column(JSON, default=[])
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
    personalDetails: Optional[PersonalDetails] = None
    education: List[Education] = []
    workExperience: List[WorkExperience] = []
    skills: List[Skill] = []
    projects: List[Project] = []
    certifications: List[Certificate] = []
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
    title: str
    skills: List[str] = []
    requirements: Optional[Any] = None
    experienceYears: Optional[float] = 0
    education: List[str] = []
    jobS3Key: Optional[str] = None

    class Config:
        from_attributes = True

class MatchScore(BaseModel):
    id: Optional[int] = None
    candidateId: int
    jobId: int
    overallScore: float
    breakdown: Dict[str, float] = {}
    matchedSkills: List[str] = []
    recommendation: str
    status: str = 'PENDING'

    class Config:
        from_attributes = True

