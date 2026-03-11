from datetime import datetime
from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

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
    level: Optional[str] = None  # 'Beginner' | 'Intermediate' | 'Advanced'
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

class GovernmentIdType(str, Enum):
    PASSPORT = "PASSPORT"
    DRIVER_LICENSE = "DRIVER_LICENSE"
    PERMANENT_RESIDENT_CARD = "PERMANENT_RESIDENT_CARD"
    WORK_PERMIT = "WORK_PERMIT"
    STUDY_PERMIT = "STUDY_PERMIT"
    SIN_CARD = "SIN_CARD"
    PROVINCIAL_ID_CARD = "PROVINCIAL_ID_CARD"
    OTHER = "OTHER"

class GovernmentId(BaseModel):
    idType: GovernmentIdType
    frontImageS3Key: str
    backImageS3Key: Optional[str] = None
    expiryDate: Optional[str] = None
    isConfirmed: bool
    uploadedAt: Optional[datetime] = None

class Reference(BaseModel):
    name: str
    title: Optional[str] = None
    company: Optional[str] = None
    email: Optional[str] = None

class CandidateProfile(BaseModel):
    id: Optional[int] = None
    personalDetails: Optional[PersonalDetails] = None
    governmentId: Optional[GovernmentId] = None
    education: List[Education] = []
    workExperience: List[WorkExperience] = []
    skills: List[Skill] = []
    projects: List[Project] = []
    certifications: List[Certificate] = []
    references: List[Reference] = []
    capabilities: Optional[Capabilities] = None
    resumeS3Key: Optional[str] = None
    resumeParseStatus: Optional[ResumeParseStatus] = None
    resumeParsedJson: Optional[Any] = None
    resumeLastParsedAt: Optional[datetime] = None
    isProfileCompleted: bool = False
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    deletedAt: Optional[datetime] = None

class JobProfile(BaseModel):
    id: Optional[int] = None
    title: str
    skills: List[str] = []
    requirements: Optional[Any] = None
    experienceYears: Optional[float] = 0
    education: List[str] = []
    jobS3Key: Optional[str] = None
    createdAt: Optional[datetime] = None

class MatchScore(BaseModel):
    id: Optional[int] = None
    candidateId: int
    jobId: int
    overallScore: float
    breakdown: Dict[str, float] = {}  # {skills: number, exp: number, edu: number, summary: number}
    matchedSkills: List[str] = []
    recommendation: str
    status: str = 'PENDING' # 'PENDING'|'SHORTLISTED'|'REJECTED'

