from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON
from app.db.session import Base

class DBJob(Base):
    __tablename__ = "jobs"
    id                           = Column(Integer, primary_key=True, index=True)
    title                        = Column(String, nullable=False, index=True)
    description                  = Column(String, nullable=True)
    summary                      = Column(String, nullable=True)
    companyName                  = Column(String, nullable=True)
    location                     = Column(String, nullable=True)
    category                     = Column(String, nullable=True)
    employmentType               = Column(String, nullable=True)
    experienceLevel              = Column(String, nullable=True)
    experienceMin                = Column(Float,  nullable=True)
    experienceMax                = Column(Float,  nullable=True)
    status                       = Column(String, nullable=True)
    applicationsCount            = Column(Integer, default=0)
    isRemote                     = Column(Boolean, default=False)
    currency                     = Column(String, nullable=True)
    compensationType             = Column(String, nullable=True)
    salaryMin                    = Column(Integer, nullable=True)
    salaryMax                    = Column(Integer, nullable=True)
    responsibilities             = Column(JSON, default=[])
    requiredSkills               = Column(JSON, default=[])
    reportingTo                  = Column(String, nullable=True)
    workSchedule                 = Column(String, nullable=True)
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

class DBNocOccupation(Base):
    __tablename__ = "noc_occupations"
    nocCode = Column(String(5), primary_key=True)
    title = Column(String, nullable=False)
    description = Column(String, nullable=True)
    mainDuties = Column(JSON, nullable=True)
    illustrativeExamples = Column(JSON, nullable=True)
    isActive = Column(Boolean, default=True)
    
    @property
    def duties(self) -> list:
        return self.mainDuties or []
