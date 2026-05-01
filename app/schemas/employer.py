from typing import Optional
from pydantic import BaseModel, Field

class EmployerProfile(BaseModel):
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    companyName: Optional[str] = None
    companyWebsite: Optional[str] = Field(None, alias="websiteUrl", serialization_alias="companyWebsite")
    companyDescription: Optional[str] = None
    industry: Optional[str] = None
    companySize: Optional[str] = None
    contactPersonName: Optional[str] = None
    contactEmail: Optional[str] = None
    contactPhone: Optional[str] = None
    sinNumber: Optional[str] = None
    legalName: Optional[str] = None
    companyType: Optional[str] = None
    province: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None

    class Config:
        extra = "allow"
