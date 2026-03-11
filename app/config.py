from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/v1"
    PROJECT_NAME: str = "Hirelynx Resume Scorer"
    
    # Database Configuration
    DB_PATH: str = "hirelynx_db.json"
    
    # Scoring Weights
    SCORING_WEIGHT: float = 0.5
    
    # S3 Configuration (Optional, for production fetch)
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = "hirelynx-resumes"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

settings = Settings()
