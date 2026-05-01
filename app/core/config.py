import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

# Get the directory where this file is located and find the project root
# app/core/config.py → go up 3 levels: core → app → project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_FILE = os.path.join(BASE_DIR, ".env")

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/v1"
    PROJECT_NAME: str = "Hirelynx Resume Scorer"
    
    # Database Configuration
    DB_HOST: str = "localhost"
    DB_PORT: str = "5432"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "password"
    DB_NAME: str = "postgres"

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        import urllib.parse
        encoded_password = urllib.parse.quote_plus(self.DB_PASSWORD)
        return f"postgresql://{self.DB_USER}:{encoded_password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    # Scoring Weights
    SCORING_WEIGHT: float = 0.5
    
    # S3 Configuration
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    AWS_S3_BUCKET_NAME: str = "hirelynx-resumes"

    # AI / LLM
    GROQ_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None   # kept for backward compat
    GEMINI_API_KEY: Optional[str] = None

    model_config = SettingsConfigDict(env_file=ENV_FILE, case_sensitive=True, extra="ignore")

settings = Settings()
