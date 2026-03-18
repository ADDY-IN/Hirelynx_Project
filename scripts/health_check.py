import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.database import engine, SessionLocal
from app.config import settings
from app.scoring import ScoringEngine
from app.s3_service import s3_service

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("HealthCheck")

def check_database():
    logger.info("--- Checking Database Connection ---")
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();")).fetchone()
            logger.info(f"✅ Database connected: {result[0]}")
            
            # Check if tables exist
            from app.models import Base
            logger.info("Checking tables...")
            with SessionLocal() as db:
                from app.models import DBCandidate
                count = db.query(DBCandidate).count()
                logger.info(f"✅ Found {count} candidates in the database.")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def check_s3():
    logger.info("\n--- Checking AWS S3 Connection ---")
    if not settings.AWS_ACCESS_KEY_ID:
        logger.warning("⚠️ AWS_ACCESS_KEY_ID not set. Skipping S3 check.")
        return True
    
    try:
        import boto3
        s3 = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        s3.head_bucket(Bucket=settings.AWS_S3_BUCKET_NAME)
        logger.info(f"✅ S3 connected. Bucket '{settings.AWS_S3_BUCKET_NAME}' is accessible.")
        return True
    except Exception as e:
        logger.error(f"❌ S3 bucket check failed: {e}")
        return False

def check_ai_models():
    logger.info("\n--- Checking AI Scoring Models ---")
    try:
        scorer = ScoringEngine(weight=0.5)
        if scorer.encoder:
            test_text = "Python developer"
            test_embedding = scorer.encoder.encode([test_text])
            if test_embedding is not None and len(test_embedding) > 0:
                logger.info("✅ SentenceTransformer loaded and encoding successful.")
                return True
        logger.error("❌ Scoring engine failed to load or encode.")
        return False
    except Exception as e:
        logger.error(f"❌ AI Model check failed: {e}")
        return False

def run_all_checks():
    logger.info("Starting Hirelynx Production Health Check...")
    
    db_ok = check_database()
    s3_ok = check_s3()
    ai_ok = check_ai_models()
    
    print("\n" + "="*40)
    if db_ok and s3_ok and ai_ok:
        logger.info("🚀 ALL SYSTEMS GO! Your environment is ready for production.")
        sys.exit(0)
    else:
        logger.error("🚫 SYSTEMS CHECK FAILED. Review the errors above before deployment.")
        sys.exit(1)

if __name__ == "__main__":
    run_all_checks()
