import os
import boto3
import tempfile
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class S3Service:
    def __init__(self):
        try:
            # Boto3 implicitly uses env vars AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY if not explicitly passed
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            self.bucket_name = settings.AWS_S3_BUCKET_NAME
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.s3_client = None

    def download_file(self, s3_key: str) -> str:
        """
        Downloads a file from S3 to a temporary local path.
        Returns the local path. Raises exception on failure.
        """
        if not self.s3_client:
            raise ValueError("S3 Client is not initialized. Check AWS credentials.")

        try:
            # Create a secure temporary file mapping the original extension
            _, ext = os.path.splitext(s3_key)
            fd, temp_path = tempfile.mkstemp(suffix=ext)
            os.close(fd) # Close immediately, we just need the path
            
            logger.info(f"Downloading s3://{self.bucket_name}/{s3_key} to {temp_path}")
            self.s3_client.download_file(self.bucket_name, s3_key, temp_path)
            
            return temp_path
            
        except Exception as e:
            logger.error(f"S3 Download Error for key {s3_key}: {e}")
            raise Exception(f"Failed to download from S3: {e}")

s3_service = S3Service()
