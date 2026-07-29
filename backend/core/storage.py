import os
import aioboto3
from loguru import logger
import aiofiles

S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

class StorageClient:
    def __init__(self):
        self.use_s3 = bool(S3_BUCKET and AWS_ACCESS_KEY_ID)
        if self.use_s3:
            self.session = aioboto3.Session(
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=S3_REGION,
            )

    async def upload_file(self, data: bytes, object_name: str) -> str:
        """Upload a file to an S3 bucket or local fallback"""
        if self.use_s3:
            try:
                import io
                async with self.session.client('s3', endpoint_url=S3_ENDPOINT if S3_ENDPOINT else None) as s3:
                    await s3.upload_fileobj(io.BytesIO(data), S3_BUCKET, object_name)
                    if S3_ENDPOINT:
                        return f"{S3_ENDPOINT}/{S3_BUCKET}/{object_name}"
                    return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{object_name}"
            except Exception as e:
                logger.error(f"Failed to upload {object_name} to S3: {e}")
                raise e
        else:
            # Fallback to local storage for dev
            from pathlib import Path
            from core.rag_config import RAG_UPLOAD_DIR
            local_path = Path(RAG_UPLOAD_DIR) / object_name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(local_path, 'wb') as f:
                await f.write(data)
            return f"local://{local_path}"

    async def download_file(self, object_name: str, local_path: str):
        """Download a file from an S3 bucket or local fallback"""
        if self.use_s3:
            try:
                async with self.session.client('s3', endpoint_url=S3_ENDPOINT if S3_ENDPOINT else None) as s3:
                    await s3.download_file(S3_BUCKET, object_name, local_path)
            except Exception as e:
                logger.error(f"Failed to download {object_name} from S3: {e}")
                raise e
        else:
            # For local fallback, if the file is already there, we can just copy it if needed,
            # but usually the path returned was the local path. We just copy it to local_path.
            from pathlib import Path
            import shutil
            if object_name.startswith("local://"):
                src = object_name.replace("local://", "")
                shutil.copy2(src, local_path)
            else:
                from core.rag_config import RAG_UPLOAD_DIR
                src = Path(RAG_UPLOAD_DIR) / object_name
                shutil.copy2(src, local_path)

    async def delete_file(self, object_name: str):
        """Delete a file from an S3 bucket"""
        if self.use_s3:
            try:
                async with self.session.client('s3', endpoint_url=S3_ENDPOINT if S3_ENDPOINT else None) as s3:
                    await s3.delete_object(Bucket=S3_BUCKET, Key=object_name)
            except Exception as e:
                logger.error(f"Failed to delete {object_name} from S3: {e}")
                # Don't throw for deletion, just log
        else:
            from pathlib import Path
            if object_name.startswith("local://"):
                src = object_name.replace("local://", "")
                Path(src).unlink(missing_ok=True)
            else:
                from core.rag_config import RAG_UPLOAD_DIR
                src = Path(RAG_UPLOAD_DIR) / object_name
                Path(src).unlink(missing_ok=True)

storage_client = StorageClient()
