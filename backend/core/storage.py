import asyncio
import os
import aioboto3
from loguru import logger
import aiofiles
import shutil
import uuid
from pathlib import Path

from core.recording_config import local_recording_dir

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

    async def upload_path(
        self,
        source_path: str | Path,
        object_name: str,
        *,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload a file without materializing it as one in-memory byte string."""
        source = Path(source_path)
        if self.use_s3:
            async with self.session.client(
                "s3", endpoint_url=S3_ENDPOINT if S3_ENDPOINT else None
            ) as s3:
                await s3.upload_file(
                    str(source),
                    S3_BUCKET,
                    object_name,
                    ExtraArgs={"ContentType": content_type},
                )
            return object_name
        destination = local_recording_dir() / object_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copyfile, source, destination)
        return object_name

    async def upload_rag_path(
        self,
        source_path: str | Path,
        object_name: str,
        *,
        content_type: str = "application/pdf",
    ) -> str:
        """Store a RAG artifact without loading it into the API process."""
        source = Path(source_path)
        if self.use_s3:
            async with self.session.client(
                "s3", endpoint_url=S3_ENDPOINT if S3_ENDPOINT else None
            ) as s3:
                await s3.upload_file(
                    str(source),
                    S3_BUCKET,
                    object_name,
                    ExtraArgs={"ContentType": content_type},
                )
            return object_name
        destination = self.local_rag_object_path(object_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copyfile, source, destination)
        return f"local://{destination}"

    async def upload_knowledge_bytes(
        self,
        data: bytes,
        object_name: str,
        *,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Archive an immutable knowledge snapshot in private object storage."""
        if self.use_s3:
            import io

            async with self.session.client(
                "s3", endpoint_url=S3_ENDPOINT if S3_ENDPOINT else None
            ) as s3:
                await s3.upload_fileobj(
                    io.BytesIO(data),
                    S3_BUCKET,
                    object_name,
                    ExtraArgs={"ContentType": content_type},
                )
            return object_name
        destination = self.local_knowledge_object_path(object_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            async with aiofiles.open(temporary, "wb") as handle:
                await handle.write(data)
            await asyncio.to_thread(os.replace, temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return f"local://{destination}"

    async def create_presigned_get_url(self, object_name: str, expires_seconds: int) -> str | None:
        if not self.use_s3:
            return None
        async with self.session.client(
            "s3", endpoint_url=S3_ENDPOINT if S3_ENDPOINT else None
        ) as s3:
            return await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": S3_BUCKET, "Key": object_name},
                ExpiresIn=expires_seconds,
            )

    def local_object_path(self, object_name: str) -> Path:
        root = local_recording_dir()
        candidate = (root / object_name).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("Invalid recording object key")
        return candidate

    def local_rag_object_path(self, object_name: str) -> Path:
        from core.rag_config import RAG_UPLOAD_DIR

        root = Path(RAG_UPLOAD_DIR).expanduser().resolve()
        candidate = (root / object_name).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("Invalid RAG object key")
        return candidate

    def local_knowledge_object_path(self, object_name: str) -> Path:
        from core.knowledge_config import KNOWLEDGE_STORAGE_DIR

        root = Path(KNOWLEDGE_STORAGE_DIR).expanduser().resolve()
        candidate = (root / object_name).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("Invalid knowledge object key")
        return candidate

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
            if object_name.startswith("local://"):
                src = object_name.replace("local://", "")
                Path(src).unlink(missing_ok=True)
            else:
                from core.rag_config import RAG_UPLOAD_DIR
                src = Path(RAG_UPLOAD_DIR) / object_name
                Path(src).unlink(missing_ok=True)

    async def delete_file_strict(self, object_name: str) -> None:
        """Delete an object and surface failures so retention jobs can retry."""
        if self.use_s3:
            async with self.session.client(
                "s3", endpoint_url=S3_ENDPOINT if S3_ENDPOINT else None
            ) as s3:
                await s3.delete_object(Bucket=S3_BUCKET, Key=object_name)
            return
        self.local_object_path(object_name).unlink(missing_ok=True)

storage_client = StorageClient()
