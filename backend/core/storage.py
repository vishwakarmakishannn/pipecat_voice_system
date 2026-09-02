import asyncio
import os
import aioboto3
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

    def local_knowledge_object_path(self, object_name: str) -> Path:
        from core.knowledge_config import KNOWLEDGE_STORAGE_DIR

        root = Path(KNOWLEDGE_STORAGE_DIR).expanduser().resolve()
        candidate = (root / object_name).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("Invalid knowledge object key")
        return candidate

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
