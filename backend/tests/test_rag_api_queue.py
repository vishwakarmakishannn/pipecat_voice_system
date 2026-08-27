from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.datastructures import UploadFile

from api.rag_files import upload_file
import core.storage as storage


@pytest.mark.anyio
async def test_upload_persists_a_durable_queued_job(monkeypatch):
    stored = []

    class Session:
        def add(self, value):
            self.value = value

        async def flush(self):
            self.value.id = 17
            self.value.created_at = datetime.now(timezone.utc)
            self.value.updated_at = self.value.created_at

        async def commit(self):
            return None

        async def refresh(self, _value):
            return None

    async def store(path, object_name, *, content_type):
        stored.append((Path(path).read_bytes(), object_name, content_type))
        return f"local://uploads/{object_name}"

    monkeypatch.setattr(storage.storage_client, "upload_rag_path", store)
    upload = UploadFile(
        filename="report.pdf",
        file=BytesIO(b"%PDF-1.4\n1 0 obj\n%%EOF"),
        headers={"content-type": "application/pdf"},
    )

    response = await upload_file(
        upload,
        current_user=SimpleNamespace(id=5),
        db=Session(),
    )

    assert response.id == 17
    assert response.status == "queued"
    assert stored == [
        (b"%PDF-1.4\n1 0 obj\n%%EOF", "5/17.pdf", "application/pdf")
    ]


@pytest.mark.anyio
async def test_upload_rejects_spoofed_pdf_before_creating_job():
    class Session:
        def add(self, _value):
            raise AssertionError("invalid input must not create a database job")

    upload = UploadFile(
        filename="report.pdf",
        file=BytesIO(b"this is not a PDF"),
        headers={"content-type": "application/pdf"},
    )

    with pytest.raises(Exception) as error:
        await upload_file(upload, current_user=SimpleNamespace(id=5), db=Session())

    assert getattr(error.value, "status_code", None) == 400
