from datetime import datetime, timezone
from io import BytesIO
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

    async def store(data, object_name):
        stored.append((data, object_name))
        return f"local://uploads/{object_name}"

    monkeypatch.setattr(storage.storage_client, "upload_file", store)
    upload = UploadFile(
        filename="report.pdf",
        file=BytesIO(b"%PDF-1.4 test"),
        headers={"content-type": "application/pdf"},
    )

    response = await upload_file(
        upload,
        current_user=SimpleNamespace(id=5),
        db=Session(),
    )

    assert response.id == 17
    assert response.status == "queued"
    assert stored == [(b"%PDF-1.4 test", "5/17.pdf")]
