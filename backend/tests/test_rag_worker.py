import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import core.rag_worker as rag_worker


@pytest.mark.anyio
async def test_claim_next_job_marks_durable_record_processing(monkeypatch):
    job = SimpleNamespace(
        id=42,
        status="queued",
        error="old",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    committed = False

    class Result:
        def scalars(self):
            return self

        def first(self):
            return job

    class Session:
        async def execute(self, _statement):
            return Result()

        async def commit(self):
            nonlocal committed
            committed = True

    class SessionContext:
        async def __aenter__(self):
            return Session()

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr(rag_worker, "AsyncSessionLocal", SessionContext)

    assert await rag_worker.claim_next_rag_job() == 42
    assert job.status == "processing"
    assert job.error is None
    assert committed is True


@pytest.mark.anyio
async def test_worker_claims_and_processes_before_stopping(monkeypatch):
    stop = asyncio.Event()
    claims = iter([7])
    processed = []

    async def recover():
        return 0

    async def claim():
        return next(claims)

    async def process(file_id):
        processed.append(file_id)
        stop.set()

    monkeypatch.setattr(rag_worker, "recover_stale_rag_jobs", recover)
    monkeypatch.setattr(rag_worker, "claim_next_rag_job", claim)
    monkeypatch.setattr(rag_worker, "process_rag_file", process)

    await rag_worker.run_rag_worker(stop)

    assert processed == [7]
