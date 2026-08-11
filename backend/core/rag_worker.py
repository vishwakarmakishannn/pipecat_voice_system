import asyncio
import os
from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy.future import select

from core.database import AsyncSessionLocal
from core.models import RagFile
from services.rag import process_rag_file


async def recover_stale_rag_jobs(stale_after_seconds: float | None = None) -> int:
    """Return abandoned processing jobs to the durable queue after a crash."""
    stale_after = stale_after_seconds or float(
        os.getenv("RAG_WORKER_STALE_AFTER_SECONDS", "1800")
    )
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(1.0, stale_after))
    recovered = 0
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(RagFile).where(
                RagFile.status == "processing", RagFile.updated_at < cutoff
            )
        )
        for rag_file in result.scalars().all():
            rag_file.status = "queued"
            rag_file.error = "Recovered after ingestion worker interruption"
            rag_file.updated_at = datetime.now(timezone.utc)
            recovered += 1
        if recovered:
            await db.commit()
    return recovered


async def claim_next_rag_job() -> int | None:
    """Atomically claim the oldest queued job across all worker replicas."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(RagFile)
            .where(RagFile.status == "queued")
            .order_by(RagFile.created_at.asc(), RagFile.id.asc())
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        rag_file = result.scalars().first()
        if rag_file is None:
            return None
        rag_file.status = "processing"
        rag_file.error = None
        rag_file.updated_at = datetime.now(timezone.utc)
        await db.commit()
        return rag_file.id


async def run_rag_worker(stop_event: asyncio.Event | None = None) -> None:
    """Poll the database-backed queue and process one CPU-heavy job at a time."""
    stop = stop_event or asyncio.Event()
    poll_seconds = max(0.1, float(os.getenv("RAG_WORKER_POLL_SECONDS", "1.0")))
    recovered = await recover_stale_rag_jobs()
    logger.info("rag_worker status=started recovered_jobs={}", recovered)
    while not stop.is_set():
        file_id = await claim_next_rag_job()
        if file_id is not None:
            await process_rag_file(file_id)
            continue
        try:
            await asyncio.wait_for(stop.wait(), timeout=poll_seconds)
        except TimeoutError:
            pass
    logger.info("rag_worker status=stopped")
