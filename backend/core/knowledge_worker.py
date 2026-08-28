import asyncio

from loguru import logger

from core.database import AsyncSessionLocal
from core.knowledge_config import KNOWLEDGE_WORKER_POLL_SECONDS
from services.knowledge.jobs import (
    claim_next_knowledge_job,
    execute_knowledge_job,
    recover_stale_knowledge_jobs,
)


async def run_knowledge_worker(stop_event: asyncio.Event | None = None) -> None:
    stop = stop_event or asyncio.Event()
    async with AsyncSessionLocal() as db:
        recovered = await recover_stale_knowledge_jobs(db)
    logger.info("knowledge_worker status=started recovered_jobs={}", recovered)
    while not stop.is_set():
        async with AsyncSessionLocal() as db:
            job_id = await claim_next_knowledge_job(db)
        if job_id:
            try:
                async with AsyncSessionLocal() as db:
                    await execute_knowledge_job(db, job_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("knowledge_worker status=job_failed job_id={}", job_id)
            continue
        try:
            await asyncio.wait_for(stop.wait(), timeout=KNOWLEDGE_WORKER_POLL_SECONDS)
        except TimeoutError:
            pass
    logger.info("knowledge_worker status=stopped")
