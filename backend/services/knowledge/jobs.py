"""Durable control-plane jobs; never executed in the live call process."""

from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from core.knowledge_config import (
    KNOWLEDGE_EMBEDDING_MODEL,
    KNOWLEDGE_EMBEDDING_PROVIDER,
    KNOWLEDGE_WORKER_STALE_SECONDS,
)
from core.models import KnowledgeEmbedding, KnowledgeJob, KnowledgeUnit
from services.knowledge.embedding import embed_knowledge_texts, embedding_identity
from services.knowledge.ingestion import crawl_website_source
from services.knowledge.conflicts import detect_knowledge_conflicts


VALID_JOB_TYPES = {"crawl_source", "embed_units", "detect_conflicts"}


async def enqueue_knowledge_job(
    db: AsyncSession,
    job_type: str,
    *,
    source_id: UUID | None = None,
    payload: dict | None = None,
) -> KnowledgeJob:
    if job_type not in VALID_JOB_TYPES:
        raise ValueError(f"Unsupported knowledge job type: {job_type}")
    if job_type == "crawl_source" and source_id is None:
        raise ValueError("crawl_source requires source_id")
    job = KnowledgeJob(job_type=job_type, source_id=source_id, payload=payload or {})
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def claim_next_knowledge_job(db: AsyncSession) -> UUID | None:
    now = datetime.now(timezone.utc)
    result = await db.execute(
        select(KnowledgeJob)
        .where(KnowledgeJob.status == "queued", KnowledgeJob.available_at <= now)
        .order_by(KnowledgeJob.available_at.asc(), KnowledgeJob.created_at.asc())
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    job = result.scalars().first()
    if job is None:
        return None
    job.status = "running"
    job.claimed_at = now
    job.attempts += 1
    job.error = None
    await db.commit()
    return job.id


async def recover_stale_knowledge_jobs(db: AsyncSession) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=KNOWLEDGE_WORKER_STALE_SECONDS)
    result = await db.execute(
        select(KnowledgeJob).where(
            KnowledgeJob.status == "running", KnowledgeJob.claimed_at < cutoff
        )
    )
    recovered = 0
    for job in result.scalars().all():
        job.status = "queued" if job.attempts < 3 else "failed"
        job.error = "Recovered after knowledge worker interruption"
        job.claimed_at = None
        recovered += 1
    if recovered:
        await db.commit()
    return recovered


async def _embed_units(db: AsyncSession, payload: dict) -> dict:
    if KNOWLEDGE_EMBEDDING_PROVIDER == "disabled":
        return {"embedded": 0, "reason": "embedding_provider_disabled"}
    requested_ids = payload.get("unit_ids") or []
    query = select(KnowledgeUnit).where(KnowledgeUnit.status.in_(("draft", "approved")))
    if requested_ids:
        query = query.where(KnowledgeUnit.id.in_([UUID(value) for value in requested_ids]))
    result = await db.execute(query.order_by(KnowledgeUnit.stable_key.asc()))
    units = result.scalars().all()
    vectors = await embed_knowledge_texts([unit.retrieval_text for unit in units])
    provider, model, dimension = embedding_identity()
    for unit, vector in zip(units, vectors, strict=True):
        if vector is None:
            raise RuntimeError(f"Missing embedding for {unit.stable_key}")
        await db.execute(
            delete(KnowledgeEmbedding).where(
                KnowledgeEmbedding.unit_id == unit.id,
                KnowledgeEmbedding.provider == provider,
                KnowledgeEmbedding.model == model,
            )
        )
        db.add(
            KnowledgeEmbedding(
                unit_id=unit.id,
                provider=provider,
                model=model,
                dimension=dimension,
                content_hash=unit.content_hash,
                embedding=vector,
            )
        )
    await db.commit()
    return {"embedded": len(units), "provider": provider, "model": model}


async def execute_knowledge_job(db: AsyncSession, job_id: UUID) -> None:
    result = await db.execute(
        select(KnowledgeJob).where(KnowledgeJob.id == job_id)
    )
    job = result.scalars().first()
    if job is None or job.status != "running":
        return
    try:
        if job.job_type == "crawl_source":
            report = await crawl_website_source(db, job.source_id)
            output = report.__dict__
        elif job.job_type == "embed_units":
            output = await _embed_units(db, dict(job.payload or {}))
        elif job.job_type == "detect_conflicts":
            output = {"conflicts_created": await detect_knowledge_conflicts(db)}
        else:
            raise ValueError(f"Unsupported job type: {job.job_type}")
        # Ingestion commits on page boundaries, so reacquire the job after its
        # transactions before recording the final durable outcome.
        result = await db.execute(select(KnowledgeJob).where(KnowledgeJob.id == job_id))
        job = result.scalars().first()
        job.status = "succeeded"
        job.payload = {**dict(job.payload or {}), "result": output}
        job.finished_at = datetime.now(timezone.utc)
        await db.commit()
    except Exception as exc:
        await db.rollback()
        result = await db.execute(select(KnowledgeJob).where(KnowledgeJob.id == job_id))
        job = result.scalars().first()
        if job:
            job.status = "queued" if job.attempts < 3 else "failed"
            job.error = f"{type(exc).__name__}: {exc}"[:4000]
            job.available_at = datetime.now(timezone.utc) + timedelta(
                seconds=min(300, 2 ** job.attempts)
            )
            job.claimed_at = None
            if job.status == "failed":
                job.finished_at = datetime.now(timezone.utc)
            await db.commit()
        raise
