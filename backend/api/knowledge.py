"""Authenticated serving and admin control-plane API for Mswipe knowledge."""

import hashlib
import re
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from api.auth import get_current_user
from core.database import get_db
from core.knowledge_config import KNOWLEDGE_ADMIN_USER_IDS, KNOWLEDGE_ENABLED
from core.models import (
    KnowledgeFeedback,
    KnowledgeAlias,
    KnowledgeConflict,
    KnowledgeJob,
    KnowledgeRelease,
    KnowledgeReleaseUnit,
    KnowledgeSnapshot,
    KnowledgeSource,
    KnowledgeUnit,
    User,
)
from services.knowledge.fetch import canonicalize_url
from services.knowledge.jobs import enqueue_knowledge_job
from services.knowledge.releases import (
    ReleaseValidationError,
    create_release,
    publish_release,
    rollback_release,
    validate_release,
)
from services.knowledge.retrieval import retrieve_knowledge
from services.knowledge.units import approve_unit
from services.knowledge.units import UnitInput, upsert_draft_unit


router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


async def require_knowledge_admin(
    user: User = Depends(get_current_user),
) -> User:
    if user.id not in KNOWLEDGE_ADMIN_USER_IDS:
        raise HTTPException(status_code=403, detail="Knowledge administrator access required")
    return user


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(4, ge=1, le=10)


class SourceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    canonical_uri: str = Field(..., min_length=3, max_length=4000)
    source_type: str = Field("website", pattern="^(website|pdf|internal|manual|taxonomy|api)$")
    authority: int = Field(3, ge=1, le=5)
    audience: str = Field("customer", min_length=1, max_length=64)
    language: str = Field("en", min_length=2, max_length=16)
    region: str = Field("IN", min_length=2, max_length=32)
    owner: str | None = Field(None, max_length=255)
    crawl_policy: dict = Field(default_factory=dict)


class JobCreate(BaseModel):
    job_type: str = Field(..., pattern="^(crawl_source|embed_units|detect_conflicts)$")
    source_id: UUID | None = None
    payload: dict = Field(default_factory=dict)


class ReleaseCreate(BaseModel):
    version: str = Field(..., min_length=1, max_length=64)
    description: str | None = Field(None, max_length=2000)
    unit_ids: list[UUID] = Field(..., min_length=1)


class FeedbackCreate(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    unit_id: UUID | None = None
    route: str = Field(..., min_length=1, max_length=32)
    outcome: str = Field(..., min_length=1, max_length=32)
    details: dict = Field(default_factory=dict)


class UnitApproval(BaseModel):
    review_notes: str | None = Field(None, max_length=2000)
    voice_answer: str | None = Field(None, min_length=1, max_length=4000)
    atomic_answer: bool | None = None
    answerability_reviewed: bool | None = None


class UnitCreate(BaseModel):
    source_id: UUID
    stable_key: str = Field(..., min_length=3, max_length=255)
    unit_type: str
    title: str = Field(..., min_length=1, max_length=1000)
    question: str | None = Field(None, max_length=2000)
    answer: str = Field(..., min_length=1, max_length=20_000)
    voice_answer: str | None = Field(None, max_length=4000)
    retrieval_text: str = Field(..., min_length=1, max_length=20_000)
    source_uri: str | None = Field(None, max_length=4000)
    product: str | None = Field(None, max_length=128)
    device: str | None = Field(None, max_length=128)
    topic: str | None = Field(None, max_length=128)
    issue_family: str | None = Field(None, max_length=128)
    intents: list[str] = Field(default_factory=list)
    requires_auth: bool = False
    requires_live_api: bool = False
    escalation_required: bool = False
    ticket_candidates: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class AliasCreate(BaseModel):
    canonical: str = Field(..., min_length=1, max_length=255)
    alias: str = Field(..., min_length=1, max_length=255)
    alias_type: str = Field("stt", min_length=1, max_length=32)
    product: str | None = Field(None, max_length=128)
    language: str = Field("en", min_length=2, max_length=16)
    priority: int = Field(0, ge=-100, le=100)


class ConflictResolution(BaseModel):
    status: str = Field(..., pattern="^(resolved|ignored)$")
    resolution: str = Field(..., min_length=1, max_length=4000)


def _release_payload(release: KnowledgeRelease) -> dict:
    return {
        "id": str(release.id),
        "version": release.version,
        "status": release.status,
        "description": release.description,
        "corpus_hash": release.corpus_hash,
        "unit_count": release.unit_count,
        "published_at": release.published_at,
        "retired_at": release.retired_at,
        "created_at": release.created_at,
    }


@router.get("/status")
async def knowledge_status(
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(KnowledgeRelease).where(KnowledgeRelease.status == "published")
    )
    release = result.scalars().first()
    return {
        "enabled": KNOWLEDGE_ENABLED,
        "serving": bool(KNOWLEDGE_ENABLED and release),
        "release": _release_payload(release) if release else None,
    }


@router.post("/search")
async def search_knowledge(
    request: SearchRequest,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    response = await retrieve_knowledge(request.query, db=db, top_k=request.top_k)
    return {
        **response.__dict__,
        "release_id": str(response.release_id) if response.release_id else None,
        "route": response.route.__dict__,
        "hits": [
            {**hit.__dict__, "unit_id": str(hit.unit_id)} for hit in response.hits
        ],
    }


@router.post("/feedback", status_code=201)
async def create_feedback(
    request: FeedbackCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    fingerprint = hashlib.sha256(request.query.strip().lower().encode("utf-8")).hexdigest()
    item = KnowledgeFeedback(
        unit_id=request.unit_id,
        query_fingerprint=fingerprint,
        route=request.route,
        outcome=request.outcome,
        details={**request.details, "reported_by_user_id": user.id},
    )
    db.add(item)
    await db.commit()
    return {"id": str(item.id), "status": "recorded"}


@router.get("/admin/sources")
async def list_sources(
    _admin: User = Depends(require_knowledge_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(KnowledgeSource).order_by(KnowledgeSource.created_at.desc()))
    return [
        {
            "id": str(item.id),
            "name": item.name,
            "canonical_uri": item.canonical_uri,
            "source_type": item.source_type,
            "authority": item.authority,
            "enabled": item.enabled,
            "crawl_policy": item.crawl_policy,
        }
        for item in result.scalars().all()
    ]


@router.post("/admin/sources", status_code=201)
async def create_source(
    request: SourceCreate,
    _admin: User = Depends(require_knowledge_admin),
    db: AsyncSession = Depends(get_db),
):
    if request.source_type in {"website", "pdf", "manual"}:
        canonical_uri = canonicalize_url(request.canonical_uri)
    else:
        canonical_uri = request.canonical_uri.strip().lower()
        expected_scheme = {
            "internal": "internal",
            "taxonomy": "taxonomy",
            "api": "api",
        }[request.source_type]
        if not re.fullmatch(
            rf"{expected_scheme}://[a-z0-9][a-z0-9._/-]+", canonical_uri
        ):
            raise HTTPException(
                status_code=400,
                detail=f"{request.source_type} sources require a safe {expected_scheme}:// logical URI",
            )
    source = KnowledgeSource(
        **request.model_dump(exclude={"canonical_uri"}),
        canonical_uri=canonical_uri,
    )
    db.add(source)
    try:
        await db.commit()
    except Exception as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Knowledge source already exists") from exc
    await db.refresh(source)
    return {"id": str(source.id), "canonical_uri": source.canonical_uri}


@router.post("/admin/jobs", status_code=202)
async def create_job(
    request: JobCreate,
    _admin: User = Depends(require_knowledge_admin),
    db: AsyncSession = Depends(get_db),
):
    try:
        job = await enqueue_knowledge_job(
            db, request.job_type, source_id=request.source_id, payload=request.payload
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": str(job.id), "status": job.status, "job_type": job.job_type}


@router.get("/admin/jobs")
async def list_jobs(
    _admin: User = Depends(require_knowledge_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(KnowledgeJob).order_by(KnowledgeJob.created_at.desc()).limit(100))
    return [
        {
            "id": str(item.id),
            "job_type": item.job_type,
            "status": item.status,
            "attempts": item.attempts,
            "payload": item.payload,
            "error": item.error,
            "created_at": item.created_at,
        }
        for item in result.scalars().all()
    ]


@router.post("/admin/jobs/{job_id}/retry", status_code=202)
async def retry_job(
    job_id: UUID,
    _admin: User = Depends(require_knowledge_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(KnowledgeJob).where(KnowledgeJob.id == job_id).with_for_update()
    )
    job = result.scalars().first()
    if job is None:
        raise HTTPException(status_code=404, detail="Knowledge job not found")
    if job.status != "failed":
        raise HTTPException(status_code=409, detail="Only a failed job can be retried")
    job.status = "queued"
    job.attempts = 0
    job.available_at = datetime.now(timezone.utc)
    job.claimed_at = None
    job.finished_at = None
    job.error = None
    await db.commit()
    return {"id": str(job.id), "status": job.status}


@router.get("/admin/snapshots")
async def list_snapshots(
    source_id: UUID | None = None,
    limit: int = 100,
    _admin: User = Depends(require_knowledge_admin),
    db: AsyncSession = Depends(get_db),
):
    query = select(KnowledgeSnapshot)
    if source_id:
        query = query.where(KnowledgeSnapshot.source_id == source_id)
    result = await db.execute(
        query.order_by(KnowledgeSnapshot.created_at.desc()).limit(max(1, min(limit, 500)))
    )
    return [
        {
            "id": str(item.id),
            "source_id": str(item.source_id),
            "requested_uri": item.requested_uri,
            "final_uri": item.final_uri,
            "status": item.status,
            "content_hash": item.content_hash,
            "raw_storage_key": item.raw_storage_key,
            "size_bytes": item.size_bytes,
            "quality_score": item.quality_score,
            "warnings": item.warnings,
            "error": item.error,
            "fetched_at": item.fetched_at,
        }
        for item in result.scalars().all()
    ]


@router.get("/admin/units")
async def list_units(
    status: str = "draft",
    limit: int = 100,
    _admin: User = Depends(require_knowledge_admin),
    db: AsyncSession = Depends(get_db),
):
    if status not in {"draft", "approved", "retired"}:
        raise HTTPException(status_code=400, detail="Invalid unit status")
    result = await db.execute(
        select(KnowledgeUnit)
        .where(KnowledgeUnit.status == status)
        .order_by(KnowledgeUnit.updated_at.desc())
        .limit(max(1, min(limit, 500)))
    )
    return [
        {
            "id": str(item.id),
            "stable_key": item.stable_key,
            "unit_type": item.unit_type,
            "title": item.title,
            "answer": item.answer,
            "source_uri": item.source_uri,
            "authority": item.authority,
            "status": item.status,
        }
        for item in result.scalars().all()
    ]


@router.post("/admin/units", status_code=201)
async def create_curated_unit(
    request: UnitCreate,
    _admin: User = Depends(require_knowledge_admin),
    db: AsyncSession = Depends(get_db),
):
    source_result = await db.execute(
        select(KnowledgeSource).where(
            KnowledgeSource.id == request.source_id,
            KnowledgeSource.enabled.is_(True),
        )
    )
    source = source_result.scalars().first()
    if source is None:
        raise HTTPException(status_code=404, detail="Enabled knowledge source not found")
    values = request.model_dump(exclude={"source_id", "source_uri"})
    try:
        unit = await upsert_draft_unit(
            db,
            UnitInput(
                **values,
                source_uri=request.source_uri or source.canonical_uri,
                source_label=source.name,
                authority=source.authority,
                audience=source.audience,
                language=source.language,
                region=source.region,
            ),
        )
        await db.commit()
        await db.refresh(unit)
    except ValueError as exc:
        await db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": str(unit.id), "stable_key": unit.stable_key, "status": unit.status}


@router.post("/admin/units/{unit_id}/approve")
async def admin_approve_unit(
    unit_id: UUID,
    request: UnitApproval,
    admin: User = Depends(require_knowledge_admin),
    db: AsyncSession = Depends(get_db),
):
    try:
        unit = await approve_unit(
            db,
            unit_id,
            approved_by_user_id=admin.id,
            review_notes=request.review_notes,
            voice_answer=request.voice_answer,
            atomic_answer=request.atomic_answer,
            answerability_reviewed=request.answerability_reviewed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "id": str(unit.id),
        "status": unit.status,
        "direct_answer_eligible": bool(
            unit.voice_answer
            and (unit.metadata_json or {}).get("atomic_answer") is True
            and (unit.metadata_json or {}).get("answerability_reviewed") is True
            and (unit.metadata_json or {}).get("voice_answer_approved") is True
            and not unit.requires_live_api
        ),
    }


@router.get("/admin/aliases")
async def list_aliases(
    _admin: User = Depends(require_knowledge_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(KnowledgeAlias).order_by(KnowledgeAlias.priority.desc(), KnowledgeAlias.alias)
    )
    return [
        {
            "id": item.id,
            "canonical": item.canonical,
            "alias": item.alias,
            "alias_type": item.alias_type,
            "product": item.product,
            "language": item.language,
            "priority": item.priority,
            "active": item.active,
        }
        for item in result.scalars().all()
    ]


@router.post("/admin/aliases", status_code=201)
async def create_alias(
    request: AliasCreate,
    _admin: User = Depends(require_knowledge_admin),
    db: AsyncSession = Depends(get_db),
):
    alias = KnowledgeAlias(**request.model_dump())
    db.add(alias)
    try:
        await db.commit()
    except Exception as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Knowledge alias already exists") from exc
    await db.refresh(alias)
    return {"id": alias.id, "active": alias.active}


@router.get("/admin/conflicts")
async def list_conflicts(
    status: str = "open",
    _admin: User = Depends(require_knowledge_admin),
    db: AsyncSession = Depends(get_db),
):
    if status not in {"open", "resolved", "ignored"}:
        raise HTTPException(status_code=400, detail="Invalid conflict status")
    result = await db.execute(
        select(KnowledgeConflict)
        .where(KnowledgeConflict.status == status)
        .order_by(KnowledgeConflict.created_at.desc())
        .limit(500)
    )
    return [
        {
            "id": str(item.id),
            "left_unit_id": str(item.left_unit_id),
            "right_unit_id": str(item.right_unit_id),
            "type": item.conflict_type,
            "status": item.status,
            "details": item.details,
            "resolution": item.resolution,
        }
        for item in result.scalars().all()
    ]


@router.post("/admin/conflicts/{conflict_id}/resolve")
async def resolve_conflict(
    conflict_id: UUID,
    request: ConflictResolution,
    admin: User = Depends(require_knowledge_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(KnowledgeConflict)
        .where(KnowledgeConflict.id == conflict_id)
        .with_for_update()
    )
    conflict = result.scalars().first()
    if conflict is None:
        raise HTTPException(status_code=404, detail="Knowledge conflict not found")
    conflict.status = request.status
    conflict.resolution = request.resolution
    conflict.resolved_by_user_id = admin.id
    conflict.resolved_at = datetime.now(timezone.utc)
    await db.commit()
    return {"id": str(conflict.id), "status": conflict.status}


@router.get("/admin/releases")
async def list_releases(
    _admin: User = Depends(require_knowledge_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(KnowledgeRelease).order_by(KnowledgeRelease.created_at.desc()))
    return [_release_payload(item) for item in result.scalars().all()]


@router.post("/admin/releases", status_code=201)
async def admin_create_release(
    request: ReleaseCreate,
    admin: User = Depends(require_knowledge_admin),
    db: AsyncSession = Depends(get_db),
):
    try:
        release = await create_release(
            db,
            version=request.version,
            description=request.description,
            unit_ids=request.unit_ids,
            created_by_user_id=admin.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _release_payload(release)


@router.get("/admin/releases/{release_id}/validate")
async def admin_validate_release(
    release_id: UUID,
    _admin: User = Depends(require_knowledge_admin),
    db: AsyncSession = Depends(get_db),
):
    validation = await validate_release(db, release_id)
    return validation.__dict__


@router.post("/admin/releases/{release_id}/publish")
async def admin_publish_release(
    release_id: UUID,
    _admin: User = Depends(require_knowledge_admin),
    db: AsyncSession = Depends(get_db),
):
    try:
        return _release_payload(await publish_release(db, release_id))
    except ReleaseValidationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/admin/releases/{release_id}/rollback")
async def admin_rollback_release(
    release_id: UUID,
    _admin: User = Depends(require_knowledge_admin),
    db: AsyncSession = Depends(get_db),
):
    try:
        return _release_payload(await rollback_release(db, release_id))
    except ReleaseValidationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
