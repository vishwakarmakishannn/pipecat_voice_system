"""Typed knowledge-unit validation and persistence."""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from core.models import KNOWLEDGE_UNIT_TYPES, KnowledgeUnit


@dataclass(frozen=True)
class UnitInput:
    stable_key: str
    unit_type: str
    title: str
    answer: str
    retrieval_text: str
    source_uri: str
    source_label: str
    document_id: UUID | None = None
    question: str | None = None
    voice_answer: str | None = None
    product: str | None = None
    device: str | None = None
    topic: str | None = None
    issue_family: str | None = None
    intents: list[str] = field(default_factory=list)
    audience: str = "customer"
    language: str = "en"
    region: str = "IN"
    authority: int = 3
    effective_at: datetime | None = None
    expires_at: datetime | None = None
    requires_auth: bool = False
    requires_live_api: bool = False
    escalation_required: bool = False
    ticket_candidates: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def unit_content_hash(value: UnitInput) -> str:
    fields = (
        value.unit_type,
        value.title,
        value.question or "",
        value.answer,
        value.voice_answer or "",
        value.retrieval_text,
        value.source_uri,
    )
    return hashlib.sha256("\0".join(fields).encode("utf-8")).hexdigest()


def _validate(value: UnitInput) -> None:
    if value.unit_type not in KNOWLEDGE_UNIT_TYPES:
        raise ValueError(f"Unsupported knowledge unit type: {value.unit_type}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._/-]{2,254}", value.stable_key):
        raise ValueError("stable_key must be 3-255 lowercase URL-safe characters")
    if not value.title.strip() or not value.answer.strip() or not value.retrieval_text.strip():
        raise ValueError("title, answer, and retrieval_text are required")
    if not 1 <= value.authority <= 5:
        raise ValueError("authority must be between 1 and 5")
    if value.expires_at and value.effective_at and value.expires_at <= value.effective_at:
        raise ValueError("expires_at must be after effective_at")


async def upsert_draft_unit(db: AsyncSession, value: UnitInput) -> KnowledgeUnit:
    _validate(value)
    digest = unit_content_hash(value)
    result = await db.execute(
        select(KnowledgeUnit).where(KnowledgeUnit.stable_key == value.stable_key)
    )
    unit = result.scalars().first()
    # An unchanged rejected/retired unit must stay retired on recrawl; otherwise
    # every incremental crawl would recreate human-rejected or duplicate text.
    if unit and unit.content_hash == digest:
        return unit
    next_version = int(unit.version or 1) if unit is not None else 1
    if unit and unit.status in {"approved", "retired"}:
        # Published identity remains immutable. A source change receives a new
        # stable version so old releases remain reproducible.
        base_key = re.sub(r"\.v\d+$", "", value.stable_key)
        versions = await db.execute(
            select(func.max(KnowledgeUnit.version)).where(
                or_(
                    KnowledgeUnit.stable_key == base_key,
                    KnowledgeUnit.stable_key.like(f"{base_key}.v%"),
                )
            )
        )
        next_version = int(versions.scalar_one_or_none() or unit.version or 1) + 1
        value = UnitInput(**{**value.__dict__, "stable_key": f"{base_key}.v{next_version}"})
        unit = None
    values = {
        **value.__dict__,
        "metadata_json": value.metadata,
        "content_hash": unit_content_hash(value),
        "version": next_version,
        "search_vector": func.to_tsvector(
            "simple", " ".join([value.title, value.question or "", value.retrieval_text])
        ),
    }
    values.pop("metadata")
    if unit is None:
        unit = KnowledgeUnit(**values)
        db.add(unit)
    else:
        for key, item in values.items():
            setattr(unit, key, item)
        unit.status = "draft"
    await db.flush()
    return unit


async def approve_unit(
    db: AsyncSession,
    unit_id: UUID,
    *,
    approved_by_user_id: int | None = None,
    review_notes: str | None = None,
) -> KnowledgeUnit:
    result = await db.execute(
        select(KnowledgeUnit).where(KnowledgeUnit.id == unit_id).with_for_update()
    )
    unit = result.scalars().first()
    if unit is None:
        raise ValueError("Knowledge unit not found")
    unit.status = "approved"
    unit.approved_by_user_id = approved_by_user_id
    unit.approved_at = datetime.now(timezone.utc)
    unit.review_notes = review_notes.strip() if review_notes else None
    await db.commit()
    await db.refresh(unit)
    return unit


async def retire_unit(
    db: AsyncSession,
    unit_id: UUID,
    *,
    review_notes: str | None = None,
) -> KnowledgeUnit:
    result = await db.execute(
        select(KnowledgeUnit).where(KnowledgeUnit.id == unit_id).with_for_update()
    )
    unit = result.scalars().first()
    if unit is None:
        raise ValueError("Knowledge unit not found")
    unit.status = "retired"
    unit.review_notes = review_notes.strip() if review_notes else "Rejected during review"
    await db.commit()
    await db.refresh(unit)
    return unit
