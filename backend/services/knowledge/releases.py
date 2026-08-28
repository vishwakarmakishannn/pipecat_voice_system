"""Review, publication, and rollback operations for immutable corpora."""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from core.knowledge_config import KNOWLEDGE_EMBEDDING_MODEL, KNOWLEDGE_EMBEDDING_PROVIDER
from core.models import (
    KnowledgeConflict,
    KnowledgeEmbedding,
    KnowledgeRelease,
    KnowledgeReleaseUnit,
    KnowledgeUnit,
)


class ReleaseValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ReleaseValidation:
    valid: bool
    unit_count: int
    errors: tuple[str, ...]


def _corpus_hash(units: list[KnowledgeUnit]) -> str:
    digest = hashlib.sha256()
    for unit in sorted(units, key=lambda item: item.stable_key):
        digest.update(unit.stable_key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(unit.content_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


async def create_release(
    db: AsyncSession,
    *,
    version: str,
    unit_ids: list[UUID],
    description: str | None = None,
    created_by_user_id: int | None = None,
) -> KnowledgeRelease:
    clean_version = version.strip()
    if not clean_version or len(clean_version) > 64:
        raise ValueError("Release version must contain 1 to 64 characters")
    if not unit_ids:
        raise ValueError("A release must contain at least one unit")
    existing = await db.execute(
        select(KnowledgeRelease.id).where(KnowledgeRelease.version == clean_version)
    )
    if existing.scalar_one_or_none() is not None:
        raise ValueError(f"Knowledge release {clean_version!r} already exists")
    units_result = await db.execute(
        select(KnowledgeUnit).where(KnowledgeUnit.id.in_(set(unit_ids)))
    )
    units = units_result.scalars().all()
    if len(units) != len(set(unit_ids)):
        raise ValueError("One or more knowledge unit IDs do not exist")
    release = KnowledgeRelease(
        version=clean_version,
        description=description,
        created_by_user_id=created_by_user_id,
        status="draft",
        unit_count=len(units),
        corpus_hash=_corpus_hash(units),
    )
    db.add(release)
    await db.flush()
    db.add_all(
        KnowledgeReleaseUnit(release_id=release.id, unit_id=unit.id) for unit in units
    )
    await db.commit()
    await db.refresh(release)
    return release


async def validate_release(
    db: AsyncSession,
    release_id: UUID,
    *,
    allow_retired_units: bool = False,
) -> ReleaseValidation:
    rows = await db.execute(
        select(KnowledgeUnit)
        .join(KnowledgeReleaseUnit, KnowledgeReleaseUnit.unit_id == KnowledgeUnit.id)
        .where(KnowledgeReleaseUnit.release_id == release_id)
    )
    units = rows.scalars().all()
    errors: list[str] = []
    if not units:
        errors.append("release_has_no_units")
    allowed_statuses = {"approved", "retired"} if allow_retired_units else {"approved"}
    nonapproved = [unit.stable_key for unit in units if unit.status not in allowed_statuses]
    if nonapproved:
        errors.append(f"units_not_approved:{','.join(nonapproved[:10])}")
    now = datetime.now(timezone.utc)
    expired = [unit.stable_key for unit in units if unit.expires_at and unit.expires_at <= now]
    if expired:
        errors.append(f"units_expired:{','.join(expired[:10])}")
    if units:
        unit_ids = [unit.id for unit in units]
        conflicts = await db.execute(
            select(func.count(KnowledgeConflict.id)).where(
                KnowledgeConflict.status == "open",
                or_(
                    KnowledgeConflict.left_unit_id.in_(unit_ids),
                    KnowledgeConflict.right_unit_id.in_(unit_ids),
                ),
            )
        )
        open_conflicts = int(conflicts.scalar_one() or 0)
        if open_conflicts:
            errors.append(f"open_conflicts:{open_conflicts}")
        if KNOWLEDGE_EMBEDDING_PROVIDER != "disabled":
            embedded = await db.execute(
                select(func.count(func.distinct(KnowledgeEmbedding.unit_id))).where(
                    KnowledgeEmbedding.unit_id.in_(unit_ids),
                    KnowledgeEmbedding.provider == KNOWLEDGE_EMBEDDING_PROVIDER,
                    KnowledgeEmbedding.model == KNOWLEDGE_EMBEDDING_MODEL,
                )
            )
            embedded_count = int(embedded.scalar_one() or 0)
            if embedded_count != len(units):
                errors.append(f"missing_embeddings:{len(units) - embedded_count}")
    return ReleaseValidation(not errors, len(units), tuple(errors))


async def publish_release(db: AsyncSession, release_id: UUID) -> KnowledgeRelease:
    result = await db.execute(
        select(KnowledgeRelease)
        .where(KnowledgeRelease.id == release_id)
        .with_for_update()
    )
    release = result.scalars().first()
    if release is None:
        raise ValueError("Knowledge release not found")
    if release.status != "draft":
        raise ValueError("Only a draft release can be published")
    validation = await validate_release(db, release.id)
    if not validation.valid:
        raise ReleaseValidationError("; ".join(validation.errors))
    now = datetime.now(timezone.utc)
    current_result = await db.execute(
        select(KnowledgeRelease)
        .where(KnowledgeRelease.status == "published")
        .with_for_update()
    )
    current = current_result.scalars().first()
    if current:
        current.status = "retired"
        current.retired_at = now
        # The partial unique index is immediate. Flush retirement before the
        # new published status while retaining one atomic transaction.
        await db.flush()
    release.status = "published"
    release.published_at = now
    release.retired_at = None
    release.unit_count = validation.unit_count
    await db.commit()
    await db.refresh(release)
    return release


async def rollback_release(db: AsyncSession, target_id: UUID) -> KnowledgeRelease:
    target_result = await db.execute(
        select(KnowledgeRelease)
        .where(KnowledgeRelease.id == target_id)
        .with_for_update()
    )
    target = target_result.scalars().first()
    if target is None:
        raise ValueError("Knowledge release not found")
    if target.status != "retired":
        raise ValueError("Rollback target must be a previously published release")
    validation = await validate_release(db, target.id, allow_retired_units=True)
    if not validation.valid:
        raise ReleaseValidationError("; ".join(validation.errors))
    current_result = await db.execute(
        select(KnowledgeRelease)
        .where(KnowledgeRelease.status == "published")
        .with_for_update()
    )
    current = current_result.scalars().first()
    now = datetime.now(timezone.utc)
    if current:
        current.status = "retired"
        current.retired_at = now
        await db.flush()
    target.status = "published"
    target.published_at = now
    target.retired_at = None
    await db.commit()
    await db.refresh(target)
    return target
