"""Deterministic contradiction candidates for human review."""

import re

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from core.models import KnowledgeConflict, KnowledgeUnit


def _normalized(value: str | None) -> str:
    return re.sub(r"\W+", " ", (value or "").lower()).strip()


def _cross_document_pair(left: KnowledgeUnit, right: KnowledgeUnit) -> bool:
    """Sections of one document are complementary, not contradictions."""

    return not (
        left.document_id is not None
        and right.document_id is not None
        and left.document_id == right.document_id
    )


async def detect_knowledge_conflicts(db: AsyncSession) -> int:
    """Flag same-question/topic units whose answers disagree.

    This intentionally produces review candidates, not automatic resolutions.
    Authority and effective dates remain visible to the reviewer/publication
    gate; neither answer is silently discarded.
    """
    result = await db.execute(
        select(KnowledgeUnit)
        .where(KnowledgeUnit.status.in_(("draft", "approved")))
        .order_by(KnowledgeUnit.created_at.asc())
    )
    groups: dict[tuple[str, str, str, str], list[KnowledgeUnit]] = {}
    for unit in result.scalars().all():
        subject = _normalized(unit.question) or _normalized(unit.title)
        key = (
            subject,
            _normalized(unit.product),
            _normalized(unit.topic),
            unit.unit_type,
        )
        if subject:
            groups.setdefault(key, []).append(unit)
    created = 0
    for units in groups.values():
        for left_index, left in enumerate(units):
            for right in units[left_index + 1 :]:
                if not _cross_document_pair(left, right):
                    continue
                if left.content_hash == right.content_hash or _normalized(left.answer) == _normalized(right.answer):
                    continue
                first, second = sorted((left, right), key=lambda item: str(item.id))
                existing = await db.execute(
                    select(KnowledgeConflict.id).where(
                        KnowledgeConflict.left_unit_id == first.id,
                        KnowledgeConflict.right_unit_id == second.id,
                        KnowledgeConflict.conflict_type == "contradictory_answer",
                    )
                )
                if existing.scalar_one_or_none() is not None:
                    continue
                db.add(
                    KnowledgeConflict(
                        left_unit_id=first.id,
                        right_unit_id=second.id,
                        conflict_type="contradictory_answer",
                        details={
                            "subject": left.question or left.title,
                            "left_source": left.source_uri,
                            "right_source": right.source_uri,
                            "left_authority": left.authority,
                            "right_authority": right.authority,
                        },
                    )
                )
                created += 1
    await db.commit()
    return created
