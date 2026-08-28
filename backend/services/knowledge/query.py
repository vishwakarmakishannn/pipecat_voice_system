"""Voice-query normalization with identifier preservation."""

import re

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from core.models import KnowledgeAlias


_PROTECTED_IDENTIFIER = re.compile(
    r"\b(?:C\d{5,}|MSW[A-Z0-9]{6,}|[A-Z0-9]{10,}|\d{10,})\b",
    re.IGNORECASE,
)


def _replace_outside_identifiers(text: str, alias: str, canonical: str) -> str:
    protected: list[str] = []

    def hold(match: re.Match) -> str:
        protected.append(match.group(0))
        return f" __IDENTIFIER_{len(protected) - 1}__ "

    held = _PROTECTED_IDENTIFIER.sub(hold, text)
    held = re.sub(
        rf"(?<!\w){re.escape(alias)}(?!\w)", canonical, held, flags=re.IGNORECASE
    )
    for index, value in enumerate(protected):
        held = held.replace(f"__IDENTIFIER_{index}__", value)
    return " ".join(held.split())


async def normalize_voice_query(
    query: str,
    db: AsyncSession,
) -> tuple[str, tuple[str, ...]]:
    normalized = " ".join((query or "").strip().split())
    result = await db.execute(
        select(KnowledgeAlias)
        .where(KnowledgeAlias.active.is_(True))
        .order_by(KnowledgeAlias.priority.desc(), KnowledgeAlias.id.asc())
    )
    applied: list[str] = []
    for item in result.scalars().all():
        candidate = _replace_outside_identifiers(normalized, item.alias, item.canonical)
        if candidate != normalized:
            applied.append(item.canonical)
            normalized = candidate
    return " ".join(normalized.split()), tuple(dict.fromkeys(applied))
