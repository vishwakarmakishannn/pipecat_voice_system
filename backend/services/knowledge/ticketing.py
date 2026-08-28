"""Active-only mDesk taxonomy matching for a future confirmed ticket tool."""

import re
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from core.models import TicketTaxonomyEntry


@dataclass(frozen=True)
class TicketCandidate:
    entry_id: int
    ticket_code: str
    ticket_subcode: str
    remark: str
    score: float


def _terms(value: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]+", (value or "").lower())
        if len(term) > 1
    }


async def classify_ticket_candidates(
    db: AsyncSession,
    description: str,
    *,
    limit: int = 5,
) -> list[TicketCandidate]:
    query_terms = _terms(description)
    if not query_terms:
        return []
    result = await db.execute(
        select(TicketTaxonomyEntry).where(TicketTaxonomyEntry.active.is_(True))
    )
    candidates = []
    for entry in result.scalars().all():
        code_terms = _terms(entry.ticket_code)
        subcode_terms = _terms(entry.ticket_subcode)
        remark_terms = _terms(entry.remark)
        weighted_matches = (
            1.0 * len(query_terms & code_terms)
            + 1.5 * len(query_terms & subcode_terms)
            + 2.0 * len(query_terms & remark_terms)
        )
        denominator = max(1.0, len(query_terms) + len(subcode_terms | remark_terms))
        score = weighted_matches / denominator
        if score > 0:
            candidates.append(
                TicketCandidate(
                    entry_id=entry.id,
                    ticket_code=entry.ticket_code,
                    ticket_subcode=entry.ticket_subcode,
                    remark=entry.remark,
                    score=round(score, 6),
                )
            )
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[: max(1, min(limit, 10))]


async def require_active_ticket_selection(
    db: AsyncSession,
    *,
    ticket_code: str,
    ticket_subcode: str,
    remark: str,
) -> TicketTaxonomyEntry:
    result = await db.execute(
        select(TicketTaxonomyEntry).where(
            TicketTaxonomyEntry.ticket_code == ticket_code,
            TicketTaxonomyEntry.ticket_subcode == ticket_subcode,
            TicketTaxonomyEntry.remark == remark,
            TicketTaxonomyEntry.active.is_(True),
        )
    )
    entry = result.scalars().first()
    if entry is None:
        raise ValueError("Ticket selection is not in the active Mswipe taxonomy")
    return entry
