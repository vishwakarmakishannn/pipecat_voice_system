"""Strict importer for the Mswipe mDesk ticket taxonomy."""

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from core.models import TicketTaxonomyEntry


@dataclass(frozen=True)
class TaxonomyImportReport:
    source_rows: int
    active_rows: int
    blocked_rows: int
    duplicate_rows: int


def _value(row: dict, *names: str) -> str:
    normalized = {
        re.sub(r"[^a-z0-9]+", "", key.lower()): (value or "").strip()
        for key, value in row.items()
    }
    for name in names:
        key = re.sub(r"[^a-z0-9]+", "", name.lower())
        if key in normalized:
            return normalized[key]
    return ""


async def import_ticket_taxonomy(
    db: AsyncSession,
    csv_path: str | Path,
) -> TaxonomyImportReport:
    path = Path(csv_path).resolve()
    if not path.is_file() or path.suffix.lower() != ".csv":
        raise ValueError("Ticket taxonomy must be an existing CSV file")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    seen: set[tuple[str, str, str]] = set()
    active_rows = blocked_rows = duplicate_rows = 0
    # Re-import is a synchronized snapshot: old entries absent from the new
    # file become inactive rather than being deleted.
    existing_result = await db.execute(select(TicketTaxonomyEntry))
    existing = {
        (item.ticket_code, item.ticket_subcode, item.remark): item
        for item in existing_result.scalars().all()
    }
    for item in existing.values():
        item.active = False
    for row_number, row in enumerate(rows, start=2):
        code = _value(row, "Ticket Code", "ticket_code", "code")
        subcode = _value(row, "Ticket Subcode", "ticket_subcode", "subcode")
        remark = " ".join(_value(row, "Remarks", "Remark", "ticket_remarks").split())
        status = _value(row, "Status").lower()
        if status != "active":
            blocked_rows += 1
            continue
        if not code or not subcode or not remark:
            blocked_rows += 1
            continue
        key = (code, subcode, remark)
        if key in seen:
            duplicate_rows += 1
            continue
        seen.add(key)
        digest = hashlib.sha256("\0".join(key).encode("utf-8")).hexdigest()
        entry = existing.get(key)
        if entry:
            entry.active = True
            entry.source_status = "Active"
            entry.source_row = row_number
            entry.content_hash = digest
        else:
            db.add(
                TicketTaxonomyEntry(
                    ticket_code=code,
                    ticket_subcode=subcode,
                    remark=remark,
                    source_status="Active",
                    active=True,
                    content_hash=digest,
                    source_row=row_number,
                )
            )
        active_rows += 1
    await db.commit()
    return TaxonomyImportReport(len(rows), active_rows, blocked_rows, duplicate_rows)
