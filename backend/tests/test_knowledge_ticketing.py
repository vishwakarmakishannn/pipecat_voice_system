import asyncio

import pytest

from core.models import TicketTaxonomyEntry
from services.knowledge.ticketing import (
    classify_ticket_candidates,
    require_active_ticket_selection,
)


class _Scalars:
    def __init__(self, values):
        self.values = values

    def all(self):
        return self.values

    def first(self):
        return self.values[0] if self.values else None


class _Result:
    def __init__(self, values):
        self.values = values

    def scalars(self):
        return _Scalars(self.values)


class _Db:
    def __init__(self, values):
        self.values = values

    async def execute(self, statement):
        return _Result([item for item in self.values if item.active])


def entry(entry_id, code, subcode, remark, active=True):
    item = TicketTaxonomyEntry(
        ticket_code=code,
        ticket_subcode=subcode,
        remark=remark,
        content_hash="a" * 64,
        active=active,
    )
    item.id = entry_id
    return item


def test_ticket_candidates_use_only_active_taxonomy_and_rank_remarks():
    database = _Db(
        [
            entry(1, "Device", "Activation", "Device activation pending"),
            entry(2, "Device", "Replacement", "Device damaged"),
            entry(3, "Device", "Activation", "Device activation pending", active=False),
        ]
    )
    candidates = asyncio.run(
        classify_ticket_candidates(database, "my device activation is pending")
    )
    assert [item.entry_id for item in candidates] == [1, 2]
    assert candidates[0].score > candidates[1].score


def test_blocked_ticket_selection_fails_closed():
    database = _Db([entry(3, "Device", "Activation", "Blocked value", active=False)])

    async def exercise():
        with pytest.raises(ValueError, match="active"):
            await require_active_ticket_selection(
                database,
                ticket_code="Device",
                ticket_subcode="Activation",
                remark="Blocked value",
            )

    asyncio.run(exercise())
