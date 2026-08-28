import asyncio

from services.knowledge.taxonomy import import_ticket_taxonomy


class _Scalars:
    def all(self):
        return []


class _Result:
    def scalars(self):
        return _Scalars()


class _Db:
    def __init__(self):
        self.added = []
        self.committed = False

    async def execute(self, statement):
        return _Result()

    def add(self, item):
        self.added.append(item)

    async def commit(self):
        self.committed = True


def test_taxonomy_import_excludes_blocked_blank_and_duplicate_rows(tmp_path):
    source = tmp_path / "tickets.csv"
    source.write_text(
        "Ticket Code,Ticket Sub Code,Ticket Remarks,Status\n"
        "Device,Activation,Activation pending,Active\n"
        "Device,Activation,Activation pending,Active\n"
        "Device,Replacement,Replace device,Blocked\n"
        "Device,Missing,Missing fields,\n",
        encoding="utf-8",
    )
    db = _Db()
    report = asyncio.run(import_ticket_taxonomy(db, source))
    assert report.source_rows == 4
    assert report.active_rows == 1
    assert report.blocked_rows == 2
    assert report.duplicate_rows == 1
    assert db.committed is True
    assert [(item.ticket_code, item.ticket_subcode, item.remark) for item in db.added] == [
        ("Device", "Activation", "Activation pending")
    ]
