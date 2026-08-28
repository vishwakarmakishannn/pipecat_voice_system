import asyncio

from core.models import KnowledgeAlias
from services.knowledge.query import normalize_voice_query


class _Scalars:
    def __init__(self, values):
        self._values = values

    def all(self):
        return self._values


class _Result:
    def __init__(self, values):
        self._values = values

    def scalars(self):
        return _Scalars(self._values)


class _Db:
    def __init__(self, values):
        self._values = values

    async def execute(self, statement):
        return _Result(self._values)


def test_voice_aliases_are_applied_but_customer_identifiers_are_preserved():
    aliases = [
        KnowledgeAlias(canonical="Mswipe", alias="em swipe", priority=100),
        KnowledgeAlias(canonical="Neo 2", alias="neo two", priority=90),
    ]

    async def exercise():
        return await normalize_voice_query(
            "My MSWABC123456 is em swipe neo two", _Db(aliases)
        )

    normalized, applied = asyncio.run(exercise())
    assert normalized == "My MSWABC123456 is Mswipe Neo 2"
    assert applied == ("Mswipe", "Neo 2")


def test_mobile_number_is_never_rewritten_by_aliases():
    aliases = [KnowledgeAlias(canonical="incorrect", alias="9876543210", priority=100)]

    async def exercise():
        return await normalize_voice_query("Call 9876543210", _Db(aliases))

    normalized, applied = asyncio.run(exercise())
    assert normalized == "Call 9876543210"
    assert applied == ()
