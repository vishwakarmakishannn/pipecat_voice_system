from core import database
import pytest


def test_voice_database_pool_is_reserved_and_bounded():
    assert database.voice_engine is not database.engine
    assert database._voice_statement_timeout_ms < database._voice_budget_ms
    assert database._voice_statement_timeout_ms <= 750
    assert database.voice_engine.pool._timeout <= 0.25


def test_voice_pool_snapshot_exposes_saturation_gauges():
    snapshot = database.voice_pool_snapshot()
    assert set(snapshot) == {"size", "checked_out", "overflow", "checked_in"}
    assert all(isinstance(value, int) for value in snapshot.values())


def test_voice_vector_queries_use_filtered_hnsw_iterative_scans():
    settings = database.voice_engine.url  # force engine construction before checking config
    assert settings is not None
    assert database._voice_hnsw_iterative_scan in {"strict_order", "relaxed_order"}


@pytest.mark.anyio
async def test_warm_voice_pool_opens_requested_connections(monkeypatch):
    opened = 0

    class Connection:
        async def __aenter__(self):
            nonlocal opened
            opened += 1
            return self

        async def __aexit__(self, *_args):
            return False

        async def execute(self, _statement):
            return None

    class Pool:
        @staticmethod
        def size():
            return 10

    class Engine:
        pool = Pool()

        @staticmethod
        def connect():
            return Connection()

    monkeypatch.setattr(database, "voice_engine", Engine())
    await database.warm_voice_pool(2)
    assert opened == 2
