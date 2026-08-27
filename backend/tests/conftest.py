"""Repository-wide pytest gates for optional external integration services."""

import os

import pytest


@pytest.fixture(autouse=True)
def isolate_persistence_spool(monkeypatch, tmp_path):
    """Never let unit-test journals leak into the runtime replay directory."""
    monkeypatch.setenv("PERSISTENCE_SPOOL_DIR", str(tmp_path / "persistence-spool"))


def pytest_collection_modifyitems(items):
    if os.getenv("RUN_DATABASE_TESTS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    skip = pytest.mark.skip(
        reason=(
            "requires PostgreSQL+pgvector; set RUN_DATABASE_TESTS=1 after "
            "starting and migrating the test database"
        )
    )
    for item in items:
        if "database" in item.keywords:
            item.add_marker(skip)
