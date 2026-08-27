from types import SimpleNamespace
import uuid

import pytest

import services.calls as calls


@pytest.mark.anyio
async def test_restore_refuses_a_call_already_claimed_for_purge(monkeypatch):
    call = SimpleNamespace(
        deleted_at=calls.utcnow(),
        purge_after=calls.utcnow(),
        purge_started_at=calls.utcnow(),
    )

    class Result:
        def scalars(self):
            return self

        def first(self):
            return call

    class Session:
        async def execute(self, _statement):
            return Result()

    class SessionContext:
        async def __aenter__(self):
            return Session()

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr(calls, "AsyncSessionLocal", SessionContext)

    assert await calls.restore_call(uuid.uuid4(), 7) is False
