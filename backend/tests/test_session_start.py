from types import SimpleNamespace

import pytest

import services.memory as memory


@pytest.mark.anyio
async def test_voice_start_creates_new_call(monkeypatch):
    class Session:
        def add(self, call):
            self.call = call

        async def commit(self):
            return None

        async def refresh(self, call):
            call.id = "00000000-0000-0000-0000-000000000123"

        async def execute(self, _statement):
            return SimpleNamespace(scalar_one_or_none=lambda: None)

    class SessionContext:
        async def __aenter__(self):
            self.session = Session()
            return self.session

        async def __aexit__(self, *_args):
            return False

    async def authenticate(_token, _db):
        return SimpleNamespace(id=7)

    monkeypatch.setattr(memory, "VoiceSessionLocal", SessionContext)
    monkeypatch.setattr(memory, "authenticate_token", authenticate)

    bundle = await memory.load_session_bundle({"token": "token"})

    assert bundle.call.id == "00000000-0000-0000-0000-000000000123"
    assert bundle.call.user_id == 7
