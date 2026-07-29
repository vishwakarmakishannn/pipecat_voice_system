from types import SimpleNamespace

import pytest

import services.memory as memory


@pytest.mark.anyio
async def test_voice_start_creates_missing_conversation(monkeypatch):
    class Session:
        def add(self, conversation):
            self.conversation = conversation

        async def commit(self):
            return None

        async def refresh(self, conversation):
            conversation.id = 123

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

    async def empty(*_args):
        return []

    async def no_prior(*_args):
        return None

    monkeypatch.setattr(memory, "VoiceSessionLocal", SessionContext)
    monkeypatch.setattr(memory, "authenticate_token", authenticate)
    monkeypatch.setattr(memory, "_load_active_facts", empty)
    monkeypatch.setattr(memory, "_load_recent_messages_in_session", empty)
    monkeypatch.setattr(memory, "_load_most_recent_prior_conversation", no_prior)

    bundle = await memory.load_memory_bundle({"token": "token"})

    assert bundle.primary_conversation.id == 123
    assert bundle.primary_conversation.user_id == 7
