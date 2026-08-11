import pytest
import asyncio
from types import SimpleNamespace

from services.memory import (
    MemoryBundle,
    build_memory_chunk,
    build_memory_messages,
    build_live_context_messages,
    build_session_memory_context,
    build_turn_memory_context,
    classify_memory_events,
    message_to_llm,
    is_memory_fact_candidate,
    maintain_memory_chunks_if_needed,
    save_conversation_summary,
)
import services.memory as memory_service
from core.models import Conversation, MemoryChunk, Message, User, UserMemory


@pytest.mark.anyio
async def test_authenticate_conversation_uses_one_joined_query(monkeypatch):
    user = User(id=1, username="kishan", password_hash="x")
    conversation = Conversation(id=10, user_id=1, title="Chat")

    class Result:
        def first(self):
            return user, conversation

    class Session:
        def __init__(self):
            self.calls = 0

        async def execute(self, _statement):
            self.calls += 1
            return Result()

    session = Session()
    monkeypatch.setattr(memory_service.jwt, "decode", lambda *_args, **_kwargs: {"sub": "kishan"})

    authenticated = await memory_service.authenticate_conversation("token", 10, session)

    assert authenticated == (user, conversation)
    assert session.calls == 1


@pytest.mark.anyio
async def test_prior_conversation_is_loaded_only_on_explicit_recall(monkeypatch):
    prior_calls = 0

    async def no_semantic(*_args, **_kwargs):
        return []

    async def prior(*_args, **_kwargs):
        nonlocal prior_calls
        prior_calls += 1
        return None

    class SessionContext:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr(memory_service, "retrieve_semantic_memories", no_semantic)
    monkeypatch.setattr(memory_service, "_load_most_recent_prior_conversation", prior)
    monkeypatch.setattr(memory_service, "VoiceSessionLocal", SessionContext)

    assert await build_turn_memory_context(1, "Tell me a joke") is None
    assert prior_calls == 0
    assert await build_turn_memory_context(1, "What did we discuss last time?") is None
    assert prior_calls == 1


@pytest.mark.anyio
async def test_current_chat_recall_prefers_prior_messages_without_embedding(monkeypatch):
    prior = Conversation(id=10, user_id=1, title="Previous chat")
    semantic_calls = 0

    async def semantic(*_args, **_kwargs):
        nonlocal semantic_calls
        semantic_calls += 1
        await asyncio.sleep(10)

    async def load_prior(*_args, **_kwargs):
        return prior

    async def load_messages(*_args, **_kwargs):
        return [
            Message(role="You", content="Tell me about solar panels."),
            Message(role="Aura", content="We discussed rooftop solar costs."),
        ]

    class SessionContext:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr(memory_service, "retrieve_semantic_memories", semantic)
    monkeypatch.setattr(memory_service, "_load_most_recent_prior_conversation", load_prior)
    monkeypatch.setattr(memory_service, "_load_recent_messages", load_messages)
    monkeypatch.setattr(memory_service, "VoiceSessionLocal", SessionContext)

    context = await asyncio.wait_for(
        build_turn_memory_context(
            1,
            "What were we talking about previously?",
            current_conversation_id=11,
        ),
        timeout=0.1,
    )

    assert "Tell me about solar panels." in context
    assert "rooftop solar costs" in context
    assert semantic_calls == 0


def test_build_memory_messages_ignores_invalid_name_memory():
    bundle = MemoryBundle(
        user=User(id=1, username="kishan", password_hash="x"),
        primary_conversation=Conversation(id=10, user_id=1, title="Old chat", summary=""),
        facts=[UserMemory(key="real_name", value="fine", status="active", fact_type="profile")],
        primary_summary="",
        primary_recent_messages=[],
    )

    assert build_memory_messages(bundle) == []


def test_summary_and_overlapping_recent_turn_use_separate_context_layers(monkeypatch):
    bundle = MemoryBundle(
        user=User(id=1, username="kishan", password_hash="x"),
        primary_conversation=Conversation(id=10, user_id=1, title="Chat", summary=""),
        facts=[],
        primary_summary="The user explicitly said that I prefer jasmine tea every morning.",
        primary_recent_messages=[
            Message(role="You", content="I prefer jasmine tea every morning."),
            Message(role="Aura", content="I will remember that preference."),
        ],
    )
    monkeypatch.setattr(memory_service, "MEMORY_PROMPT_MAX_TOKENS", 100)

    messages = build_memory_messages(bundle)

    assert any(
        message.get("content") == "I prefer jasmine tea every morning."
        for message in messages
    )
    assert any("I will remember" in message.get("content", "") for message in messages)
    assert "jasmine tea" in build_session_memory_context(bundle)


def test_memory_prompt_keeps_latest_turn_within_shared_budget(monkeypatch):
    bundle = MemoryBundle(
        user=User(id=1, username="kishan", password_hash="x"),
        primary_conversation=Conversation(id=10, user_id=1, title="Chat", summary=""),
        facts=[],
        primary_summary="",
        primary_recent_messages=[
            Message(role="You", content="old " * 50),
            Message(role="Aura", content="latest answer"),
        ],
    )
    monkeypatch.setattr(memory_service, "MEMORY_PROMPT_MAX_TOKENS", 10)

    messages = build_memory_messages(bundle)

    assert sum(len(message["content"]) for message in messages) <= 40
    assert messages[-1]["content"] == "latest answer"


def test_message_to_llm_maps_supported_roles_only():
    assert message_to_llm(Message(role="You", content="hello")) == {
        "role": "user",
        "content": "hello",
    }
    assert message_to_llm(Message(role="Aura", content="hi")) == {
        "role": "assistant",
        "content": "hi",
    }
    assert message_to_llm(Message(role="ToolCall", content="{}")) is None
    assert message_to_llm(Message(role="RagCall", content="{}")) is None


def test_build_memory_messages_is_safe_alias_for_recent_live_dialogue():
    bundle = MemoryBundle(
        user=User(id=1, username="kishan", password_hash="x"),
        primary_conversation=Conversation(id=10, user_id=1, title="Old chat", summary="Discussed memory."),
        facts=[
            UserMemory(key="real_name", value="Kishan", status="active", fact_type="profile"),
            UserMemory(key="likes", value="football", status="active", fact_type="preference"),
            UserMemory(key="likes", value="apple", status="active", fact_type="preference"),
        ],
        primary_summary="Discussed memory.",
        primary_recent_messages=[
            Message(role="You", content="What did we discuss?"),
            Message(role="Aura", content="Memory implementation."),
        ],
    )

    messages = build_memory_messages(bundle)

    assert messages == [
        {"role": "user", "content": "What did we discuss?"},
        {"role": "assistant", "content": "Memory implementation."},
    ]


def test_build_memory_messages_does_not_eagerly_install_prior_context():
    bundle = MemoryBundle(
        user=User(id=1, username="kishan", password_hash="x"),
        primary_conversation=Conversation(id=11, user_id=1, title="New conversation", summary=""),
        facts=[],
        primary_summary="",
        primary_recent_messages=[],
        prior_conversation=Conversation(
            id=10,
            user_id=1,
            title="AI chat",
            summary="The user asked what AI is.",
        ),
        prior_recent_messages=[
            Message(role="You", content="Explain AI in one line."),
            Message(role="Aura", content="AI lets computers simulate human intelligence."),
        ],
    )

    messages = build_memory_messages(bundle)

    assert messages == []


def test_session_instruction_memory_contains_stable_data_not_recent_dialogue():
    bundle = MemoryBundle(
        user=User(id=1, username="kishan", password_hash="x"),
        primary_conversation=Conversation(id=10, user_id=1, title="Chat"),
        facts=[
            UserMemory(
                key="preferred_name",
                value="Kishan",
                status="active",
                fact_type="profile",
            )
        ],
        primary_summary="The user is comparing phones.",
        primary_recent_messages=[Message(role="You", content="Latest question")],
    )

    memory_context = build_session_memory_context(bundle)

    assert "preferred_name: Kishan" in memory_context
    assert "The user is comparing phones." in memory_context
    assert "Latest question" not in memory_context


def test_live_context_contains_only_recent_primary_dialogue():
    bundle = MemoryBundle(
        user=User(id=1, username="kishan", password_hash="x"),
        primary_conversation=Conversation(id=11, user_id=1, title="Chat"),
        facts=[
            UserMemory(
                key="likes", value="football", status="active", fact_type="preference"
            )
        ],
        primary_summary="Old canonical summary.",
        primary_recent_messages=[
            Message(role="You", content="Question"),
            Message(role="Aura", content="Answer"),
        ],
        prior_conversation=Conversation(id=10, user_id=1, title="Older"),
        prior_recent_messages=[Message(role="You", content="Prior chat")],
    )

    assert build_live_context_messages(bundle) == [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]


@pytest.mark.anyio
async def test_chunk_maintenance_no_longer_generates_competing_summary(
    monkeypatch,
):
    recent = [
        Message(id=index, role="You" if index % 2 else "Aura", content=str(index))
        for index in range(1, 9)
    ]

    class CountResult:
        def scalar_one(self):
            return 16

    class MessagesResult:
        class Scalars:
            @staticmethod
            def all():
                return list(reversed(recent))

        def scalars(self):
            return self.Scalars()

    class Session:
        def __init__(self):
            self.calls = 0

        async def execute(self, _statement):
            self.calls += 1
            return CountResult() if self.calls == 1 else MessagesResult()

    stored = []

    async def store(_db, _conversation, messages):
        stored.append(messages)

    monkeypatch.setattr(memory_service, "store_memory_chunk", store)
    conversation = Conversation(id=10, user_id=1, summary="canonical")

    await maintain_memory_chunks_if_needed(Session(), conversation)

    assert conversation.summary == "canonical"
    assert stored == [recent]


@pytest.mark.anyio
async def test_applied_live_summary_updates_canonical_conversation(monkeypatch):
    conversation = Conversation(id=10, user_id=1, summary="old")

    class Result:
        class Scalars:
            @staticmethod
            def first():
                return conversation

        def scalars(self):
            return self.Scalars()

    class Session:
        committed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def execute(self, _statement):
            return Result()

        async def commit(self):
            self.committed = True

    session = Session()
    monkeypatch.setattr(memory_service, "AsyncSessionLocal", lambda: session)

    assert await save_conversation_summary(10, " new canonical summary ") is True
    assert conversation.summary == "new canonical summary"
    assert session.committed is True


@pytest.mark.anyio
async def test_classifier_rejects_invalid_name_memory(monkeypatch):
    async def fake_generate(_prompt):
        return '{"events":[{"action":"upsert","fact_type":"profile","key":"real_name","value":"fine","confidence":0.99,"durability":"stable"}]}'

    monkeypatch.setattr("services.memory._generate_text_with_memory_llm", fake_generate)

    events = await classify_memory_events("I'm fine.")

    assert events == []


@pytest.mark.anyio
async def test_classifier_treats_temporary_call_me_as_ignored(monkeypatch):
    async def fake_generate(_prompt):
        return '{"events":[{"action":"upsert","fact_type":"profile","key":"preferred_name","value":"Raj","confidence":0.95,"durability":"temporary"}]}'

    monkeypatch.setattr("services.memory._generate_text_with_memory_llm", fake_generate)

    events = await classify_memory_events("Call me Raj for now.")

    assert events == []


def test_build_memory_chunk_from_turn_window():
    chunk = build_memory_chunk(
        7,
        [
            Message(id=1, role="You", content="I like football."),
            Message(id=2, role="Aura", content="Nice, football is fun."),
        ],
    )

    assert chunk["conversation_id"] == 7
    assert chunk["message_start_id"] == 1
    assert chunk["message_end_id"] == 2
    assert "I like football." in chunk["chunk_text"]


def test_memory_fact_candidate_gate():
    assert is_memory_fact_candidate("My name is Raj")
    assert is_memory_fact_candidate("I prefer concise answers")
    assert not is_memory_fact_candidate("Okay, thank you.")
    assert not is_memory_fact_candidate("What is my name?")


@pytest.mark.anyio
@pytest.mark.parametrize("provider", ["google", "groq", "openai"])
async def test_memory_text_inference_uses_only_selected_llm_provider(monkeypatch, provider):
    calls = []

    class GoogleModels:
        async def generate_content(self, **_kwargs):
            calls.append("google")
            return SimpleNamespace(text="google response")

    class Completions:
        def __init__(self, name):
            self.name = name

        async def create(self, **_kwargs):
            calls.append(self.name)
            message = SimpleNamespace(content=f"{self.name} response")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    google_client = SimpleNamespace(aio=SimpleNamespace(models=GoogleModels()))
    groq_client = SimpleNamespace(chat=SimpleNamespace(completions=Completions("groq")))
    openai_client = SimpleNamespace(chat=SimpleNamespace(completions=Completions("openai")))

    monkeypatch.setenv("MEMORY_LLM_PROVIDER", provider)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setattr(memory_service, "_get_google_client", lambda: google_client)
    monkeypatch.setattr(memory_service, "_get_groq_client", lambda: groq_client)
    monkeypatch.setattr(memory_service, "_get_openai_client", lambda: openai_client)
    monkeypatch.setattr(memory_service, "_memory_llm_backoff_until", 0.0)

    response = await memory_service._generate_text_with_memory_llm("prompt")

    assert response == f"{provider} response"
    assert calls == [provider]


@pytest.mark.anyio
async def test_memory_text_inference_reuses_local_llm_runtime(monkeypatch):
    calls = []

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content='{"events":[]}')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    config = SimpleNamespace(
        model="qwen3-4b-local",
        max_tokens=192,
        extra_body={
            "top_k": 20,
            "min_p": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
        },
    )
    runtime = SimpleNamespace(
        config=config,
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=Completions()),
        ),
    )

    monkeypatch.setenv("MEMORY_LLM_PROVIDER", "local")
    monkeypatch.setattr(memory_service, "_get_local_memory_runtime", lambda: runtime)
    monkeypatch.setattr(memory_service, "_memory_llm_backoff_until", 0.0)

    response = await memory_service._generate_text_with_memory_llm("prompt")

    assert response == '{"events":[]}'
    assert calls == [
        {
            "model": "qwen3-4b-local",
            "messages": [{"role": "user", "content": "prompt"}],
            "stream": False,
            "temperature": 0.0,
            "max_tokens": 192,
            "extra_body": config.extra_body,
        }
    ]


@pytest.mark.anyio
async def test_memory_llm_can_be_decoupled_from_local_voice_provider(monkeypatch):
    calls = []

    class Completions:
        async def create(self, **_kwargs):
            calls.append("groq")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="remote"))]
            )

    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("MEMORY_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setattr(
        memory_service,
        "_get_groq_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        ),
    )
    monkeypatch.setattr(memory_service, "_memory_llm_backoff_until", 0.0)

    assert await memory_service._generate_text_with_memory_llm("prompt") == "remote"
    assert calls == ["groq"]


@pytest.mark.anyio
async def test_turn_memory_context_formats_retrieved_chunks(monkeypatch):
    async def fake_retrieve(_user_id, _query, _top_k, query_embedding=None):
        assert query_embedding is None
        return [
            (
                MemoryChunk(
                    user_id=1,
                    conversation_id=3,
                    message_start_id=1,
                    message_end_id=2,
                    chunk_text="User: What is AI?\nAura: AI simulates intelligence.",
                    summary="Discussed AI basics.",
                ),
                0.91,
            )
        ]

    monkeypatch.setattr("services.memory.retrieve_semantic_memories", fake_retrieve)

    context = await build_turn_memory_context(1, "What did we discuss about AI?")

    assert "Relevant long-term episodic memories" in context
    assert "Discussed AI basics." in context


@pytest.mark.anyio
async def test_embed_text_deduplicates_concurrent_requests(monkeypatch):
    calls = 0

    async def fake_embed(value, provider):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return [0.25] * memory_service.MEMORY_EMBEDDING_DIMENSION

    memory_service._embedding_cache.clear()
    memory_service._embedding_inflight.clear()
    monkeypatch.setattr(memory_service, "_embed_uncached", fake_embed)
    monkeypatch.setattr(memory_service, "MEMORY_VECTOR_DB", "pgvector")
    monkeypatch.setenv("MEMORY_EMBEDDING_PROVIDER", "google")

    first, second = await asyncio.gather(
        memory_service.embed_text("same   query"),
        memory_service.embed_text("same query"),
    )
    third = await memory_service.embed_text("same query")

    assert calls == 1
    assert first == second == third


@pytest.mark.anyio
async def test_disabled_embedding_provider_never_calls_remote_provider(monkeypatch):
    calls = 0

    async def fake_embed(*_args):
        nonlocal calls
        calls += 1
        return [0.25] * memory_service.MEMORY_EMBEDDING_DIMENSION

    monkeypatch.setattr(memory_service, "_embed_uncached", fake_embed)
    monkeypatch.setattr(memory_service, "MEMORY_VECTOR_DB", "pgvector")
    monkeypatch.setenv("MEMORY_EMBEDDING_PROVIDER", "disabled")

    assert await memory_service.embed_text("do not send this") is None
    assert calls == 0


def test_embedding_provider_rejects_fake_local_fallback(monkeypatch):
    from core.memory_config import memory_embedding_provider

    monkeypatch.setenv("MEMORY_EMBEDDING_PROVIDER", "local")

    with pytest.raises(ValueError, match="google, openai, or disabled"):
        memory_embedding_provider()
