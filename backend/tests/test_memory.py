import pytest
import asyncio
import uuid
from types import SimpleNamespace

from services.memory import (
    MemoryBundle,
    build_memory_chunk,
    build_memory_messages,
    build_live_context_messages,
    build_session_memory_context,
    build_turn_memory_context,
    classify_memory_events,
    transcript_to_llm,
    is_memory_fact_candidate,
    is_recall_query,
    maintain_memory_chunks_if_needed,
)
import services.memory as memory_service
from core.models import Call, MemoryChunk, TranscriptEntry, User, UserMemory


@pytest.mark.parametrize(
    "query",
    [
        "What did we discuss last time?",
        "What were we talking about previously?",
        "Find that topic from a previous conversation",
        "Did I mention it in another call?",
    ],
)
def test_cross_call_recall_requires_explicit_prior_session_scope(query):
    assert is_recall_query(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "I said. And one quote.",
        "What did I say?",
        "What were we discussing?",
        "Remember my name",
    ],
)
def test_current_conversation_references_do_not_trigger_cross_call_retrieval(query):
    assert is_recall_query(query) is False


@pytest.mark.anyio
async def test_prior_call_is_loaded_only_on_explicit_recall(monkeypatch):
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
    monkeypatch.setattr(memory_service, "_load_most_recent_prior_call", prior)
    monkeypatch.setattr(memory_service, "VoiceSessionLocal", SessionContext)

    assert await build_turn_memory_context(1, "Tell me a joke") is None
    assert prior_calls == 0
    assert await build_turn_memory_context(1, "What did we discuss last time?") is None
    assert prior_calls == 1


@pytest.mark.anyio
async def test_current_call_recall_prefers_prior_transcripts_without_embedding(
    monkeypatch,
):
    prior = Call(id=uuid.uuid4(), user_id=1, title="Previous call")
    semantic_calls = 0

    async def semantic(*_args, **_kwargs):
        nonlocal semantic_calls
        semantic_calls += 1
        await asyncio.sleep(10)

    async def load_prior(*_args, **_kwargs):
        return prior

    async def load_transcripts(*_args, **_kwargs):
        return [
            TranscriptEntry(
                speaker="You", source="stt_final", text="Tell me about solar panels."
            ),
            TranscriptEntry(
                speaker="Aura", source="llm", text="We discussed rooftop solar costs."
            ),
        ]

    class SessionContext:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr(memory_service, "retrieve_semantic_memories", semantic)
    monkeypatch.setattr(memory_service, "_load_most_recent_prior_call", load_prior)
    monkeypatch.setattr(memory_service, "_load_recent_transcripts", load_transcripts)
    monkeypatch.setattr(memory_service, "VoiceSessionLocal", SessionContext)

    context = await asyncio.wait_for(
        build_turn_memory_context(
            1,
            "What were we talking about previously?",
            current_call_id=prior.id,
        ),
        timeout=0.1,
    )

    assert "Tell me about solar panels." in context
    assert "rooftop solar costs" in context
    assert semantic_calls == 0


def test_new_call_live_context_is_always_fresh():
    bundle = MemoryBundle(
        user=User(id=1, username="kishan", password_hash="x"),
        call=Call(user_id=1, title="New call", summary=""),
        facts=[
            UserMemory(
                key="real_name", value="fine", status="active", fact_type="profile"
            )
        ],
        prior_call=Call(user_id=1, title="Prior", summary="Old summary"),
        prior_recent_transcripts=[
            TranscriptEntry(speaker="You", source="stt_final", text="Old text")
        ],
    )
    assert build_memory_messages(bundle) == []
    assert build_live_context_messages(bundle) == []


def test_only_stable_facts_are_seeded_into_new_call_system_instruction():
    bundle = MemoryBundle(
        user=User(id=1, username="kishan", password_hash="x"),
        call=Call(user_id=1, title="New call"),
        facts=[
            UserMemory(
                key="preferred_name",
                value="Kishan",
                status="active",
                fact_type="profile",
            )
        ],
        prior_call=Call(
            user_id=1, title="Prior", summary="The user prefers jasmine tea."
        ),
    )
    context = build_session_memory_context(bundle)
    assert "preferred_name: Kishan" in context
    assert "jasmine tea" not in context
    assert build_live_context_messages(bundle) == []


def test_transcript_to_llm_maps_supported_speakers_only():
    assert transcript_to_llm(
        TranscriptEntry(speaker="You", source="stt_final", text="hello")
    ) == {
        "role": "user",
        "content": "hello",
    }
    assert transcript_to_llm(
        TranscriptEntry(speaker="Aura", source="llm", text="hi")
    ) == {
        "role": "assistant",
        "content": "hi",
    }
    assert (
        transcript_to_llm(
            TranscriptEntry(speaker="Tool", source="tool_filler", text="{}")
        )
        is None
    )


def test_memory_excludes_recovery_and_simulated_tool_output():
    recovery = TranscriptEntry(
        id=1,
        speaker="Aura",
        source="invalid_output_recovery",
        text="Please try again.",
    )
    simulated = TranscriptEntry(
        id=2,
        speaker="Aura",
        source="llm",
        text='<function=tavily_search>{"query":"x"}</function>',
    )
    valid = TranscriptEntry(
        id=3,
        speaker="Aura",
        source="llm",
        text="The Dell G15 is a gaming laptop.",
    )

    assert transcript_to_llm(recovery) is None
    assert transcript_to_llm(simulated) is None
    assert transcript_to_llm(valid) == {
        "role": "assistant",
        "content": "The Dell G15 is a gaming laptop.",
    }


@pytest.mark.anyio
async def test_chunk_maintenance_no_longer_generates_competing_summary(
    monkeypatch,
):
    recent = [
        TranscriptEntry(
            id=index,
            speaker="You" if index % 2 else "Aura",
            source="llm",
            text=str(index),
        )
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

    async def store(_db, _call, transcripts):
        stored.append(transcripts)

    monkeypatch.setattr(memory_service, "store_memory_chunk", store)
    call = Call(user_id=1, summary="canonical")

    await maintain_memory_chunks_if_needed(Session(), call)

    assert call.summary == "canonical"
    assert stored == [recent]


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
            TranscriptEntry(
                id=1, speaker="You", source="stt_final", text="I like football."
            ),
            TranscriptEntry(
                id=2, speaker="Aura", source="llm", text="Nice, football is fun."
            ),
        ],
    )

    assert chunk["call_id"] == 7
    assert chunk["transcript_start_id"] == 1
    assert chunk["transcript_end_id"] == 2
    assert "I like football." in chunk["chunk_text"]


def test_memory_fact_candidate_gate():
    assert is_memory_fact_candidate("My name is Raj")
    assert is_memory_fact_candidate("I prefer concise answers")
    assert not is_memory_fact_candidate("Okay, thank you.")
    assert not is_memory_fact_candidate("What is my name?")


@pytest.mark.anyio
@pytest.mark.parametrize("provider", ["google", "groq", "openai"])
async def test_memory_text_inference_uses_only_selected_llm_provider(
    monkeypatch, provider
):
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
    openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions("openai"))
    )

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
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
    )
    monkeypatch.setattr(memory_service, "_memory_llm_backoff_until", 0.0)

    assert await memory_service._generate_text_with_memory_llm("prompt") == "remote"
    assert calls == ["groq"]


@pytest.mark.anyio
async def test_turn_memory_context_formats_retrieved_chunks(monkeypatch):
    async def fake_retrieve(
        _user_id,
        _query,
        _top_k,
        query_embedding=None,
        current_call_id=None,
    ):
        assert query_embedding is None
        assert current_call_id is None
        return [
            (
                MemoryChunk(
                    user_id=1,
                    call_id=None,
                    transcript_start_id=1,
                    transcript_end_id=2,
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
async def test_embed_texts_batches_inputs_and_preserves_order(monkeypatch):
    calls = []

    async def fake_batch(values, provider):
        calls.append((provider, list(values)))
        return [
            [float(index)] * memory_service.MEMORY_EMBEDDING_DIMENSION
            for index, _value in enumerate(values, start=1)
        ]

    memory_service._embedding_cache.clear()
    memory_service._embedding_inflight.clear()
    monkeypatch.setattr(memory_service, "_embed_batch_uncached", fake_batch)
    monkeypatch.setattr(memory_service, "MEMORY_VECTOR_DB", "pgvector")
    monkeypatch.setattr(memory_service, "MEMORY_EMBEDDING_BATCH_SIZE", 50)
    monkeypatch.setenv("MEMORY_EMBEDDING_PROVIDER", "google")

    result = await memory_service.embed_texts(
        ["first text", "second text", "first   text"],
        require_all=True,
    )

    assert calls == [("google", ["first text", "second text"])]
    assert result[0] == result[2]
    assert result[0][0] == 1.0
    assert result[1][0] == 2.0


@pytest.mark.anyio
async def test_embed_texts_retries_quota_errors_then_fails_atomically(monkeypatch):
    calls = 0
    delays = []

    async def quota_error(_values, _provider):
        nonlocal calls
        calls += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED; Please retry in 0.25s")

    async def fake_sleep(delay):
        delays.append(delay)

    memory_service._embedding_cache.clear()
    monkeypatch.setattr(memory_service, "_embed_batch_uncached", quota_error)
    monkeypatch.setattr(memory_service, "MEMORY_VECTOR_DB", "pgvector")
    monkeypatch.setattr(memory_service, "MEMORY_EMBEDDING_RETRY_ATTEMPTS", 2)
    monkeypatch.setattr(memory_service.asyncio, "sleep", fake_sleep)
    monkeypatch.setenv("MEMORY_EMBEDDING_PROVIDER", "google")

    with pytest.raises(RuntimeError, match="previous RAG index was preserved"):
        await memory_service.embed_texts(["one", "two"], require_all=True)

    assert calls == 2
    assert delays == [1.0]


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
