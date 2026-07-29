import pytest
import asyncio
from types import SimpleNamespace

from core.processors import (
    ContextRetrievalProcessor,
    LatencyBoundaryProcessor,
    ToolRoutingProcessor,
    TurnContextCleanupProcessor,
    TurnLatencyState,
)
from pipecat.frames.frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    OutputTransportMessageFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.aggregators.llm_context import LLMContext
from services.rag import (
    RetrievedRagChunk,
    build_rag_call_payload,
    build_rag_context,
    chunk_link_markdown,
    is_rag_query,
    normalize_retrieval_query,
    should_inject_rag_context,
    should_attempt_rag_retrieval,
    validate_public_http_url,
    retrieve_rag_chunks,
    bump_rag_corpus_version,
    clear_rag_result_cache,
)
import services.rag as rag_service


def test_voice_years_are_canonicalized_without_source_phrase_rules():
    assert normalize_retrieval_query("documentaries from twenty twenty one") == "documentaries from 2021"
    assert normalize_retrieval_query("films from nineteen ninety nine") == "films from 1999"
    assert normalize_retrieval_query("awards from two thousand and twenty two") == "awards from 2022"


def test_lightweight_reranker_prefers_exact_heading_metadata_over_dual_channel_noise():
    exact = {
        "chunk": SimpleNamespace(
            heading_path="2021 Top 5 Documentaries",
            content="Ascension Attica Flee The Rescue Summer of Soul",
        ),
        "vector_similarity": 0.762,
        "text_rank": None,
    }
    noisy = {
        "chunk": SimpleNamespace(
            heading_path="Top 5 Documentaries Archive",
            content=" ".join(f"[Award {index}](https://example.com/{index})" for index in range(20)),
        ),
        "vector_similarity": 0.744,
        "text_rank": 0.6,
    }
    query = "top five documentaries from twenty twenty one"

    assert rag_service._candidate_relevance(exact, query) > rag_service._candidate_relevance(noisy, query)


def test_link_chunking_rejects_dense_navigation_but_keeps_content_sections():
    navigation = " ".join(
        f"[Award category {index}](https://example.com/award/{index})"
        for index in range(20)
    )
    markdown = (
        f"# Archive\n{navigation}\n"
        "# 2021 Top 5 Documentaries\n"
        "[Ascension](https://example.com/ascension) "
        "[Attica](https://example.com/attica) "
        "[Flee](https://example.com/flee) "
        "[The Rescue](https://example.com/rescue) "
        "[Summer of Soul](https://example.com/summer)"
    )

    chunks = chunk_link_markdown(markdown, "Documentary Archive", "https://example.com")

    assert len(chunks) == 1
    assert chunks[0].heading_path == "Documentary Archive > 2021 Top 5 Documentaries"
    assert "Ascension" in chunks[0].content
    assert "https://" not in chunks[0].content


def test_rag_call_payload_contains_full_untruncated_chunk_content():
    content = "Mswipe complaint: " + ("device is not processing transactions. " * 20)
    chunk = RetrievedRagChunk(
        id=355,
        file_id=28,
        filename="issue.pdf",
        content=content,
        page_start=1,
        page_end=1,
        heading_path=None,
        score=0.1,
        text_rank=0.2,
        source_types=("text",),
    )

    payload = build_rag_call_payload("What is the complaint?", [chunk])

    assert len(content) > 240
    assert payload["result"]["chunks"][0]["content"] == content
    assert not payload["result"]["chunks"][0]["content"].endswith("...")


def test_is_rag_query_detects_document_questions():
    assert is_rag_query("What does my PDF say about invoices?")
    assert is_rag_query("Summarize the uploaded report")
    assert is_rag_query("According to my file, what is the deadline?")
    assert is_rag_query("What are the top five documentaries according to the documents?")


def test_is_rag_query_detects_saved_link_questions():
    assert is_rag_query("What does the link say about the 2022 awards?")
    assert is_rag_query("Summarize my saved webpage")
    assert is_rag_query("According to the article, what are the top five documentaries?")


def test_is_rag_query_ignores_general_chat():
    assert not is_rag_query("Who is the president of the USA?")
    assert not is_rag_query("What did we talk about previously?")


def test_pre_router_explicit_mode_bypasses_general_chat():
    assert not should_attempt_rag_retrieval("")
    assert not should_attempt_rag_retrieval("?")
    assert not should_attempt_rag_retrieval("Okay, thank you", mode="explicit")
    assert not should_attempt_rag_retrieval("What is your name?", mode="explicit")
    assert not should_attempt_rag_retrieval("Explain what AI is", mode="explicit")
    assert should_attempt_rag_retrieval("What is the device ID of Rohan Sharma in my PDF?")
    assert should_attempt_rag_retrieval("I mean, according to my documents.")
    assert should_attempt_rag_retrieval("Use my saved link for the answer")
    assert should_attempt_rag_retrieval("What is your name?", mode="hybrid")
    assert should_attempt_rag_retrieval("Okay, thank you", mode="always")


@pytest.mark.anyio
async def test_ready_corpus_gate_bypasses_rag_before_embedding(monkeypatch):
    context = LLMContext(
        messages=[{"role": "user", "content": "What does my document say?"}]
    )
    latency_state = TurnLatencyState(session_id="test")
    retrieval_called = False

    async def no_ready_corpus(_user_id):
        return False

    async def forbidden_retrieval(*_args, **_kwargs):
        nonlocal retrieval_called
        retrieval_called = True
        raise AssertionError("RAG retrieval should have been bypassed")

    processor = ContextRetrievalProcessor(
        1,
        1,
        context,
        latency_state,
        ready_corpus_check=no_ready_corpus,
    )
    delivered = []

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr(
        "core.processors.build_rag_context_with_payload", forbidden_retrieval
    )
    monkeypatch.setattr(processor, "push_frame", capture)
    frame = LLMContextFrame(context)

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert delivered == [frame]
    assert retrieval_called is False
    assert latency_state.rag_considered is True
    assert latency_state.rag_bypassed is True


def test_retrieval_deadline_is_derived_from_concurrent_route_branches(monkeypatch):
    monkeypatch.setattr("core.processors.RAG_VOICE_MEMORY_TIMEOUT_SECONDS", 0.4)
    monkeypatch.setattr("core.processors.RAG_VOICE_RAG_TIMEOUT_SECONDS", 1.2)
    monkeypatch.setattr("core.processors.RAG_VOICE_WEB_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr("core.processors.RAG_VOICE_RETRIEVAL_TIMEOUT_SECONDS", 2.5)

    assert ContextRetrievalProcessor._route_deadline(True, False, False) == pytest.approx(0.5)
    assert ContextRetrievalProcessor._route_deadline(False, True, False) == pytest.approx(1.3)
    assert ContextRetrievalProcessor._route_deadline(True, True, True) == pytest.approx(2.1)


def test_interrupted_response_does_not_remove_fresh_rag_context():
    context = LLMContext(messages=[])
    processor = ContextRetrievalProcessor(1, 1, context)
    message = {"role": "developer", "content": "RAG_GROUNDED_TURN: PDF answer"}
    context.add_message(message)
    processor._dynamic_messages.append(message)

    processor.finish_response()
    assert message in context.messages

    processor.clear_dynamic_context()
    assert message not in context.messages


def test_authoritative_user_turn_start_clears_dynamic_context():
    context = LLMContext(messages=[])
    processor = ContextRetrievalProcessor(1, 1, context)
    message = {"role": "developer", "content": "old turn context"}
    context.add_message(message)
    processor._dynamic_messages.append(message)
    processor.start_user_turn()

    assert message not in context.messages


@pytest.mark.anyio
async def test_completed_user_turn_routes_once_after_multiple_stt_fragments(monkeypatch):
    retrieval_started = asyncio.Event()
    release_retrieval = asyncio.Event()
    delivered = []
    queries = []

    async def fake_rag(_user_id, query, query_embedding=None):
        assert query_embedding is None
        queries.append(query)
        retrieval_started.set()
        await release_retrieval.wait()
        return (f"context for {query}", None)

    context = LLMContext(messages=[])
    processor = ContextRetrievalProcessor(1, 1, context)

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr("core.processors.build_rag_context_with_payload", fake_rag)
    monkeypatch.setattr("core.processors.should_attempt_rag_retrieval", lambda _query: True)
    monkeypatch.setattr(processor, "push_frame", capture)

    first = TranscriptionFrame("first document", "user", "1", finalized=True)
    second = TranscriptionFrame("question", "user", "2", finalized=True)
    await processor.process_frame(first, FrameDirection.DOWNSTREAM)
    await processor.process_frame(second, FrameDirection.DOWNSTREAM)
    assert queries == []

    context.add_message({"role": "user", "content": "first document question"})
    context_frame = LLMContextFrame(context)
    await processor.process_frame(context_frame, FrameDirection.DOWNSTREAM)
    await asyncio.wait_for(retrieval_started.wait(), timeout=0.2)
    release_retrieval.set()
    await asyncio.wait_for(processor._active_task, timeout=0.2)

    assert queries == ["first document question"]
    assert delivered[:2] == [first, second]
    assert delivered[-1] is context_frame
    assert sum(isinstance(item, LLMContextFrame) for item in delivered) == 1


@pytest.mark.anyio
async def test_slow_optional_rag_branch_fails_open_at_its_own_deadline(monkeypatch):
    delivered = []
    processor = ContextRetrievalProcessor(1, 1, LLMContext(messages=[]))
    processor._retrieval_generation = 1

    async def slow_rag(*_args, **_kwargs):
        await asyncio.Event().wait()

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr("core.processors.build_rag_context_with_payload", slow_rag)
    monkeypatch.setattr("core.processors.RAG_VOICE_RAG_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(processor, "push_frame", capture)
    frame = LLMContextFrame(LLMContext(messages=[{"role": "user", "content": "my document"}]))

    await asyncio.wait_for(
        processor._retrieve_and_push(
            frame, "my document", FrameDirection.DOWNSTREAM,
            False, True, False, 1,
        ),
        timeout=0.1,
    )

    assert delivered == [frame]


@pytest.mark.anyio
async def test_failed_retrieval_branch_preserves_successful_branch(monkeypatch):
    context = LLMContext(messages=[])
    delivered = []
    processor = ContextRetrievalProcessor(1, 1, context)
    processor._retrieval_generation = 1

    async def successful_memory(*_args, **_kwargs):
        return "Relevant memory that must survive."

    async def failed_rag(*_args, **_kwargs):
        raise RuntimeError("temporary vector backend failure")

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr("core.processors.embed_text", lambda _query: asyncio.sleep(0, result=[0.5]))
    monkeypatch.setattr("core.processors.build_turn_memory_context", successful_memory)
    monkeypatch.setattr("core.processors.build_rag_context_with_payload", failed_rag)
    monkeypatch.setattr(processor, "push_frame", capture)
    frame = LLMContextFrame(context)

    await processor._retrieve_and_push(
        frame,
        "remember my document",
        FrameDirection.DOWNSTREAM,
        True,
        True,
        False,
        1,
    )

    assert delivered == [frame]
    assert context.messages[-1]["content"] == "Relevant memory that must survive."


@pytest.mark.anyio
async def test_memory_timeout_does_not_cancel_shared_embedding_for_rag(monkeypatch):
    context = LLMContext(messages=[])
    delivered = []
    processor = ContextRetrievalProcessor(1, 1, context)
    processor._retrieval_generation = 1

    async def delayed_embedding(_query):
        await asyncio.sleep(0.03)
        return [0.5]

    async def slow_memory(*_args, query_embedding=None, **_kwargs):
        await query_embedding
        await asyncio.sleep(1)

    async def successful_rag(*_args, query_embedding=None, **_kwargs):
        assert await query_embedding == [0.5]
        return "RAG context survived.", None

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr("core.processors.embed_text", delayed_embedding)
    monkeypatch.setattr("core.processors.build_turn_memory_context", slow_memory)
    monkeypatch.setattr("core.processors.build_rag_context_with_payload", successful_rag)
    monkeypatch.setattr("core.processors.RAG_VOICE_MEMORY_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr("core.processors.RAG_VOICE_RAG_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(processor, "push_frame", capture)
    frame = LLMContextFrame(context)

    await processor._retrieve_and_push(
        frame,
        "What did my document say?",
        FrameDirection.DOWNSTREAM,
        True,
        True,
        False,
        1,
    )

    assert delivered == [frame]
    assert context.messages[-1]["content"] == "RAG context survived."


def test_smart_router_injects_without_document_words_when_vector_match_is_strong():
    chunks = [
        RetrievedRagChunk(
            id=1,
            file_id=1,
            filename="abc.pdf",
            content="Rohan Sharma device ID is A-123.",
            page_start=1,
            page_end=1,
            heading_path=None,
            score=0.81,
            vector_similarity=0.81,
            source_types=("vector",),
        )
    ]

    assert should_inject_rag_context(chunks, query="What is the device ID of Rohan Sharma?")


def test_smart_router_skips_weak_unrelated_matches():
    chunks = [
        RetrievedRagChunk(
            id=1,
            file_id=1,
            filename="abc.pdf",
            content="A weak unrelated candidate.",
            page_start=None,
            page_end=None,
            heading_path=None,
            score=0.10,
            vector_similarity=0.22,
            text_rank=0.0,
            source_types=("vector",),
        )
    ]

    assert not should_inject_rag_context(chunks, query="Who is the president of the USA?")


def test_smart_router_allows_text_only_fallback_when_rank_is_strong():
    chunks = [
        RetrievedRagChunk(
            id=1,
            file_id=1,
            filename="abc.pdf",
            content="Rohan Sharma device ID is A-123.",
            page_start=1,
            page_end=1,
            heading_path=None,
            score=0.50,
            text_rank=0.2,
            source_types=("text",),
        )
    ]

    assert should_inject_rag_context(chunks, query="What is the device ID of Rohan Sharma?")


def test_latency_state_counts_transcript_fragments_as_one_active_turn():
    state = TurnLatencyState(session_id="test")

    state.start_turn()
    state.start_turn()
    assert state.turn_id == 1

    state.finish_turn()
    state.start_turn()
    assert state.turn_id == 2


def test_latency_state_starts_fresh_text_turn_after_previous_audio(monkeypatch):
    state = TurnLatencyState(session_id="test")
    state.start_turn()
    state.first_audio_seen = True
    state.final_stt_at = 10.0
    monkeypatch.setattr("core.processors.time.monotonic", lambda: 20.0)

    state.start_turn()

    assert state.turn_id == 2
    assert state.started_at == 20.0
    assert state.final_stt_at == 20.0
    assert state.speech_stopped_at is None
    assert state.first_audio_seen is False


def test_latency_state_uses_latest_final_fragment_for_completed_turn():
    state = TurnLatencyState(session_id="test")
    state.mark_user_started()
    state.record_final_stt_fragment()
    first_final = state.final_stt_at
    state.record_final_stt_fragment()

    assert state.active is False
    assert state.turn_id == 1
    assert state.final_stt_at >= first_final

    state.start_turn()
    assert state.active is True
    assert state.started_at == state.final_stt_at


def test_latency_state_reports_interim_and_final_stt_counts():
    state = TurnLatencyState(session_id="test")
    state.mark_user_started()
    state.record_interim_stt()
    state.record_interim_stt()
    state.record_final_stt_fragment()
    state.start_turn()

    payload = state.telemetry_payload()
    assert payload["interim_stt_count"] == 2
    assert payload["final_stt_fragment_count"] == 1
    assert "first_interim_stt" in payload["stages_ms"]
    assert "latest_interim_stt" in payload["stages_ms"]
    assert "final_stt" in payload["stages_ms"]
    assert "turn_ready" in payload["stages_ms"]


@pytest.mark.anyio
async def test_llm_latency_distinguishes_first_frame_from_speakable_text(monkeypatch):
    state = TurnLatencyState(session_id="test")
    state.start_turn()
    processor = LatencyBoundaryProcessor(state, "llm")

    async def capture(_frame, _direction):
        return None

    monkeypatch.setattr(processor, "push_frame", capture)
    await processor.process_frame(LLMTextFrame("  **"), FrameDirection.DOWNSTREAM)

    assert state.first_llm_seen is True
    assert state.first_speakable_text_seen is False
    assert "first_llm_text" in state.stage_times
    assert "first_speakable_text" not in state.stage_times

    await processor.process_frame(LLMTextFrame("Hello"), FrameDirection.DOWNSTREAM)

    assert state.first_speakable_text_seen is True
    assert state.first_speakable_text_ms is not None
    assert state.stage_times["first_speakable_text"] >= state.stage_times["first_llm_text"]


def test_latency_turn_identity_starts_with_speech_not_final_transcript():
    state = TurnLatencyState(session_id="test")

    state.mark_user_started()
    assert state.turn_id == 1
    assert state.started_at is None
    state.mark_user_stopped()
    state.start_turn()
    assert state.turn_id == 1
    assert state.started_at == state.final_stt_at

    state.finish_turn()
    state.mark_user_started()
    assert state.turn_id == 2
    assert state.started_at is None


def test_latency_state_reports_endpoint_relative_stage_telemetry():
    state = TurnLatencyState(session_id="test")
    state.mark_user_started()
    state.mark_user_stopped()
    state.start_turn()
    state.mark_stage("retrieval_queued")

    payload = state.telemetry_payload()

    assert payload["basis"] == "user_stopped"
    assert payload["speech_ms"] is not None
    assert payload["stages_ms"]["user_stopped"] == 0.0
    assert payload["stages_ms"]["final_stt"] >= 0.0
    assert payload["stages_ms"]["retrieval_queued"] >= 0.0
    assert payload["server_emitted_unix_ms"] > 0


@pytest.mark.anyio
async def test_first_audio_is_pushed_before_latency_diagnostics(monkeypatch):
    state = TurnLatencyState(session_id="test")
    state.start_turn()
    state.first_llm_seen = True
    from core.processors import LatencyBoundaryProcessor
    boundary = LatencyBoundaryProcessor(state, "tts")
    delivered = []

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr(boundary, "push_frame", capture)
    audio = TTSAudioRawFrame(b"\x00\x00", 24000, 1)
    await boundary.process_frame(audio, FrameDirection.DOWNSTREAM)

    assert delivered[0] is audio
    assert isinstance(delivered[1], OutputTransportMessageFrame)


@pytest.mark.anyio
async def test_answer_audio_uses_turn_release_not_final_stt_as_origin(monkeypatch):
    state = TurnLatencyState(session_id="test")
    state.started_at = 12.0
    state.final_stt_at = 12.0
    state.speech_stopped_at = 10.0
    state.audio_speech_stopped_at = 9.0
    state.active = True
    state.first_llm_seen = True
    state.stage_times["first_speakable_text"] = 18.0
    state.stage_times["tts_request_started"] = 19.0
    boundary = LatencyBoundaryProcessor(state, "tts")
    delivered = []

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr("core.processors.time.monotonic", lambda: 20.0)
    monkeypatch.setattr(boundary, "push_frame", capture)
    await boundary.process_frame(
        TTSAudioRawFrame(b"\x01\x00", 24000, 1),
        FrameDirection.DOWNSTREAM,
    )

    payload = delivered[1].message["data"]["payload"]
    assert payload["answer_audio_ms"] == 11000.0
    assert payload["final_stt_to_audio_ms"] == 8000.0
    assert payload["tts_aggregation_ms"] == 1000.0
    assert payload["tts_provider_ms"] == 1000.0
    assert payload["speakable_to_audio_ms"] == 2000.0


@pytest.mark.anyio
async def test_llm_completion_keeps_realtime_gate_until_first_audio(monkeypatch):
    from core.realtime_gate import realtime_turn_gate

    baseline = realtime_turn_gate.active
    state = TurnLatencyState(session_id="gate-test")
    state.start_turn()
    state.first_llm_seen = True
    retrieval = ContextRetrievalProcessor(1, 1, LLMContext(messages=[]))
    cleanup = TurnContextCleanupProcessor(retrieval, state)
    tts_boundary = LatencyBoundaryProcessor(state, "tts")

    async def discard(_frame, _direction):
        return None

    monkeypatch.setattr(cleanup, "push_frame", discard)
    monkeypatch.setattr(tts_boundary, "push_frame", discard)

    await cleanup.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    assert realtime_turn_gate.active == baseline + 1

    await tts_boundary.process_frame(
        TTSAudioRawFrame(b"\x01\x00", 24000, 1),
        FrameDirection.DOWNSTREAM,
    )
    assert realtime_turn_gate.active == baseline


@pytest.mark.anyio
async def test_no_audio_tts_stop_releases_realtime_gate(monkeypatch):
    from core.realtime_gate import realtime_turn_gate

    baseline = realtime_turn_gate.active
    state = TurnLatencyState(session_id="no-audio")
    state.start_turn()
    state.first_llm_seen = True
    boundary = LatencyBoundaryProcessor(state, "tts")

    async def discard(_frame, _direction):
        return None

    monkeypatch.setattr(boundary, "push_frame", discard)
    await boundary.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)

    assert realtime_turn_gate.active == baseline


def test_vad_endpoint_backdates_confirmation_window(monkeypatch):
    state = TurnLatencyState(session_id="test")
    monkeypatch.setattr("core.processors.time.monotonic", lambda: 20.0)

    state.mark_vad_user_stopped(0.15)

    assert state.audio_speech_stopped_at == 19.85
    assert state.stage_times["audio_speech_stopped"] == 19.85


@pytest.mark.anyio
async def test_hybrid_retrieval_runs_vector_and_text_queries_concurrently(monkeypatch):
    clear_rag_result_cache()
    vector_started = asyncio.Event()
    text_started = asyncio.Event()

    async def fake_embed(_query):
        return [0.1]

    async def fake_vector(_user_id, _embedding):
        vector_started.set()
        await asyncio.wait_for(text_started.wait(), timeout=0.2)
        return []

    async def fake_text(_user_id, _query):
        text_started.set()
        await asyncio.wait_for(vector_started.wait(), timeout=0.2)
        return []

    monkeypatch.setattr(rag_service, "embed_text", fake_embed)
    monkeypatch.setattr(rag_service, "_retrieve_vector_candidates", fake_vector)
    monkeypatch.setattr(rag_service, "_retrieve_text_candidates", fake_text)

    assert await retrieve_rag_chunks(1, "What does my document say?") == []


@pytest.mark.anyio
async def test_text_retrieval_starts_before_embedding_finishes(monkeypatch):
    clear_rag_result_cache()
    text_started = asyncio.Event()

    async def fake_embed(_query):
        await asyncio.wait_for(text_started.wait(), timeout=0.2)
        return [0.1]

    async def fake_vector(_user_id, _embedding):
        return []

    async def fake_text(_user_id, _query):
        text_started.set()
        return []

    monkeypatch.setattr(rag_service, "embed_text", fake_embed)
    monkeypatch.setattr(rag_service, "_retrieve_vector_candidates", fake_vector)
    monkeypatch.setattr(rag_service, "_retrieve_text_candidates", fake_text)

    assert await retrieve_rag_chunks(1, "What does my document say?") == []


@pytest.mark.anyio
async def test_rag_result_cache_uses_corpus_version(monkeypatch):
    calls = 0

    async def fake_retrieve(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return []

    clear_rag_result_cache()
    monkeypatch.setattr(rag_service, "_retrieve_rag_chunks_uncached", fake_retrieve)

    await retrieve_rag_chunks(9, "  My   Document ")
    await retrieve_rag_chunks(9, "my document")
    assert calls == 1

    bump_rag_corpus_version(9)
    await retrieve_rag_chunks(9, "my document")
    assert calls == 2


@pytest.mark.anyio
async def test_combined_memory_and_rag_share_one_embedding(monkeypatch):
    embedding_calls = 0
    seen_embeddings = []
    processor = ContextRetrievalProcessor(1, 1, LLMContext(messages=[]))
    processor._retrieval_generation = 1

    async def fake_embed(_query):
        nonlocal embedding_calls
        embedding_calls += 1
        await asyncio.sleep(0)
        return [0.5]

    async def fake_memory(
        _user_id,
        _query,
        query_embedding=None,
        current_conversation_id=None,
    ):
        assert current_conversation_id == 1
        seen_embeddings.append(await query_embedding)
        return None

    async def fake_rag(_user_id, _query, query_embedding=None):
        seen_embeddings.append(await query_embedding)
        return None, None

    async def capture(_frame, _direction):
        return None

    monkeypatch.setattr("core.processors.embed_text", fake_embed)
    monkeypatch.setattr("core.processors.build_turn_memory_context", fake_memory)
    monkeypatch.setattr("core.processors.build_rag_context_with_payload", fake_rag)
    monkeypatch.setattr(processor, "push_frame", capture)
    frame = TranscriptionFrame("remember my document", "user", "1", finalized=True)

    await processor._retrieve_and_push(
        frame,
        frame.text,
        FrameDirection.DOWNSTREAM,
        True,
        True,
        False,
        1,
    )

    assert embedding_calls == 1
    assert seen_embeddings == [[0.5], [0.5]]


@pytest.mark.anyio
async def test_deterministic_web_search_runs_before_llm_and_suppresses_tool_pass(monkeypatch):
    context = LLMContext(messages=[])
    state = TurnLatencyState(session_id="test")
    state.start_turn()
    delivered = []

    async def fake_web_search(query):
        await asyncio.sleep(0)
        return {"query": query, "answer": "It is sunny."}

    async def capture(frame, _direction):
        delivered.append(frame)

    processor = ContextRetrievalProcessor(
        1,
        1,
        context,
        state,
        web_search=fake_web_search,
    )
    monkeypatch.setattr("core.processors.should_attempt_rag_retrieval", lambda _query: False)
    monkeypatch.setattr(processor, "push_frame", capture)
    context.add_message({"role": "user", "content": "look up the latest weather"})
    frame = LLMContextFrame(context)

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
    task = processor._active_task
    assert task is not None
    await asyncio.wait_for(task, timeout=0.2)

    tool_messages = [
        item.message["data"]
        for item in delivered
        if isinstance(item, OutputTransportMessageFrame)
        and item.message["data"]["type"] == "tool_call"
    ]
    assert [message["payload"]["status"] for message in tool_messages] == [
        "in_progress",
        "completed",
    ]
    assert tool_messages[-1]["payload"]["result"]["answer"] == "It is sunny."
    assert isinstance(delivered[0], OutputTransportMessageFrame)
    assert delivered[0].message["data"]["type"] == "tool_call"
    assert not any(isinstance(item, TTSSpeakFrame) for item in delivered)
    assert delivered[-1] is frame
    assert processor.web_search_resolved is True
    assert processor.tool_filler_emitted is False
    assert state.tool_used is True
    assert any("It is sunny" in message.get("content", "") for message in context.messages)

    router = ToolRoutingProcessor(
        context,
        search_tool=lambda: None,
        issue_tool=lambda: None,
        retrieval=processor,
    )
    assert router.route() == []


@pytest.mark.anyio
async def test_slow_deterministic_web_search_emits_filler_audio_and_transcript(monkeypatch):
    context = LLMContext(messages=[
        {"role": "user", "content": "look up today's weather"}
    ])
    release_search = asyncio.Event()
    delivered = []
    state = TurnLatencyState(session_id="test")
    state.start_turn()

    async def slow_web_search(_query):
        await release_search.wait()
        return {"answer": "Sunny"}

    async def capture(frame, _direction):
        delivered.append(frame)

    processor = ContextRetrievalProcessor(
        1,
        1,
        context,
        state,
        web_search=slow_web_search,
        filler_delay_seconds=0.01,
        filler_enabled=True,
    )
    monkeypatch.setattr("core.processors.should_attempt_rag_retrieval", lambda _query: False)
    monkeypatch.setattr(processor, "push_frame", capture)

    await processor.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.02)

    assert any(isinstance(frame, TTSSpeakFrame) for frame in delivered)
    assert any(
        isinstance(frame, OutputTransportMessageFrame)
        and frame.message["data"]["type"] == "assistant_transcript"
        for frame in delivered
    )
    assert processor.tool_filler_emitted is True
    assert state.tool_filler_spoken is True

    release_search.set()
    await asyncio.wait_for(processor._active_task, timeout=0.2)


@pytest.mark.anyio
async def test_timed_out_deterministic_search_is_not_exposed_to_llm_again(monkeypatch):
    context = LLMContext(messages=[
        {"role": "user", "content": "Who is the current president of India?"}
    ])
    delivered = []

    async def stalled_web_search(_query):
        await asyncio.Event().wait()

    async def capture(frame, _direction):
        delivered.append(frame)

    processor = ContextRetrievalProcessor(
        1,
        1,
        context,
        web_search=stalled_web_search,
        filler_enabled=False,
    )
    monkeypatch.setattr("core.processors.should_attempt_rag_retrieval", lambda _query: False)
    monkeypatch.setattr("core.processors.RAG_VOICE_WEB_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(processor, "push_frame", capture)

    await processor.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
    await asyncio.wait_for(processor._active_task, timeout=0.2)

    assert processor.web_search_attempted is True
    assert processor.web_search_resolved is True
    assert any(
        message.get("role") == "developer" and "timed out" in message.get("content", "")
        for message in context.messages
    )
    router = ToolRoutingProcessor(
        context,
        search_tool=lambda: None,
        issue_tool=lambda: None,
        retrieval=processor,
    )
    assert router.route() == []

    tool_events = [
        frame.message["data"]["payload"]
        for frame in delivered
        if isinstance(frame, OutputTransportMessageFrame)
        and frame.message["data"]["type"] == "tool_call"
    ]
    assert [event["status"] for event in tool_events] == ["in_progress", "completed"]
    assert tool_events[-1]["result"]["status"] == "timeout"


@pytest.mark.anyio
async def test_deterministic_web_search_waits_for_tool_result_after_filler_window(monkeypatch):
    context = LLMContext(messages=[
        {"role": "user", "content": "Who is the current prime minister of India?"}
    ])
    delivered = []

    async def slower_web_search(_query):
        await asyncio.sleep(0.05)
        return {"answer": "Narendra Modi is the current Prime Minister of India."}

    async def capture(frame, _direction):
        delivered.append(frame)

    processor = ContextRetrievalProcessor(
        1,
        1,
        context,
        web_search=slower_web_search,
        filler_delay_seconds=0.01,
        filler_enabled=True,
    )
    monkeypatch.setattr("core.processors.should_attempt_rag_retrieval", lambda _query: False)
    monkeypatch.setattr("core.processors.RAG_VOICE_WEB_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr("core.processors.RAG_VOICE_RETRIEVAL_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(processor, "push_frame", capture)

    await processor.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
    await asyncio.wait_for(processor._active_task, timeout=0.2)

    assert any(isinstance(frame, TTSSpeakFrame) for frame in delivered)
    assert any(
        "Narendra Modi is the current Prime Minister of India"
        in message.get("content", "")
        for message in context.messages
    )
    tool_events = [
        frame.message["data"]["payload"]
        for frame in delivered
        if isinstance(frame, OutputTransportMessageFrame)
        and frame.message["data"]["type"] == "tool_call"
    ]
    assert tool_events[-1]["result"]["answer"] == (
        "Narendra Modi is the current Prime Minister of India."
    )


@pytest.mark.anyio
async def test_fast_rag_cancels_delayed_filler_before_releasing_llm(monkeypatch):
    context = LLMContext(messages=[
        {"role": "user", "content": "Use my saved document"}
    ])
    delivered = []

    async def fast_rag(*_args, **_kwargs):
        return "retrieved context", None

    async def capture(frame, _direction):
        delivered.append(frame)

    processor = ContextRetrievalProcessor(
        1,
        1,
        context,
        filler_delay_seconds=0.01,
        filler_enabled=True,
    )
    monkeypatch.setattr("core.processors.should_attempt_rag_retrieval", lambda _query: True)
    monkeypatch.setattr("core.processors.build_rag_context_with_payload", fast_rag)
    monkeypatch.setattr(processor, "push_frame", capture)
    context_frame = LLMContextFrame(context)

    await processor.process_frame(context_frame, FrameDirection.DOWNSTREAM)
    await asyncio.wait_for(processor._active_task, timeout=0.2)
    await asyncio.sleep(0.02)

    assert delivered == [context_frame]
    assert processor.tool_filler_emitted is False


@pytest.mark.anyio
async def test_rag_call_transcript_is_delivered_before_llm_context(monkeypatch):
    delivered = []
    context = LLMContext(messages=[])
    processor = ContextRetrievalProcessor(1, 7, context)
    processor._retrieval_generation = 1
    rag_payload = {
        "rag_call_id": "rag-test",
        "function_name": "rag_retrieval",
        "arguments": {"query": "saved document"},
        "result": {"chunk_count": 1, "chunks": [{"id": 42}]},
    }

    async def fake_rag(*_args, **_kwargs):
        return "retrieved context", rag_payload

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr("core.processors.build_rag_context_with_payload", fake_rag)
    monkeypatch.setattr("core.task_queue.task_queue.enqueue", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(processor, "push_frame", capture)
    frame = LLMContextFrame(context)

    await processor._retrieve_and_push(
        frame,
        "saved document",
        FrameDirection.DOWNSTREAM,
        False,
        True,
        False,
        1,
    )

    assert isinstance(delivered[0], OutputTransportMessageFrame)
    assert delivered[0].message["data"] == {
        "type": "rag_call",
        "payload": rag_payload,
    }
    assert delivered[1] is frame


@pytest.mark.anyio
async def test_rag_filler_precedes_rag_call_transcript(monkeypatch):
    delivered = []
    context = LLMContext(messages=[
        {"role": "user", "content": "Use my saved document"}
    ])
    state = TurnLatencyState(session_id="test")
    state.start_turn()
    processor = ContextRetrievalProcessor(
        1,
        7,
        context,
        state,
        filler_delay_seconds=0,
        filler_enabled=True,
    )
    rag_payload = {
        "rag_call_id": "rag-ordered",
        "function_name": "rag_retrieval",
        "arguments": {"query": "Use my saved document"},
        "result": {"chunk_count": 1, "chunks": [{"id": 42}]},
    }

    async def fake_rag(*_args, **_kwargs):
        return "retrieved context", rag_payload

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr("core.processors.should_attempt_rag_retrieval", lambda _query: True)
    monkeypatch.setattr("core.processors.build_rag_context_with_payload", fake_rag)
    monkeypatch.setattr("core.task_queue.task_queue.enqueue", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(processor, "push_frame", capture)

    await processor.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
    await asyncio.wait_for(processor._active_task, timeout=0.2)

    assert isinstance(delivered[0], OutputTransportMessageFrame)
    assert delivered[0].message["data"]["type"] == "assistant_transcript"
    assert isinstance(delivered[1], TTSSpeakFrame)
    rag_index = next(
        index
        for index, frame in enumerate(delivered)
        if isinstance(frame, OutputTransportMessageFrame)
        and frame.message["data"]["type"] == "rag_call"
    )
    assert rag_index > 1
    assert state.tool_filler_spoken is True


@pytest.mark.anyio
async def test_generic_current_language_does_not_launch_pre_llm_web_search(monkeypatch):
    context = LLMContext(messages=[
        {"role": "user", "content": "I am currently fixing the searchlight"}
    ])
    searches = []
    delivered = []

    async def web_search(query):
        searches.append(query)
        return {"answer": "unexpected"}

    async def capture(frame, _direction):
        delivered.append(frame)

    processor = ContextRetrievalProcessor(1, 1, context, web_search=web_search)
    monkeypatch.setattr("core.processors.should_attempt_rag_retrieval", lambda _query: False)
    monkeypatch.setattr(processor, "push_frame", capture)
    frame = LLMContextFrame(context)

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert searches == []
    assert processor._active_task is None
    assert delivered == [frame]


@pytest.mark.anyio
async def test_validate_public_http_url_normalizes_bare_domains(monkeypatch):
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 443))],
    )

    assert await validate_public_http_url("example.com/article") == "https://example.com/article"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "data:text/plain,hello", "ftp://example.com/file"])
@pytest.mark.anyio
async def test_validate_public_http_url_rejects_non_http_schemes(url):
    with pytest.raises(ValueError):
        await validate_public_http_url(url)


@pytest.mark.anyio
async def test_validate_public_http_url_rejects_private_addresses(monkeypatch):
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("127.0.0.1", 80))],
    )

    with pytest.raises(ValueError):
        await validate_public_http_url("https://example.com")


def test_chunk_link_markdown_preserves_heading_context():
    chunks = chunk_link_markdown(
        "# Device Report\nRohan Sharma device ID is A-123 and the warranty is active.",
        title="Complaint Receipt",
        final_url="https://example.com/receipt",
    )

    assert chunks
    assert chunks[0].heading_path == "Complaint Receipt > Device Report"
    assert "Heading: Device Report" in chunks[0].embedding_text
    assert "https://example.com/receipt" not in chunks[0].embedding_text


@pytest.mark.anyio
async def test_build_rag_context_formats_retrieved_chunks(monkeypatch):
    async def fake_retrieve(user_id, query, query_embedding=None):
        assert user_id == 7
        assert query == "What does my PDF say about AI?"
        return [
            RetrievedRagChunk(
                id=1,
                file_id=2,
                filename="paper.pdf",
                content="The paper says AI systems can help summarize long documents.",
                page_start=3,
                page_end=3,
                heading_path="Findings",
                score=0.1,
            )
        ]

    monkeypatch.setattr("services.rag.retrieve_rag_chunks", fake_retrieve)

    context = await build_rag_context(7, "What does my PDF say about AI?")

    assert "paper.pdf" in context
    assert "page 3" in context
    assert "Findings" in context
    assert "summarize long documents" in context
    assert "RAG_GROUNDED_TURN" in context
    assert "Do not call the web-search tool" in context


@pytest.mark.anyio
async def test_build_rag_context_formats_link_chunks(monkeypatch):
    async def fake_retrieve(user_id, query, query_embedding=None):
        return [
            RetrievedRagChunk(
                id=1,
                file_id=2,
                filename="Example Article",
                content="The article says the launch date is July 10.",
                page_start=None,
                page_end=None,
                heading_path="Example Article > Launch",
                score=0.1,
                source_type="link",
                url="https://example.com/article",
                title="Example Article",
                site_name="example.com",
            )
        ]

    monkeypatch.setattr("services.rag.retrieve_rag_chunks", fake_retrieve)

    context = await build_rag_context(7, "What does the article say?")

    assert "Link: Example Article <https://example.com/article>" in context
    assert "untrusted retrieved context" in context
    assert "launch date is July 10" in context
