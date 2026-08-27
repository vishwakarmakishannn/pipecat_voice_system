import pytest
import asyncio
from pathlib import Path
from types import SimpleNamespace

from core.processors import (
    ContextRetrievalProcessor,
    LatencyBoundaryProcessor,
    ToolRoutingProcessor,
    TurnContextCleanupProcessor,
    TurnLatencyState,
)
from pipecat.frames.frames import (
    FunctionCallInProgressFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.aggregators.llm_context import LLMContext
from services.rag import (
    RetrievedRagChunk,
    build_rag_call_payload,
    build_rag_context,
    chunk_link_markdown,
    compact_rag_result,
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
from services.document_ingestion import (
    canonical_document_from_html,
    canonical_document_to_markdown,
    chunk_canonical_document,
)
from core.context_summary import ContextMutationEpoch, QUERY_SCOPED_CONTEXT_MARKER


def test_voice_years_are_canonicalized_without_source_phrase_rules():
    assert (
        normalize_retrieval_query("documentaries from twenty twenty one")
        == "documentaries from 2021"
    )
    assert (
        normalize_retrieval_query("films from nineteen ninety nine")
        == "films from 1999"
    )
    assert (
        normalize_retrieval_query("awards from two thousand and twenty two")
        == "awards from 2022"
    )


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
            content=" ".join(
                f"[Award {index}](https://example.com/{index})" for index in range(20)
            ),
        ),
        "vector_similarity": 0.744,
        "text_rank": 0.6,
    }
    query = "top five documentaries from twenty twenty one"

    assert rag_service._candidate_relevance(
        exact, query
    ) > rag_service._candidate_relevance(noisy, query)


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


def test_semantic_html_preserves_linked_headings_and_anchor_record_boundaries():
    html = (
        Path(__file__).parent / "fixtures" / "rag" / "linked_heading_archive.html"
    ).read_text(encoding="utf-8")

    document, signals = canonical_document_from_html(
        html, "https://example.com/archive"
    )
    chunks = chunk_canonical_document(
        document,
        max_tokens=400,
        overlap_tokens=40,
        min_content_chars=30,
    )
    target = next(chunk for chunk in chunks if "2022" in (chunk.heading_path or ""))

    assert document.extractor == "semantic_html"
    assert "2022 Top 5 Documentaries" in signals.headings
    assert "2022" in canonical_document_to_markdown(document)
    assert target.heading_path == (
        "Documentary Archive > Top 5 Documentaries Archive > 2022 Top 5 Documentaries"
    )
    assert target.content.splitlines() == [
        "- All the Beauty and the Bloodshed",
        "- All That Breathes",
        "- Descendant",
        "- Turn Every Page - The Adventures of Robert Caro and Robert Gottlieb",
        "- Wildcat",
    ]
    assert "2023" not in target.content
    assert "2021" not in target.content


def test_pdf_parser_configures_and_enforces_document_token_budget(monkeypatch):
    raw_tokenizer = SimpleNamespace(model_max_length=512)
    observed = {}

    class FakeEmbeddingTokenizer:
        def __init__(self, *, tokenizer, max_tokens):
            observed["raw_tokenizer"] = tokenizer
            self.max_tokens = max_tokens

        def count_tokens(self, text):
            return len(text.split())

        def get_max_tokens(self):
            return self.max_tokens

    class FakeHybridChunker:
        def __init__(self, **kwargs):
            observed["chunker_kwargs"] = kwargs

        def chunk(self, *, dl_doc):
            assert dl_doc == "converted-document"
            meta = SimpleNamespace(
                headings=["Section A"],
                doc_items=[SimpleNamespace(prov=[SimpleNamespace(page_no=7)])],
            )
            return [
                SimpleNamespace(
                    text="A sufficiently long PDF paragraph for the parser regression test.",
                    meta=meta,
                )
            ]

        def contextualize(self, chunk):
            return f"Section A\n{chunk.text}"

    class FakeDocumentConverter:
        def convert(self, *, source):
            assert source == "/tmp/example.pdf"
            return SimpleNamespace(document="converted-document")

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda model: raw_tokenizer,
    )
    monkeypatch.setattr(
        "docling_core.transforms.chunker.tokenizer.huggingface.HuggingFaceTokenizer",
        FakeEmbeddingTokenizer,
    )
    monkeypatch.setattr("docling.chunking.HybridChunker", FakeHybridChunker)
    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter", FakeDocumentConverter
    )
    monkeypatch.setattr(rag_service, "RAG_DOCUMENT_CHUNK_TOKENS", 400)
    monkeypatch.setattr(rag_service, "RAG_DOCUMENT_CONTEXT_RESERVE_TOKENS", 64)
    monkeypatch.setattr(
        rag_service,
        "RAG_DOCUMENT_CHUNK_TOKENIZER",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    chunks = rag_service._parse_pdf_to_chunks("/tmp/example.pdf")

    assert raw_tokenizer.model_max_length == 1_000_000
    assert observed["chunker_kwargs"]["tokenizer"].get_max_tokens() == 336
    assert observed["chunker_kwargs"]["omit_header_on_overflow"] is True
    assert chunks[0].page_start == chunks[0].page_end == 7
    assert chunks[0].heading_path == "Section A"
    assert chunks[0].metadata["embedding_tokens"] <= 400


def test_linked_heading_archive_query_reranks_the_exact_section_first():
    html = (
        Path(__file__).parent / "fixtures" / "rag" / "linked_heading_archive.html"
    ).read_text(encoding="utf-8")
    document, _signals = canonical_document_from_html(
        html, "https://example.com/archive"
    )
    chunks = chunk_canonical_document(
        document,
        max_tokens=400,
        overlap_tokens=40,
        min_content_chars=30,
    )
    candidates = [
        {
            "chunk": SimpleNamespace(
                heading_path=chunk.heading_path,
                content=chunk.content,
            ),
            "vector_similarity": 0.72,
            "text_rank": 0.2,
        }
        for chunk in chunks
    ]

    ranked = sorted(
        candidates,
        key=lambda item: rag_service._candidate_relevance(
            item, "top 5 documentaries from 2022 according to the documents"
        ),
        reverse=True,
    )

    assert "2022 Top 5 Documentaries" in ranked[0]["chunk"].heading_path
    assert ranked[0]["chunk"].content.splitlines()[-1] == "- Wildcat"


@pytest.mark.anyio
async def test_static_link_extraction_selects_quality_scored_structural_document(
    monkeypatch,
):
    html = (
        Path(__file__).parent / "fixtures" / "rag" / "linked_heading_archive.html"
    ).read_text(encoding="utf-8")

    async def fake_fetch(_url):
        return html, "https://example.com/archive"

    monkeypatch.setattr(rag_service, "_fetch_html", fake_fetch)

    extracted = await rag_service._extract_link_with_trafilatura(
        "https://example.com/archive"
    )
    chunks = rag_service._parsed_link_chunks(extracted.document)
    target = next(chunk for chunk in chunks if "2022" in (chunk.heading_path or ""))

    assert extracted.extractor == "semantic_html"
    assert extracted.quality_score == pytest.approx(1.0)
    assert extracted.warnings == ()
    assert "2022 Top 5 Documentaries" in extracted.markdown
    assert target.content.count("\n") == 4


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


def test_model_rag_result_is_relevant_compact_and_limited_to_two_chunks():
    result = {
        "chunk_count": 3,
        "chunks": [
            {
                "filename": "rohan.pdf",
                "content": (
                    "General company background with no person details. " * 40
                    + "Rohan Sharma is the regional account manager."
                ),
            },
            {"filename": "team.pdf", "content": "Rohan joined in 2024."},
            {"filename": "private.pdf", "content": "THIRD CHUNK MUST NOT APPEAR"},
        ],
    }

    compact = compact_rag_result(result, "information about Rohan Sharma")

    assert compact["chunk_count"] == 2
    assert len(compact["chunks"]) == 2
    assert "regional account manager" in compact["chunks"][0]["content"]
    assert "General company background" not in compact["chunks"][0]["content"]
    assert "THIRD CHUNK" not in str(compact)
    assert result["chunk_count"] == 3


def test_is_rag_query_detects_document_questions():
    assert is_rag_query("What does my PDF say about invoices?")
    assert is_rag_query("Summarize the uploaded report")
    assert is_rag_query("According to my file, what is the deadline?")
    assert is_rag_query(
        "What are the top five documentaries according to the documents?"
    )


def test_is_rag_query_detects_saved_link_questions():
    assert is_rag_query("What does the link say about the 2022 awards?")
    assert is_rag_query("Summarize my saved webpage")
    assert is_rag_query(
        "According to the article, what are the top five documentaries?"
    )


def test_is_rag_query_ignores_general_chat():
    assert not is_rag_query("Who is the president of the USA?")
    assert not is_rag_query("What did we talk about previously?")


def test_pre_router_explicit_mode_bypasses_general_chat():
    assert not should_attempt_rag_retrieval("")
    assert not should_attempt_rag_retrieval("?")
    assert not should_attempt_rag_retrieval("Okay, thank you", mode="explicit")
    assert not should_attempt_rag_retrieval("What is your name?", mode="explicit")
    assert not should_attempt_rag_retrieval("Explain what AI is", mode="explicit")
    assert should_attempt_rag_retrieval(
        "What is the device ID of Rohan Sharma in my PDF?"
    )
    assert should_attempt_rag_retrieval("I mean, according to my documents.")
    assert should_attempt_rag_retrieval("Use my saved link for the answer")
    assert should_attempt_rag_retrieval("What is your name?", mode="hybrid")
    assert should_attempt_rag_retrieval("Okay, thank you", mode="always")


def test_source_status_is_metadata_only_and_mixed_requests_stay_on_content_path():
    intent = rag_service.source_status_intent(
        "Can you check whether there is a PDF uploaded?"
    )

    assert intent is not None
    assert intent.operation == "availability"
    assert intent.source_type == "pdf"
    assert (
        rag_service.source_status_intent(
            "Is the uploaded report ready, and summarize what it says?"
        )
        is None
    )
    assert (
        rag_service.source_status_intent(
            "I have uploaded a PDF and want information from that."
        )
        is None
    )


def test_lexical_query_keeps_arbitrary_evidence_terms_not_conversation_scaffolding():
    assert rag_service.lexical_retrieval_query(
        "Could you provide information about Amara Okafor from my uploaded PDF?"
    ) == "amara okafor"


@pytest.mark.anyio
async def test_source_status_route_does_not_run_content_retrieval(monkeypatch):
    context = LLMContext(
        messages=[
            {
                "role": "user",
                "content": "Can you check whether there is a PDF uploaded?",
            }
        ]
    )
    status_checks = []

    async def corpus_status(user_id):
        status_checks.append(user_id)
        return {
            "total": 2,
            "ready": 1,
            "by_source_type": {"pdf": {"ready": 1, "processing": 1}},
        }

    async def forbidden_retrieval(*_args, **_kwargs):
        raise AssertionError("source metadata must not run semantic retrieval")

    delivered = []
    processor = ContextRetrievalProcessor(
        7,
        1,
        context,
        corpus_status_check=corpus_status,
        filler_enabled=False,
    )

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr(
        "core.processors.build_rag_context_with_payload",
        forbidden_retrieval,
    )
    monkeypatch.setattr(processor, "push_frame", capture)
    frame = LLMContextFrame(context)

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert status_checks == [7]
    assert delivered == [frame]
    assert processor._active_task is None
    assert processor.document_tool_available() is True
    status_messages = [
        message["content"]
        for message in context.messages
        if isinstance(message, dict)
        and "PRIVATE_SOURCE_STATUS" in str(message.get("content", ""))
    ]
    assert len(status_messages) == 1
    assert "scope=pdf" in status_messages[0]
    assert "total=2" in status_messages[0]
    assert "ready=1" in status_messages[0]


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
    monkeypatch.setattr("core.processors.RAG_VOICE_RETRIEVAL_TIMEOUT_SECONDS", 2.5)

    assert ContextRetrievalProcessor._route_deadline(True, False) == pytest.approx(0.5)
    assert ContextRetrievalProcessor._route_deadline(False, True) == pytest.approx(1.3)
    assert ContextRetrievalProcessor._route_deadline(True, True) == pytest.approx(1.3)


@pytest.mark.anyio
async def test_rag_soft_deadline_continues_and_records_recovery(monkeypatch):
    events = []
    recorder = SimpleNamespace(record=lambda **payload: events.append(payload))
    processor = ContextRetrievalProcessor(
        1,
        1,
        LLMContext(messages=[]),
        diagnostic_recorder=recorder,
    )

    async def completes_after_soft_deadline():
        await asyncio.sleep(0.03)
        return "retrieved context", {"result": {"chunks": []}}

    monkeypatch.setattr("core.processors.RAG_VOICE_RAG_SOFT_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr("core.processors.RAG_VOICE_RAG_TIMEOUT_SECONDS", 0.1)

    result, timed_out = await processor._await_rag_branch(
        completes_after_soft_deadline(),
        query="Rohan Sharma from my PDF",
    )

    assert timed_out is False
    assert result[0] == "retrieved context"
    assert [event["code"] for event in events] == [
        "rag.retrieval_slow",
        "rag.retrieval_slow_recovered",
    ]
    assert events[-1]["recovered"] is True


@pytest.mark.anyio
async def test_source_correction_carries_the_previous_specific_question(monkeypatch):
    context = LLMContext(
        messages=[
            {
                "role": "user",
                "content": "I want information on Rohan Sharma from the video.",
            }
        ]
    )
    processor = ContextRetrievalProcessor(
        1,
        1,
        context,
        ready_corpus_check=lambda _user_id: asyncio.sleep(0, result=True),
        filler_enabled=False,
    )
    delivered = []
    retrieval_queries = []

    async def fake_rag(_user_id, query, **_kwargs):
        retrieval_queries.append(query)
        return None, None

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr("core.processors.build_rag_context_with_payload", fake_rag)
    monkeypatch.setattr(processor, "push_frame", capture)

    processor.start_user_turn()
    await processor.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
    assert retrieval_queries == []

    context.add_message(
        {
            "role": "assistant",
            "content": "I cannot access the video.",
        }
    )
    context.add_message({"role": "user", "content": "I mean PDF"})
    processor.start_user_turn()
    await processor.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
    await asyncio.wait_for(processor._active_task, timeout=0.2)

    assert retrieval_queries == [
        "i want information on rohan sharma.\ni mean pdf"
    ]
    assert rag_service.lexical_retrieval_query(retrieval_queries[0]) == "rohan sharma"
    assert delivered

    context.add_message(
        {
            "role": "assistant",
            "content": "I did not find a sufficiently relevant passage.",
        }
    )
    context.add_message(
        {
            "role": "user",
            "content": "I have uploaded a PDF and want information from that.",
        }
    )
    processor.start_user_turn()
    await processor.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
    await asyncio.wait_for(processor._active_task, timeout=0.2)

    assert retrieval_queries[-1] == (
        "i want information on rohan sharma. i mean pdf\n"
        "i have uploaded a pdf and want information from that."
    )
    assert rag_service.lexical_retrieval_query(retrieval_queries[-1]) == "rohan sharma"
    assert processor._pending_rag_attempt.query == retrieval_queries[-1]


@pytest.mark.anyio
async def test_hard_rag_timeout_installs_truthful_status_and_retry_query(monkeypatch):
    context = LLMContext(messages=[])
    processor = ContextRetrievalProcessor(1, 1, context)
    processor._retrieval_generation = 1
    delivered = []

    async def slow_rag(*_args, **_kwargs):
        await asyncio.Event().wait()

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr("core.processors.build_rag_context_with_payload", slow_rag)
    monkeypatch.setattr("core.processors.RAG_VOICE_RAG_SOFT_TIMEOUT_SECONDS", 0.005)
    monkeypatch.setattr("core.processors.RAG_VOICE_RAG_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(processor, "push_frame", capture)

    await processor._retrieve_and_push(
        LLMContextFrame(context),
        "I mean PDF",
        FrameDirection.DOWNSTREAM,
        False,
        True,
        1,
        rag_query="Rohan Sharma from my PDF",
    )

    status = [
        message["content"]
        for message in context.messages
        if isinstance(message, dict)
        and "RAG_RETRIEVAL_STATUS" in str(message.get("content", ""))
    ]
    assert len(status) == 1
    assert "Do not claim" in status[0]
    assert processor._pending_rag_attempt.query == "Rohan Sharma from my PDF"
    assert len(delivered) == 1
    assert isinstance(delivered[0], LLMContextFrame)

    retry_queries = []

    async def successful_retry(_user_id, query, **_kwargs):
        retry_queries.append(query)
        return (
            "retrieved context",
            {
                "rag_call_id": "rag-retry",
                "result": {
                    "chunk_count": 1,
                    "chunks": [
                        {
                            "filename": "rohan.pdf",
                            "content": "Rohan Sharma profile",
                        }
                    ],
                },
            },
        )

    monkeypatch.setattr(
        "core.processors.build_rag_context_with_payload",
        successful_retry,
    )
    processor.start_user_turn()
    result = await processor.retrieve_for_tool("It is uploaded, check again")

    assert retry_queries == [
        "rohan sharma from my pdf\nit is uploaded, check again"
    ]
    assert result["status"] == "ok"
    assert processor._pending_rag_attempt is None


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


def test_authoritative_user_turn_preserves_latest_rag_evidence_for_followup_tools():
    context = LLMContext(messages=[])
    processor = ContextRetrievalProcessor(1, 1, context)
    message = {"role": "developer", "content": "old turn context"}
    evidence = {
        "result": {
            "chunks": [{"id": 7, "file_id": 3, "content": "Customer C001122"}]
        }
    }
    context.add_message(message)
    processor._dynamic_messages.append(message)
    processor._latest_rag_evidence = evidence

    processor.start_user_turn()

    assert message not in context.messages
    assert processor.latest_rag_evidence is evidence


@pytest.mark.anyio
async def test_direct_followup_receives_bounded_grounded_evidence_anchor(monkeypatch):
    context = LLMContext(messages=[])
    processor = ContextRetrievalProcessor(1, 1, context)
    processor._record_grounded_evidence(
        "Tell me about Rohan Sharma from my file",
        {
            "rag_call_id": "rag-followup",
            "result": {"chunks": [{
                "id": 7,
                "filename": "issue.pdf",
                "content": "Rohan Sharma has device MSW12345678.",
            }]},
        },
    )
    processor.start_user_turn()
    context.add_message({"role": "user", "content": "Can you help with that?"})
    delivered = []

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr(
        "core.processors.should_attempt_rag_retrieval",
        lambda _query: False,
    )
    monkeypatch.setattr(processor, "push_frame", capture)
    frame = LLMContextFrame(context)

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    developer = next(
        message for message in context.messages
        if message.get("role") == "developer"
    )
    assert "GROUNDED_EVIDENCE_ANCHOR" in developer["content"]
    assert "evidence_id=rag-followup" in developer["content"]
    assert "MSW12345678" in developer["content"]
    assert delivered == [frame]


@pytest.mark.anyio
async def test_closure_turn_does_not_receive_grounded_evidence(monkeypatch):
    context = LLMContext(messages=[])
    processor = ContextRetrievalProcessor(1, 1, context)
    processor._record_grounded_evidence(
        "Explain the warranty in my document",
        {
            "rag_call_id": "rag-warranty",
            "result": {
                "chunks": [
                    {
                        "filename": "warranty.pdf",
                        "content": "The warranty lasts for two years.",
                    }
                ]
            },
        },
    )
    processor.start_user_turn()
    context.add_message({"role": "user", "content": "Okay, thank you."})
    delivered = []

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr(
        "core.processors.should_attempt_rag_retrieval",
        lambda _query: False,
    )
    monkeypatch.setattr(processor, "push_frame", capture)
    frame = LLMContextFrame(context)

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert not any(
        "GROUNDED_EVIDENCE_ANCHOR" in str(message.get("content", ""))
        for message in context.messages
        if isinstance(message, dict)
    )
    assert delivered == [frame]


@pytest.mark.anyio
async def test_completed_user_message_advances_state_once_per_semantic_turn(monkeypatch):
    context = LLMContext(messages=[{"role": "user", "content": "Hello"}])
    processor = ContextRetrievalProcessor(1, 1, context)

    async def capture(_frame, _direction):
        return None

    monkeypatch.setattr(
        "core.processors.should_attempt_rag_retrieval",
        lambda _query: False,
    )
    monkeypatch.setattr(processor, "push_frame", capture)
    frame = LLMContextFrame(context)

    processor.start_user_turn()
    processor.start_user_turn()
    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
    assert processor._turn_sequence == 1

    context.add_message({"role": "user", "content": "Hello"})
    await processor.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
    assert processor._turn_sequence == 2


def test_dynamic_context_removal_increments_mutation_epoch_only_when_removed():
    epoch = ContextMutationEpoch()
    context = LLMContext(messages=[])
    processor = ContextRetrievalProcessor(1, 1, context, mutation_epoch=epoch)
    message = {"role": "developer", "content": "old turn context"}
    context.add_message(message)
    processor._dynamic_messages.append(message)

    processor.clear_dynamic_context()
    processor.clear_dynamic_context()

    assert epoch.value == 1


@pytest.mark.anyio
async def test_completed_user_turn_routes_once_after_multiple_stt_fragments(
    monkeypatch,
):
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
    monkeypatch.setattr(
        "core.processors.should_attempt_rag_retrieval", lambda _query: True
    )
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
    frame = LLMContextFrame(
        LLMContext(messages=[{"role": "user", "content": "my document"}])
    )

    await asyncio.wait_for(
        processor._retrieve_and_push(
            frame,
            "my document",
            FrameDirection.DOWNSTREAM,
            False,
            True,
            1,
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

    monkeypatch.setattr(
        "core.processors.embed_text", lambda _query: asyncio.sleep(0, result=[0.5])
    )
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
        1,
    )

    assert delivered == [frame]
    assert context.messages[-1]["content"] == (
        f"{QUERY_SCOPED_CONTEXT_MARKER}\nRelevant memory that must survive."
    )


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
    monkeypatch.setattr(
        "core.processors.build_rag_context_with_payload", successful_rag
    )
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
        1,
    )

    assert delivered == [frame]
    assert context.messages[-1]["content"] == (
        f"{QUERY_SCOPED_CONTEXT_MARKER}\nRAG context survived."
    )


@pytest.mark.anyio
async def test_memory_only_timeout_does_not_leak_unused_rag_coroutine(monkeypatch):
    context = LLMContext(messages=[])
    delivered = []
    processor = ContextRetrievalProcessor(1, 1, context)
    processor._retrieval_generation = 1

    async def slow_memory(*_args, **_kwargs):
        await asyncio.sleep(1)

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr("core.processors.build_turn_memory_context", slow_memory)
    monkeypatch.setattr("core.processors.RAG_VOICE_MEMORY_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(processor, "push_frame", capture)
    frame = LLMContextFrame(context)

    await processor._retrieve_and_push(
        frame,
        "What did we discuss?",
        FrameDirection.DOWNSTREAM,
        True,
        False,
        1,
    )

    assert delivered == [frame]


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

    assert should_inject_rag_context(
        chunks, query="What is the device ID of Rohan Sharma?"
    )


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

    assert not should_inject_rag_context(
        chunks, query="Who is the president of the USA?"
    )


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
            score=0.10,
            text_rank=0.2,
            source_types=("text",),
        )
    ]

    assert should_inject_rag_context(
        chunks, query="What is the device ID of Rohan Sharma?"
    )


def test_text_rank_alone_cannot_inject_a_generic_unrelated_passage():
    chunks = [
        RetrievedRagChunk(
            id=1,
            file_id=1,
            filename="unrelated.pdf",
            content="An ISO standard for generic information systems.",
            page_start=1,
            page_end=1,
            heading_path=None,
            score=0.8,
            text_rank=0.9,
            source_types=("text",),
        )
    ]

    assert not should_inject_rag_context(
        chunks, query="Give me information from the PDF"
    )


@pytest.mark.anyio
async def test_lexical_fallback_is_bounded_when_vector_embedding_is_slow(monkeypatch):
    chunk = SimpleNamespace(
        id=8,
        file_id=5,
        content="Rohan Sharma reported an issue with his Mswipe device.",
        page_start=1,
        page_end=1,
        heading_path=None,
        chunk_index=0,
    )
    rag_file = SimpleNamespace(
        filename="issue.pdf",
        source_type="pdf",
        final_url=None,
        url=None,
        title=None,
        site_name=None,
    )
    vector_called = False
    text_queries = []

    async def slow_embedding(_query):
        await asyncio.sleep(10)

    async def vector_candidates(_user_id, _embedding):
        nonlocal vector_called
        vector_called = True
        return []

    async def text_candidates(_user_id, _query):
        text_queries.append(_query)
        return [(chunk, rag_file, 0.3)]

    monkeypatch.setattr(rag_service, "embed_text", slow_embedding)
    monkeypatch.setattr(
        rag_service,
        "_retrieve_vector_candidates",
        vector_candidates,
    )
    monkeypatch.setattr(
        rag_service,
        "_retrieve_text_candidates",
        text_candidates,
    )
    monkeypatch.setattr(rag_service, "RAG_VECTOR_FUSION_TIMEOUT_SECONDS", 0.02)

    result = await asyncio.wait_for(
        rag_service._retrieve_rag_chunks_uncached(
            1,
            "Give me information on Rohan Sharma from the PDF.",
        ),
        timeout=0.2,
    )

    assert [item.content for item in result] == [chunk.content]
    assert result[0].text_rank == 0.3
    assert result[0].vector_similarity is None
    assert vector_called is False
    assert text_queries == ["rohan sharma"]


@pytest.mark.anyio
async def test_vector_budget_is_absolute_from_retrieval_start(monkeypatch):
    chunk = SimpleNamespace(
        id=9,
        file_id=5,
        content="Rohan submitted the request.",
        page_start=1,
        page_end=1,
        heading_path=None,
        chunk_index=0,
    )
    rag_file = SimpleNamespace(
        filename="issue.pdf",
        source_type="pdf",
        final_url=None,
        url=None,
        title=None,
        site_name=None,
    )

    async def slow_embedding(_query):
        await asyncio.Event().wait()

    async def delayed_text(_user_id, _query):
        await asyncio.sleep(0.06)
        return [(chunk, rag_file, 0.3)]

    monkeypatch.setattr(rag_service, "embed_text", slow_embedding)
    monkeypatch.setattr(rag_service, "_retrieve_text_candidates", delayed_text)
    monkeypatch.setattr(rag_service, "RAG_VECTOR_FUSION_TIMEOUT_SECONDS", 0.08)
    started = asyncio.get_running_loop().time()

    result = await rag_service._retrieve_rag_chunks_uncached(
        1,
        "Rohan Sharma",
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert result
    assert elapsed < 0.12


@pytest.mark.anyio
async def test_completed_embedding_receives_bounded_vector_db_grace(monkeypatch):
    lexical_chunk = SimpleNamespace(
        id=9,
        file_id=5,
        content="General uploaded document content.",
        page_start=1,
        page_end=1,
        heading_path=None,
        chunk_index=0,
    )
    vector_chunk = SimpleNamespace(
        id=10,
        file_id=5,
        content="Rohan Sharma reported a transaction failure.",
        page_start=2,
        page_end=2,
        heading_path=None,
        chunk_index=1,
    )
    rag_file = SimpleNamespace(
        filename="issue.pdf",
        source_type="pdf",
        final_url=None,
        url=None,
        title=None,
        site_name=None,
    )

    async def embedding(_query):
        await asyncio.sleep(0.055)
        return [0.1, 0.2]

    async def vector_candidates(_user_id, _embedding):
        await asyncio.sleep(0.04)
        return [(vector_chunk, rag_file, 0.1)]

    async def text_candidates(_user_id, _query):
        return [(lexical_chunk, rag_file, 0.05)]

    monkeypatch.setattr(rag_service, "embed_text", embedding)
    monkeypatch.setattr(rag_service, "_retrieve_vector_candidates", vector_candidates)
    monkeypatch.setattr(rag_service, "_retrieve_text_candidates", text_candidates)
    monkeypatch.setattr(rag_service, "RAG_VECTOR_FUSION_TIMEOUT_SECONDS", 0.07)
    monkeypatch.setattr(rag_service, "RAG_VECTOR_DB_GRACE_SECONDS", 0.05)

    result = await rag_service._retrieve_rag_chunks_uncached(1, "Rohan Sharma")

    assert any(item.content == vector_chunk.content for item in result)


@pytest.mark.anyio
async def test_tool_reuses_current_turn_pre_llm_evidence_without_retrieval(monkeypatch):
    processor = ContextRetrievalProcessor(1, 1, LLMContext(messages=[]))
    processor._pre_llm_rag_attempted = True
    payload = {
        "rag_call_id": "rag-existing",
        "result": {
            "chunk_count": 1,
            "chunks": [
                {
                    "filename": "rohan.pdf",
                    "page_start": 2,
                    "content": "Rohan Sharma manages the regional account.",
                }
            ],
        },
    }
    processor._record_grounded_evidence("Rohan Sharma from my PDF", payload)

    async def unexpected_retrieval(*_args, **_kwargs):
        raise AssertionError("duplicate retrieval should not run")

    monkeypatch.setattr(
        "core.processors.build_rag_context_with_payload",
        unexpected_retrieval,
    )

    result = await processor.retrieve_for_tool("Rohan Sharma")

    assert result["status"] == "ok"
    assert result["chunks"][0]["filename"] == "rohan.pdf"
    assert "rag_call" not in result
    assert processor.document_tool_available() is True


def test_unanswered_grounded_evidence_survives_recovered_llm_turns():
    processor = ContextRetrievalProcessor(1, 1, LLMContext(messages=[]))
    payload = {
        "rag_call_id": "rag-timeout",
        "result": {
            "chunks": [
                {
                    "filename": "rohan.pdf",
                    "page_start": 4,
                    "content": "Rohan Sharma is the service lead for Pune.",
                }
            ]
        },
    }
    processor._record_grounded_evidence("Rohan Sharma", payload)

    assert "service lead" in processor.timeout_recovery_text()
    processor.finish_response(recovered=True)
    processor.start_user_turn()
    processor.start_user_turn()
    assert processor.unanswered_grounded_evidence() is not None

    processor._install_continuation_evidence()
    processor.finish_response(recovered=False)
    assert processor.grounded_evidence() is not None
    assert processor.unanswered_grounded_evidence() is None


def test_short_rag_followup_is_fused_with_previous_grounded_query():
    query = rag_service.contextualize_retrieval_query(
        "And what about 2022?",
        "Which documentaries are listed in my saved archive?",
    )

    assert "which documentaries" in query
    assert query.endswith("\nand what about 2022?")


def test_specific_rag_query_does_not_inherit_previous_subject():
    query = rag_service.contextualize_retrieval_query(
        "Tell me about Rohan Sharma's device problem",
        "Which documentaries are listed in my saved archive?",
    )

    assert query == "tell me about rohan sharma's device problem"


def test_source_only_followup_replaces_obsolete_scope_without_cue_phrase():
    query = rag_service.contextualize_retrieval_query(
        "I'm in PDF",
        "I want information about Amara Okafor from the video.",
    )

    assert query == "i want information about amara okafor.\ni'm in pdf"
    assert "video" not in query
    assert rag_service.lexical_retrieval_query(query) == "amara okafor"


def test_referential_source_followup_keeps_grounded_subject_despite_extra_words():
    query = rag_service.contextualize_retrieval_query(
        "The PDF has it all.",
        "I want information about Rohan Sharma from the PDF.",
    )

    assert query.startswith("i want information about rohan sharma")
    assert rag_service.lexical_retrieval_query(query) == "rohan sharma"


def test_grounded_retrieval_focus_survives_intervening_direct_turns():
    processor = ContextRetrievalProcessor(1, 1, LLMContext(messages=[]))
    processor._turn_sequence = 1
    processor._record_grounded_evidence(
        "Rohan Sharma from my PDF",
        {"rag_call_id": "rag-focus", "result": {"chunks": []}},
    )

    # Direct conversation and a failed action attempt must not replace the
    # retrieval subject. This focus does not extend action authorization.
    processor._turn_sequence = 4
    previous = processor._recent_query_for_followup()

    assert previous == "Rohan Sharma from my PDF"
    assert processor.grounded_evidence() is None
    contextualized = rag_service.contextualize_retrieval_query(
        "The PDF has it all.", previous
    )
    assert contextualized.startswith("rohan sharma")
    assert rag_service.lexical_retrieval_query(contextualized) == "rohan sharma"


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


def test_latency_state_reports_stt_identity():
    state = TurnLatencyState(
        session_id="test",
        stt_provider="whisper",
        stt_model="small",
    )

    payload = state.telemetry_payload()

    assert payload["stt_provider"] == "whisper"
    assert payload["stt_model"] == "small"


def test_latency_state_captures_vad_diagnostics_at_external_stop():
    state = TurnLatencyState(
        session_id="test",
        vad_diagnostics_getter=lambda: {
            "confidence_at_stop": 0.12,
            "volume_at_stop": 0.08,
        },
    )
    state.mark_user_started()

    state.mark_vad_user_stopped(0.15)

    assert state.telemetry_payload()["vad_diagnostics"] == {
        "confidence_at_stop": 0.12,
        "volume_at_stop": 0.08,
    }


def test_latency_stats_preserve_recovered_llm_outcome():
    state = TurnLatencyState(session_id="test")
    state.start_turn()
    state.response_outcome = "recovered"

    assert state.latency_stats_payload("llm")["outcome"] == "recovered"


@pytest.mark.anyio
async def test_vad_stt_latency_survives_turn_start_interruption(monkeypatch):
    state = TurnLatencyState(session_id="test")
    vad_boundary = LatencyBoundaryProcessor(state, "vad", enable_direct_mode=True)
    stt_boundary = LatencyBoundaryProcessor(state, "stt", enable_direct_mode=True)
    turn_boundary = LatencyBoundaryProcessor(state, "turn", enable_direct_mode=True)
    tts_boundary = LatencyBoundaryProcessor(state, "tts", enable_direct_mode=True)
    delivered = []

    async def discard(_frame, _direction):
        return None

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr(vad_boundary, "push_frame", discard)
    monkeypatch.setattr(stt_boundary, "push_frame", discard)
    monkeypatch.setattr(turn_boundary, "push_frame", discard)
    monkeypatch.setattr(tts_boundary, "push_frame", capture)
    now = [10.0]
    monkeypatch.setattr("core.processors.time.monotonic", lambda: now[0])

    await vad_boundary.process_frame(
        VADUserStartedSpeakingFrame(start_secs=0.15),
        FrameDirection.UPSTREAM,
    )
    await turn_boundary.process_frame(
        UserStartedSpeakingFrame(),
        FrameDirection.DOWNSTREAM,
    )
    await turn_boundary.process_frame(
        InterruptionFrame(),
        FrameDirection.DOWNSTREAM,
    )

    now[0] = 12.0
    await vad_boundary.process_frame(
        VADUserStoppedSpeakingFrame(stop_secs=0.2),
        FrameDirection.UPSTREAM,
    )
    now[0] = 12.5
    await stt_boundary.process_frame(
        TranscriptionFrame("hello", "", "", finalized=True),
        FrameDirection.DOWNSTREAM,
    )
    now[0] = 12.6
    await turn_boundary.process_frame(
        UserStoppedSpeakingFrame(),
        FrameDirection.DOWNSTREAM,
    )
    now[0] = 12.7
    state.start_turn()
    state.first_llm_seen = True
    state.first_speakable_text_seen = True
    state.mark_stage("first_speakable_text", 12.8)
    state.mark_stage("tts_request_started", 12.9)

    now[0] = 13.0
    await tts_boundary.process_frame(
        TTSAudioRawFrame(b"\x01\x00", 24000, 1),
        FrameDirection.DOWNSTREAM,
    )

    assert isinstance(delivered[0], TTSAudioRawFrame)
    payload = delivered[1].message["data"]["payload"]
    assert state.turn_id == 1
    assert payload["stt_latency_ms"] == 700.0
    assert payload["llm_latency_ms"] == 400.0
    assert payload["tts_latency_ms"] == 100.0
    assert payload["answer_audio_ms"] == 1200.0
    assert payload["latency_stage"] == "tts"
    assert payload["latency_complete"] is True


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
    assert (
        state.stage_times["first_speakable_text"] >= state.stage_times["first_llm_text"]
    )


@pytest.mark.anyio
async def test_tool_only_llm_output_enables_tool_speech_latency(monkeypatch):
    state = TurnLatencyState(session_id="test")
    state.start_turn()
    llm_boundary = LatencyBoundaryProcessor(state, "llm")
    tts_boundary = LatencyBoundaryProcessor(state, "tts")
    delivered = []

    async def discard(_frame, _direction):
        return None

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr(llm_boundary, "push_frame", discard)
    monkeypatch.setattr(tts_boundary, "push_frame", capture)

    await llm_boundary.process_frame(
        FunctionCallInProgressFrame("manage_issue_draft", "call-1", {}),
        FrameDirection.DOWNSTREAM,
    )
    await tts_boundary.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
    await tts_boundary.process_frame(
        TTSAudioRawFrame(b"\x01\x00", 24000, 1),
        FrameDirection.DOWNSTREAM,
    )

    assert state.first_llm_seen is True
    assert "first_llm_tool_call" in state.stage_times
    assert state.first_audio_seen is True
    assert any(
        isinstance(frame, OutputTransportMessageUrgentFrame)
        and frame.message["data"]["payload"]["latency_stage"] == "tts"
        for frame in delivered
    )


@pytest.mark.anyio
async def test_latency_stages_are_emitted_urgently_as_they_complete(monkeypatch):
    state = TurnLatencyState(session_id="test")
    turn_boundary = LatencyBoundaryProcessor(state, "turn")
    llm_request_boundary = LatencyBoundaryProcessor(state, "llm_request")
    llm_boundary = LatencyBoundaryProcessor(state, "llm")
    tts_boundary = LatencyBoundaryProcessor(state, "tts")
    turn_delivered = []
    tts_delivered = []

    async def capture_turn(frame, _direction):
        turn_delivered.append(frame)

    async def discard(_frame, _direction):
        return None

    async def capture_tts(frame, _direction):
        tts_delivered.append(frame)

    monkeypatch.setattr(turn_boundary, "push_frame", capture_turn)
    monkeypatch.setattr(llm_request_boundary, "push_frame", discard)
    monkeypatch.setattr(llm_boundary, "push_frame", discard)
    monkeypatch.setattr(tts_boundary, "push_frame", capture_tts)
    now = [10.0]
    monkeypatch.setattr("core.processors.time.monotonic", lambda: now[0])

    state.mark_user_started()
    now[0] = 12.0
    state.mark_vad_user_stopped(0.2)
    now[0] = 12.5
    state.record_final_stt_fragment()
    context_frame = LLMContextFrame(LLMContext(messages=[]))
    await turn_boundary.process_frame(context_frame, FrameDirection.DOWNSTREAM)

    assert isinstance(turn_delivered[0], OutputTransportMessageUrgentFrame)
    assert turn_delivered[1] is context_frame
    stt_payload = turn_delivered[0].message["data"]["payload"]
    assert stt_payload["latency_stage"] == "stt"
    assert stt_payload["latency_complete"] is False
    assert stt_payload["stt_latency_ms"] == 700.0
    assert stt_payload["llm_latency_ms"] is None

    # A repeated context in the same turn must not emit duplicate STT telemetry.
    await turn_boundary.process_frame(context_frame, FrameDirection.DOWNSTREAM)
    assert turn_delivered[2] is context_frame

    now[0] = 12.55
    await llm_request_boundary.process_frame(
        context_frame,
        FrameDirection.DOWNSTREAM,
    )
    now[0] = 12.6
    await llm_boundary.process_frame(
        LLMTextFrame("Hello"),
        FrameDirection.DOWNSTREAM,
    )
    now[0] = 12.9
    started = TTSStartedFrame()
    await tts_boundary.process_frame(started, FrameDirection.DOWNSTREAM)

    assert isinstance(tts_delivered[0], OutputTransportMessageUrgentFrame)
    assert tts_delivered[1] is started
    llm_payload = tts_delivered[0].message["data"]["payload"]
    assert llm_payload["latency_stage"] == "llm"
    assert llm_payload["latency_complete"] is False
    assert llm_payload["stt_latency_ms"] == 700.0
    assert llm_payload["llm_latency_ms"] == 400.0
    assert llm_payload["response_preparation_ms"] == 400.0
    assert llm_payload["turn_release_ms"] == 0.0
    assert llm_payload["pre_llm_ms"] == 50.0
    assert llm_payload["llm_ttft_ms"] == 50.0
    assert llm_payload["tts_latency_ms"] is None

    now[0] = 13.0
    audio = TTSAudioRawFrame(b"\x01\x00", 24000, 1)
    await tts_boundary.process_frame(audio, FrameDirection.DOWNSTREAM)

    assert tts_delivered[2] is audio
    assert isinstance(tts_delivered[3], OutputTransportMessageUrgentFrame)
    final_payload = tts_delivered[3].message["data"]["payload"]
    assert final_payload["latency_stage"] == "tts"
    assert final_payload["latency_complete"] is True
    assert final_payload["tts_latency_ms"] == 100.0
    assert final_payload["answer_audio_ms"] == 1200.0
    assert (
        final_payload["stt_latency_ms"]
        + final_payload["llm_latency_ms"]
        + final_payload["tts_latency_ms"]
        == final_payload["answer_audio_ms"]
    )


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


def test_final_stt_records_provider_finalization_breakdown():
    state = TurnLatencyState(session_id="test")
    state.mark_user_started()

    state.record_final_stt_fragment(
        {
            "finalization_ms": {
                "force_queue_ms": 1.26,
                "force_update_ms": 91.44,
                "vad_downstream_ms": 418,
                "ignored": "not-a-number",
            }
        }
    )

    payload = state.latency_stats_payload("stt")
    assert payload["stt_finalization_ms"] == {
        "force_queue_ms": 1.3,
        "force_update_ms": 91.4,
        "vad_downstream_ms": 418.0,
    }


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
async def test_first_audio_leads_urgent_latency_diagnostics(monkeypatch):
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
    assert isinstance(delivered[1], OutputTransportMessageUrgentFrame)
    assert delivered[1].message["data"]["payload"]["latency_stage"] == "tts"


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

    assert isinstance(delivered[0], TTSAudioRawFrame)
    payload = delivered[1].message["data"]["payload"]
    assert payload["answer_audio_ms"] == 11000.0
    assert payload["final_stt_to_audio_ms"] == 8000.0
    assert payload["stt_latency_ms"] == 3000.0
    assert payload["llm_latency_ms"] == 7000.0
    assert payload["tts_latency_ms"] == 1000.0
    assert payload["tts_aggregation_ms"] == 1000.0
    assert payload["tts_provider_ms"] == 1000.0
    assert payload["speakable_to_audio_ms"] == 2000.0


@pytest.mark.anyio
async def test_first_audio_persists_server_latency_without_browser_callback(
    monkeypatch,
):
    queued = []
    state = TurnLatencyState(session_id="server-session", user_id=9)
    state.start_turn()
    state.first_llm_seen = True
    boundary = LatencyBoundaryProcessor(state, "tts")

    async def discard(_frame, _direction):
        return None

    monkeypatch.setattr(boundary, "push_frame", discard)
    monkeypatch.setattr(
        "core.processors.task_queue.enqueue",
        lambda *args, **kwargs: queued.append((args, kwargs)) or True,
    )

    await boundary.process_frame(
        TTSAudioRawFrame(b"\x01\x00", 24000, 1),
        FrameDirection.DOWNSTREAM,
    )

    assert len(queued) == 1
    args, kwargs = queued[0]
    assert args[1] == 9
    assert args[2]["measurement_source"] == "server"
    assert args[2]["latency_complete"] is True
    assert kwargs["key"] == "voice-latency-9"


@pytest.mark.anyio
async def test_llm_completion_keeps_realtime_gate_until_tts_stops(monkeypatch):
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
    assert realtime_turn_gate.active == baseline + 1

    await tts_boundary.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
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
    monkeypatch.setattr("core.processors.time.time", lambda: 100.0)

    state.mark_vad_user_stopped(0.15, event_timestamp=99.9)

    assert state.audio_speech_stopped_at == pytest.approx(19.75)
    assert state.stage_times["audio_speech_stopped"] == pytest.approx(19.75)


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
        current_call_id=None,
    ):
        assert current_call_id == 1
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
        1,
    )

    assert embedding_calls == 1
    assert seen_embeddings == [[0.5], [0.5]]


@pytest.mark.anyio
async def test_rag_only_route_does_not_disable_query_embedding(monkeypatch):
    default_embedding = object()
    seen_embedding = None
    processor = ContextRetrievalProcessor(1, 1, LLMContext(messages=[]))
    processor._retrieval_generation = 1

    async def fake_rag(
        _user_id,
        _query,
        query_embedding=default_embedding,
    ):
        nonlocal seen_embedding
        seen_embedding = query_embedding
        return None, None

    async def capture(_frame, _direction):
        return None

    monkeypatch.setattr(
        "core.processors.build_rag_context_with_payload",
        fake_rag,
    )
    monkeypatch.setattr(processor, "push_frame", capture)

    await processor._retrieve_and_push(
        TranscriptionFrame("Rohan Sharma from the PDF", "user", "1", finalized=True),
        "Rohan Sharma from the PDF",
        FrameDirection.DOWNSTREAM,
        False,
        True,
        1,
    )

    assert seen_embedding is default_embedding


@pytest.mark.anyio
async def test_context_retrieval_leaves_web_query_planning_to_llm_router(monkeypatch):
    messages = [
        {"role": "user", "content": "I was thinking to buy galaxy a30s"},
        {"role": "assistant", "content": "It has a 48MP main camera."},
        {"role": "user", "content": "You are wrong with the camera specs"},
    ]
    context = LLMContext(messages=list(messages))
    delivered = []

    async def capture(frame, _direction):
        delivered.append(frame)

    processor = ContextRetrievalProcessor(1, 1, context)
    monkeypatch.setattr(
        "core.processors.should_attempt_rag_retrieval", lambda _query: False
    )
    monkeypatch.setattr(processor, "push_frame", capture)
    frame = LLMContextFrame(context)

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert processor._active_task is None
    assert delivered == [frame]
    assert context.messages == messages


@pytest.mark.anyio
async def test_current_price_followup_is_left_to_llm_with_full_history(monkeypatch):
    messages = [
        {"role": "user", "content": "I was thinking to buy galaxy a30s"},
        {"role": "assistant", "content": "It is an older Samsung phone."},
        {"role": "user", "content": "What is the current price?"},
    ]
    context = LLMContext(messages=list(messages))
    state = TurnLatencyState(session_id="test")
    delivered = []
    processor = ContextRetrievalProcessor(1, 1, context, state)

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr(
        "core.processors.should_attempt_rag_retrieval", lambda _query: False
    )
    monkeypatch.setattr(processor, "push_frame", capture)
    frame = LLMContextFrame(context)

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert delivered == [frame]
    assert context.messages == messages
    assert state.tool_used is False


@pytest.mark.anyio
async def test_fast_rag_cancels_delayed_filler_before_releasing_llm(monkeypatch):
    context = LLMContext(
        messages=[{"role": "user", "content": "Use my saved document"}]
    )
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
    monkeypatch.setattr(
        "core.processors.should_attempt_rag_retrieval", lambda _query: True
    )
    monkeypatch.setattr("core.processors.build_rag_context_with_payload", fast_rag)
    monkeypatch.setattr(processor, "push_frame", capture)
    context_frame = LLMContextFrame(context)

    await processor.process_frame(context_frame, FrameDirection.DOWNSTREAM)
    await asyncio.wait_for(processor._active_task, timeout=0.2)
    await asyncio.sleep(0.02)

    assert delivered == [context_frame]
    assert processor.tool_filler_emitted is False


@pytest.mark.anyio
async def test_llm_context_is_delivered_before_rag_diagnostic(monkeypatch):
    delivered = []
    queued = []
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
    monkeypatch.setattr(
        "core.task_queue.task_queue.enqueue",
        lambda *args, **kwargs: queued.append((args, kwargs)) or True,
    )
    monkeypatch.setattr(processor, "push_frame", capture)
    frame = LLMContextFrame(context)

    await processor._retrieve_and_push(
        frame,
        "saved document",
        FrameDirection.DOWNSTREAM,
        False,
        True,
        1,
    )

    assert delivered[0] is frame
    assert isinstance(delivered[1], OutputTransportMessageUrgentFrame)
    delivered_payload = delivered[1].message["data"]["payload"]
    assert delivered[1].message["data"]["type"] == "rag_call"
    assert delivered_payload["rag_call_id"] == rag_payload["rag_call_id"]
    assert delivered_payload["result"] == rag_payload["result"]
    assert delivered_payload["evidence_id"] == "rag-test"
    assert queued[0][0][1] == 7
    assert queued[0][1]["operation_type"] == "rag"


@pytest.mark.anyio
async def test_rag_filler_precedes_rag_call_transcript(monkeypatch):
    delivered = []
    context = LLMContext(
        messages=[{"role": "user", "content": "Use my saved document"}]
    )
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

    monkeypatch.setattr(
        "core.processors.should_attempt_rag_retrieval", lambda _query: True
    )
    monkeypatch.setattr("core.processors.build_rag_context_with_payload", fake_rag)
    monkeypatch.setattr(
        "core.task_queue.task_queue.enqueue", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(processor, "push_frame", capture)

    await processor.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
    await asyncio.wait_for(processor._active_task, timeout=0.2)

    assert isinstance(delivered[0], OutputTransportMessageUrgentFrame)
    assert delivered[0].message["data"]["type"] == "assistant_transcript"
    assert isinstance(delivered[1], TTSSpeakFrame)
    rag_index = next(
        index
        for index, frame in enumerate(delivered)
        if isinstance(frame, OutputTransportMessageUrgentFrame)
        and frame.message["data"]["type"] == "rag_call"
    )
    assert rag_index > 1
    assert state.tool_filler_spoken is True


@pytest.mark.anyio
async def test_generic_current_language_bypasses_context_retrieval(monkeypatch):
    context = LLMContext(
        messages=[{"role": "user", "content": "I am currently fixing the searchlight"}]
    )
    delivered = []

    async def capture(frame, _direction):
        delivered.append(frame)

    processor = ContextRetrievalProcessor(1, 1, context)
    monkeypatch.setattr(
        "core.processors.should_attempt_rag_retrieval", lambda _query: False
    )
    monkeypatch.setattr(processor, "push_frame", capture)
    frame = LLMContextFrame(context)

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert processor._active_task is None
    assert delivered == [frame]


@pytest.mark.anyio
async def test_validate_public_http_url_normalizes_bare_domains(monkeypatch):
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 443))],
    )

    assert (
        await validate_public_http_url("example.com/article")
        == "https://example.com/article"
    )


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "data:text/plain,hello", "ftp://example.com/file"]
)
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


def test_safe_fetch_rejects_redirect_to_private_target(monkeypatch):
    class Headers(dict):
        def get_content_charset(self):
            return None

    monkeypatch.setattr(
        rag_service,
        "_request_pinned",
        lambda *_args: (302, Headers(location="http://127.0.0.1/admin"), b""),
    )

    with pytest.raises(ValueError, match="private or local"):
        rag_service._fetch_bytes(
            "https://example.com/article",
            max_bytes=1024,
            accepted_content_types=("text/html",),
        )


@pytest.mark.anyio
async def test_robots_rules_are_enforced_without_network_robotparser(monkeypatch):
    monkeypatch.setattr(rag_service, "RAG_LINK_RESPECT_ROBOTS", True)
    monkeypatch.setattr(
        rag_service,
        "_fetch_bytes",
        lambda *_args, **_kwargs: (
            b"User-agent: *\nDisallow: /private\n",
            "https://example.com/robots.txt",
            "utf-8",
        ),
    )

    assert await rag_service._robots_allowed("https://example.com/public") is True
    assert (
        await rag_service._robots_allowed("https://example.com/private/report") is False
    )


@pytest.mark.anyio
async def test_extract_link_stops_when_robots_disallow(monkeypatch):
    monkeypatch.setattr(
        rag_service,
        "validate_public_http_url",
        lambda url: asyncio.sleep(0, result=url),
    )
    monkeypatch.setattr(
        rag_service, "_robots_allowed", lambda _url: asyncio.sleep(0, result=False)
    )
    called = False

    async def extractor(_url):
        nonlocal called
        called = True

    monkeypatch.setattr(rag_service, "_extract_link_with_trafilatura", extractor)

    with pytest.raises(ValueError, match="robots.txt"):
        await rag_service.extract_link("https://example.com/private")

    assert called is False


def test_untrusted_link_extraction_defaults_to_non_browser_path():
    from core.rag_config import RAG_ALLOW_BROWSER_EXTRACTOR, RAG_LINK_EXTRACTOR

    assert RAG_LINK_EXTRACTOR == "trafilatura"
    assert RAG_ALLOW_BROWSER_EXTRACTOR is False


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
