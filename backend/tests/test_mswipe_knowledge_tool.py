import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pipecat.processors.aggregators.llm_context import LLMContext

from services.knowledge.types import KnowledgeHit, KnowledgeResponse, TurnRoute
import tools.mswipe_knowledge as knowledge_tool


def _response(*, status="ok", approved_direct=False):
    hit = KnowledgeHit(
        unit_id=uuid4(),
        stable_key="soundbox.overview",
        unit_type="definition",
        title="Mswipe Soundbox",
        answer="Mswipe Soundbox provides audible payment confirmation.",
        voice_answer="It provides audible payment confirmation.",
        source_uri="https://www.mswipe.com/soundbox",
        source_label="Mswipe public website",
        product="Soundbox",
        topic="payments",
        requires_live_api=False,
        escalation_required=False,
        ticket_candidates=[],
        score=0.91,
        metadata=(
            {
                "voice_answer_approved": True,
                "atomic_answer": True,
                "answerability_reviewed": True,
            }
            if approved_direct
            else {}
        ),
        source_span={"block_start": 4, "block_end": 5},
    )
    return KnowledgeResponse(
        status=status,
        query="What does Mswipe Soundbox do?",
        normalized_query="What does Mswipe Soundbox do?",
        route=TurnRoute("knowledge", 1.0, ("test",)),
        release_version="release-1",
        confidence=0.91 if status == "ok" else 0.1,
        hits=[hit] if status == "ok" else [],
        reason=None if status == "ok" else "below_confidence_threshold",
        answer_path="approved_direct" if approved_direct else "grounded_synthesis",
        direct_answer=(
            "**It** provides audible payment confirmation."
            if approved_direct
            else None
        ),
    )


def _params():
    delivered = []
    context = LLMContext(messages=[{"role": "user", "content": "Soundbox?"}])
    context.set_tools([knowledge_tool.search_mswipe_knowledge])
    context.set_tool_choice("auto")
    latency = SimpleNamespace(rag_used=False, rag_latency_ms=None)

    async def capture(result, *, properties=None):
        delivered.append((result, properties))

    params = SimpleNamespace(
        context=context,
        app_resources={"latency_state": latency},
        result_callback=capture,
    )
    return params, context, latency, delivered


def test_schema_exposes_one_semantic_query_argument():
    schema = knowledge_tool.openai_mswipe_knowledge_tool_schema()["function"]

    assert schema["name"] == "search_mswipe_knowledge"
    assert schema["parameters"]["required"] == ["query"]
    assert set(schema["parameters"]["properties"]) == {
        "query",
        "answer_type",
        "requires_live_data",
    }
    assert "semantically" in schema["description"]


@pytest.mark.anyio
async def test_success_returns_bounded_evidence_and_runs_grounded_llm(monkeypatch):
    calls = []

    async def fake_retrieve(query, *, top_k, answer_type, requires_live_data):
        calls.append((query, top_k, answer_type, requires_live_data))
        return _response()

    monkeypatch.setattr(knowledge_tool, "retrieve_knowledge", fake_retrieve)
    params, context, latency, delivered = _params()

    await knowledge_tool.search_mswipe_knowledge(
        params,
        query="  What does   Mswipe Soundbox do? ",
    )

    result, properties = delivered[0]
    assert calls == [(
        "What does Mswipe Soundbox do?",
        knowledge_tool.KNOWLEDGE_TOOL_TOP_K,
        "fact",
        False,
    )]
    assert result["status"] == "ok"
    assert result["evidence"][0]["answer"] == (
        "It provides audible payment confirmation."
    )
    assert result["evidence"][0]["evidence_id"]
    assert result["evidence"][0]["source_span"] == {
        "block_start": 4,
        "block_end": 5,
    }
    assert properties.run_llm is True
    assert repr(context.tools) == "NOT_GIVEN"
    assert repr(context.tool_choice) == "NOT_GIVEN"
    assert latency.rag_used is True
    assert latency.rag_latency_ms is not None


@pytest.mark.anyio
async def test_reviewed_atomic_answer_skips_second_llm_pass(monkeypatch):
    frames = []

    async def fake_retrieve(_query, *, top_k, **_requirements):
        assert top_k == knowledge_tool.KNOWLEDGE_TOOL_TOP_K
        return _response(approved_direct=True)

    class Worker:
        @staticmethod
        async def queue_frames(items):
            frames.extend(items)

    monkeypatch.setattr(knowledge_tool, "retrieve_knowledge", fake_retrieve)
    params, _context, _latency, delivered = _params()
    params.pipeline_worker = Worker()
    params.tool_call_id = "approved-answer"

    await knowledge_tool.search_mswipe_knowledge(
        params,
        query="What does Mswipe Soundbox do?",
    )

    result, properties = delivered[0]
    assert result["answer_path"] == "approved_direct"
    assert properties.run_llm is False
    assert any(
        getattr(frame, "text", None)
        == "It provides audible payment confirmation."
        for frame in frames
    )


@pytest.mark.anyio
async def test_no_answer_never_fabricates_evidence(monkeypatch):
    async def fake_retrieve(_query, *, top_k, **_requirements):
        assert top_k == knowledge_tool.KNOWLEDGE_TOOL_TOP_K
        return _response(status="no_answer")

    monkeypatch.setattr(knowledge_tool, "retrieve_knowledge", fake_retrieve)
    params, _context, latency, delivered = _params()

    await knowledge_tool.search_mswipe_knowledge(params, query="Unknown Mswipe fact")

    result, properties = delivered[0]
    assert result["status"] == "no_answer"
    assert "evidence" not in result
    assert "Do not guess" in result["message"]
    assert properties.run_llm is True
    assert latency.rag_used is False


@pytest.mark.anyio
async def test_empty_query_returns_without_retrieval(monkeypatch):
    async def should_not_run(*_args, **_kwargs):
        raise AssertionError("empty query must not reach retrieval")

    monkeypatch.setattr(knowledge_tool, "retrieve_knowledge", should_not_run)
    params, _context, _latency, delivered = _params()

    await knowledge_tool.search_mswipe_knowledge(params, query="   ")

    assert delivered[0][0]["status"] == "no_answer"


@pytest.mark.anyio
async def test_timeout_returns_safe_structured_result(monkeypatch):
    async def slow_retrieve(*_args, **_kwargs):
        await asyncio.sleep(1)

    monkeypatch.setattr(knowledge_tool, "retrieve_knowledge", slow_retrieve)
    monkeypatch.setattr(knowledge_tool, "KNOWLEDGE_TOOL_TIMEOUT_SECONDS", 0.001)
    params, _context, latency, delivered = _params()

    await knowledge_tool.search_mswipe_knowledge(params, query="Mswipe setup")

    assert delivered[0][0]["status"] == "timeout"
    assert latency.rag_used is False
