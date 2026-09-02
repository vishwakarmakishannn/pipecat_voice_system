import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

import services.knowledge.retrieval as retrieval
from services.knowledge.retrieval import (
    _field_overlap,
    _lexical_query_text,
    _truncate_voice_text,
    format_voice_knowledge_context,
)
from services.knowledge.types import KnowledgeHit, KnowledgeResponse, TurnRoute

def test_lexical_query_uses_generic_discriminative_or_terms():
    assert _lexical_query_text("How can I contact Mswipe support?") == (
        "contact OR mswipe OR support"
    )
    assert _lexical_query_text("B3") == "b3"


@pytest.mark.anyio
async def test_dense_query_timeout_degrades_to_lexical(monkeypatch):
    degraded = retrieval.QueryEmbeddingResult(
        vector=None,
        failure_class="timeout",
        circuit_state="closed",
        cache_outcome="miss",
        duration_ms=1.0,
    )

    async def slow_embedding(_query):
        return degraded

    monkeypatch.setattr(retrieval, "query_knowledge_embedding", slow_embedding)

    assert await retrieval._query_embedding("Mswipe support") == degraded


def test_field_overlap_rewards_identity_fields_without_intent_vocabulary():
    official_page = SimpleNamespace(
        title="Mswipe Support and Customer Assistance",
        question=None,
        source_label="Mswipe Support",
        source_uri="https://www.mswipe.com/support",
        product=None,
        device=None,
        topic=None,
    )
    product_page = SimpleNamespace(
        title="Mswipe Boombox B3",
        question=None,
        source_label="Mswipe Boombox",
        source_uri="https://www.mswipe.com/in-store-solutions/boombox-b3",
        product="Boombox B3",
        device="Boombox B3",
        topic="payments",
    )

    assert _field_overlap("How can I contact Mswipe support?", official_page) > (
        _field_overlap("How can I contact Mswipe support?", product_page)
    )


def test_voice_text_is_bounded_and_marks_truncation():
    result = _truncate_voice_text("Useful sentence. " * 100, 120)
    assert len(result) <= 120
    assert result.endswith("…")


def test_formatted_voice_context_respects_configured_budget():
    hit = KnowledgeHit(
        unit_id=uuid4(),
        stable_key="demo.product",
        unit_type="definition",
        title="Demo product",
        answer="Long product information. " * 500,
        voice_answer=None,
        source_uri="https://www.mswipe.com/product",
        source_label="Mswipe",
        product=None,
        topic=None,
        requires_live_api=False,
        escalation_required=False,
        ticket_candidates=[],
        score=0.9,
    )
    response = KnowledgeResponse(
        status="ok",
        query="What is the product?",
        normalized_query="What is the product?",
        route=TurnRoute("knowledge", 1.0, ("test",)),
        release_version="demo",
        confidence=0.9,
        hits=[hit],
    )
    context = format_voice_knowledge_context(response)
    assert context is not None
    assert len(context) <= 2400
