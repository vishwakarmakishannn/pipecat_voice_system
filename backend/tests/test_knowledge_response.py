import uuid

from services.knowledge.retrieval import format_voice_knowledge_context
from services.knowledge.types import KnowledgeHit, KnowledgeResponse, TurnRoute


def test_voice_context_is_grounded_bounded_policy_handoff():
    response = KnowledgeResponse(
        status="ok",
        query="How do I activate it?",
        normalized_query="How do I activate Mswipe POS?",
        route=TurnRoute("knowledge", 0.9),
        release_id=uuid.uuid4(),
        release_version="2026-08-27.1",
        confidence=0.81,
        hits=[
            KnowledgeHit(
                unit_id=uuid.uuid4(),
                stable_key="mswipe.activation",
                unit_type="procedure",
                title="Activate the terminal",
                answer="First, switch on the terminal.",
                voice_answer="First, switch on the terminal. Tell me when it is on.",
                source_uri="https://www.mswipe.com/activation",
                source_label="Mswipe activation guide",
                product="POS",
                topic="activation",
                requires_live_api=False,
                escalation_required=False,
                ticket_candidates=[],
                score=0.81,
            )
        ],
    )
    context = format_voice_knowledge_context(response)
    assert "release=2026-08-27.1" in context
    assert "one step at a time" in context
    assert "Tell me when it is on" in context
    assert "source=Mswipe activation guide" in context


def test_no_answer_never_formats_as_evidence():
    response = KnowledgeResponse(
        status="no_answer",
        query="unknown",
        normalized_query="unknown",
        route=TurnRoute("knowledge", 0.6),
    )
    assert format_voice_knowledge_context(response) is None
