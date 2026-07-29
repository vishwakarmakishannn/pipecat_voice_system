import pytest
from pipecat.frames.frames import LLMContextFrame, TTSSpeakFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from core.processors import ToolRoutingProcessor, TurnLatencyState


async def search_tool(params):
    return None


async def issue_tool(params):
    return None


def _router(text):
    context = LLMContext(messages=[{"role": "user", "content": text}])
    return context, ToolRoutingProcessor(context, search_tool, issue_tool)


def test_normal_turn_exposes_no_tools():
    context, router = _router("Tell me a short joke")
    assert router.route() == []
    assert repr(context.tools) == "NOT_GIVEN"


def test_current_information_turn_exposes_only_search():
    _context, router = _router("Look up the latest weather online")
    assert router.route() == [search_tool]


def test_router_avoids_substring_and_generic_current_false_positives():
    _context, router = _router("I am currently polishing a searchlight")
    assert router.route() == []


def test_router_recognizes_browse_web_paraphrase():
    _context, router = _router("Please browse the web for this")
    assert router.route() == [search_tool]


def test_explicit_issue_turn_exposes_only_issue_tool():
    _context, router = _router("Please create an issue for this failure")
    assert router.route() == [issue_tool]


def test_issue_tool_stays_exposed_during_field_collection_continuations():
    context = LLMContext(messages=[
        {"role": "user", "content": "Please create an issue"},
    ])
    router = ToolRoutingProcessor(context, search_tool, issue_tool)
    assert router.route() == [issue_tool]

    context.add_message({
        "role": "assistant",
        "content": "Please provide your email, mobile, customer ID, and device ID.",
    })
    context.add_message({"role": "user", "content": "person@example.com"})

    assert router.route() == [issue_tool]


def test_issue_confirmation_reconstructs_route_for_proceed():
    context = LLMContext(messages=[
        {
            "role": "assistant",
            "content": "All details are valid. Would you like me to raise the complaint?",
        },
        {"role": "user", "content": "Proceed."},
    ])
    router = ToolRoutingProcessor(context, search_tool, issue_tool)

    assert router.route() == [issue_tool]


def test_issue_success_or_cancellation_closes_workflow():
    context = LLMContext(messages=[{"role": "user", "content": "Open an issue"}])
    router = ToolRoutingProcessor(context, search_tool, issue_tool)
    assert router.route() == [issue_tool]

    context.add_message({"role": "assistant", "content": "Issue #42 was successfully raised."})
    context.add_message({"role": "user", "content": "Are you done?"})
    assert router.route() == []

    context.set_messages([{"role": "user", "content": "Open an issue"}])
    assert router.route() == [issue_tool]
    context.add_message({"role": "user", "content": "Never mind, cancel that issue."})
    assert router.route() == []


def test_rag_answer_with_contact_details_does_not_open_issue_workflow():
    context = LLMContext(messages=[
        {"role": "user", "content": "Tell me about Rohan from the PDF"},
        {
            "role": "assistant",
            "content": "Rohan's email is rohan@example.com and his customer ID is 42.",
        },
        {"role": "user", "content": "What documentaries are listed?"},
    ])
    router = ToolRoutingProcessor(context, search_tool, issue_tool)

    assert router.route() == []


def test_unrelated_question_closes_pending_issue_workflow():
    context = LLMContext(messages=[
        {"role": "user", "content": "Tell me about Rohan from the PDF"},
        {
            "role": "assistant",
            "content": "Would you like me to go ahead and raise this complaint?",
        },
        {"role": "user", "content": "Who is the current president of India?"},
    ])
    router = ToolRoutingProcessor(context, search_tool, issue_tool)

    assert router.route() == [search_tool]


@pytest.mark.anyio
async def test_tool_route_does_not_queue_filler_before_llm_context(monkeypatch):
    context = LLMContext(messages=[{"role": "user", "content": "Create an issue for this"}])
    state = TurnLatencyState(session_id="test")
    state.start_turn()
    processor = ToolRoutingProcessor(context, search_tool, issue_tool, latency_state=state)
    delivered = []

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr(processor, "push_frame", capture)
    context_frame = LLMContextFrame(context)
    await processor.process_frame(context_frame, FrameDirection.DOWNSTREAM)

    assert delivered == [context_frame]
    assert state.tool_used is True
