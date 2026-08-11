import pytest
from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from core.processors import ToolRoutingProcessor, TurnLatencyState


async def search_tool(params):
    return None


async def issue_tool(params):
    return None


async def datetime_tool(params):
    return None


READ_ONLY_TOOLS = [datetime_tool, search_tool]


def _tool_router(context, *, latency_state=None):
    return ToolRoutingProcessor(
        context,
        search_tool,
        issue_tool,
        datetime_tool,
        latency_state=latency_state,
    )


def _router(text):
    context = LLMContext(messages=[{"role": "user", "content": text}])
    return context, _tool_router(context)


def test_search_routing_has_no_regex_or_keyword_gate():
    assert not hasattr(ToolRoutingProcessor, "SEARCH_PATTERNS")
    assert not hasattr(ToolRoutingProcessor, "needs_web_search")


def test_normal_turn_exposes_clock_and_tavily_with_automatic_choice():
    context, router = _router("Tell me a short joke")
    original_messages = list(context.messages)

    assert router.route() == READ_ONLY_TOOLS
    assert context.tool_choice == "auto"
    assert context.messages == original_messages


@pytest.mark.parametrize(
    "text",
    [
        "Who is the current Prime Minister of India?",
        "What is Taylor Swift's latest album?",
        "Is the CJP protest going on in India?",
        "List five top Hollywood movies released in this year",
        "Use your tools to verify that",
        "You are wrong with the camera specifications",
        "I ordered the wrong color",
        "I am currently polishing a searchlight",
    ],
)
def test_all_meanings_receive_the_same_read_only_tool_availability(text):
    context, router = _router(text)

    assert router.route() == READ_ONLY_TOOLS
    assert context.tool_choice == "auto"


def test_referential_search_keeps_full_history_for_semantic_planning():
    messages = [
        {"role": "user", "content": "What is the CJP protest in India?"},
        {"role": "assistant", "content": "I cannot verify that live."},
        {"role": "user", "content": "Search for it"},
    ]
    context = LLMContext(messages=list(messages))
    router = _tool_router(context)

    assert router.route() == READ_ONLY_TOOLS
    assert context.messages == messages
    assert context.tool_choice == "auto"


def test_repeated_routing_never_mutates_conversation_messages():
    context, router = _router("Search for current weather")
    assert router.route() == READ_ONLY_TOOLS

    context.add_message({"role": "user", "content": "Tell me a joke"})
    expected_messages = list(context.messages)
    assert router.route() == READ_ONLY_TOOLS

    assert context.messages == expected_messages
    assert context.tool_choice == "auto"


def test_explicit_issue_turn_adds_write_tool_without_hiding_read_only_tools():
    context, router = _router("Please create an issue for this failure")

    assert router.route() == [*READ_ONLY_TOOLS, issue_tool]
    assert context.tool_choice == "auto"


def test_issue_tool_stays_exposed_during_field_collection_continuations():
    context = LLMContext(messages=[
        {"role": "user", "content": "Please create an issue"},
    ])
    router = _tool_router(context)
    assert router.route() == [*READ_ONLY_TOOLS, issue_tool]

    context.add_message({
        "role": "assistant",
        "content": "Please provide your email, mobile, customer ID, and device ID.",
    })
    context.add_message({"role": "user", "content": "person@example.com"})

    assert router.route() == [*READ_ONLY_TOOLS, issue_tool]


def test_issue_confirmation_reconstructs_route_for_proceed():
    context = LLMContext(messages=[
        {
            "role": "assistant",
            "content": "All details are valid. Would you like me to raise the complaint?",
        },
        {"role": "user", "content": "Proceed."},
    ])
    router = _tool_router(context)

    assert router.route() == [*READ_ONLY_TOOLS, issue_tool]


def test_issue_success_or_cancellation_closes_write_workflow():
    context = LLMContext(messages=[{"role": "user", "content": "Open an issue"}])
    router = _tool_router(context)
    assert router.route() == [*READ_ONLY_TOOLS, issue_tool]

    context.add_message({"role": "assistant", "content": "Issue #42 was successfully raised."})
    context.add_message({"role": "user", "content": "Are you done?"})
    assert router.route() == READ_ONLY_TOOLS

    context.set_messages([{"role": "user", "content": "Open an issue"}])
    assert router.route() == [*READ_ONLY_TOOLS, issue_tool]
    context.add_message({"role": "user", "content": "Never mind, cancel that issue."})
    assert router.route() == READ_ONLY_TOOLS


def test_rag_answer_with_contact_details_does_not_open_issue_workflow():
    context = LLMContext(messages=[
        {"role": "user", "content": "Tell me about Rohan from the PDF"},
        {
            "role": "assistant",
            "content": "Rohan's email is rohan@example.com and his customer ID is 42.",
        },
        {"role": "user", "content": "What documentaries are listed?"},
    ])
    router = _tool_router(context)

    assert router.route() == READ_ONLY_TOOLS


def test_unrelated_question_closes_pending_issue_workflow():
    context = LLMContext(messages=[
        {"role": "user", "content": "Tell me about Rohan from the PDF"},
        {
            "role": "assistant",
            "content": "Would you like me to go ahead and raise this complaint?",
        },
        {"role": "user", "content": "Who is the current president of India?"},
    ])
    router = _tool_router(context)

    assert router.route() == READ_ONLY_TOOLS


@pytest.mark.anyio
async def test_tool_availability_does_not_count_as_actual_tool_use(monkeypatch):
    context = LLMContext(messages=[{"role": "user", "content": "Tell me a joke"}])
    state = TurnLatencyState(session_id="test")
    state.start_turn()
    processor = _tool_router(context, latency_state=state)
    delivered = []

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr(processor, "push_frame", capture)
    context_frame = LLMContextFrame(context)
    await processor.process_frame(context_frame, FrameDirection.DOWNSTREAM)

    assert delivered == [context_frame]
    assert state.tool_used is False
    assert "tool_routed" in state.stage_times
