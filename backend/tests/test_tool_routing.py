import pytest
from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from core.processors import ToolRoutingProcessor, TurnLatencyState
from core.tool_schema import tool_schema_hash
from tools.datetime_tool import get_current_datetime
from tools.raise_issue import IssueWorkflowState, manage_issue_draft
from tools.rag import search_uploaded_content
from tools.tavily import tavily_search


async def search_tool(params):
    return None


async def issue_tool(params):
    return None


async def datetime_tool(params):
    return None


async def document_tool(params):
    return None


ALL_TOOLS = [datetime_tool, document_tool, search_tool, issue_tool]


def test_tool_schema_fingerprint_covers_the_real_provider_schemas():
    digest = tool_schema_hash(
        tavily_search,
        search_uploaded_content,
        manage_issue_draft,
        get_current_datetime,
    )

    assert len(digest) == 64
    assert digest == tool_schema_hash(
        tavily_search,
        search_uploaded_content,
        manage_issue_draft,
        get_current_datetime,
    )
    assert digest != tool_schema_hash(
        tavily_search,
        manage_issue_draft,
        get_current_datetime,
    )


def _tool_router(
    context,
    *,
    issue_workflow=None,
    document_available=lambda: True,
):
    return ToolRoutingProcessor(
        context,
        search_tool,
        issue_tool,
        datetime_tool,
        document_tool=document_tool,
        document_tool_available=document_available,
        issue_workflow=issue_workflow,
    )


def _router(text):
    context = LLMContext(messages=[{"role": "user", "content": text}])
    return context, _tool_router(context)


def test_normal_turn_exposes_stable_native_planning_surface():
    context, router = _router("Tell me a short joke")
    original_messages = list(context.messages)

    assert router.route() == ALL_TOOLS
    assert context.tool_choice == "auto"
    assert context.messages == original_messages


def test_document_tool_can_be_suppressed_when_private_retrieval_is_unavailable():
    context = LLMContext(messages=[{"role": "user", "content": "Use my PDF"}])
    router = _tool_router(context, document_available=lambda: False)

    assert router.route() == [datetime_tool, search_tool, issue_tool]


def test_uploaded_document_request_keeps_stable_native_planning_surface():
    context, router = _router("Use my uploaded PDF")

    assert router.route() == ALL_TOOLS
    assert context.tool_choice == "auto"


@pytest.mark.parametrize(
    "text",
    [
        "Who is the current Prime Minister of India?",
        "What is the current price?",
        "What is Taylor Swift's latest album?",
        "Is the CJP protest going on in India?",
        "List five top Hollywood movies released in this year",
        "Use your tools to verify that",
        "You are wrong with the camera specifications",
    ],
)
def test_current_or_verification_requests_use_native_semantic_selection(text):
    context, router = _router(text)

    assert router.route() == ALL_TOOLS
    assert context.tool_choice == "auto"


@pytest.mark.parametrize(
    "text",
    ["I ordered the wrong color", "I am currently polishing a searchlight"],
)
def test_incidental_words_do_not_change_the_native_planning_surface(text):
    context, router = _router(text)

    assert router.route() == ALL_TOOLS


def test_referential_search_keeps_full_history_for_semantic_planning():
    messages = [
        {"role": "user", "content": "What is the CJP protest in India?"},
        {"role": "assistant", "content": "I cannot verify that live."},
        {"role": "user", "content": "Search for it"},
    ]
    context = LLMContext(messages=list(messages))
    router = _tool_router(context)

    assert router.route() == ALL_TOOLS
    assert context.messages == messages
    assert context.tool_choice == "auto"


def test_repeated_routing_never_mutates_conversation_messages():
    context, router = _router("Search for current weather")
    assert router.route() == ALL_TOOLS

    context.add_message({"role": "user", "content": "Tell me a joke"})
    expected_messages = list(context.messages)
    assert router.route() == ALL_TOOLS

    assert context.messages == expected_messages
    assert context.tool_choice == "auto"


@pytest.mark.parametrize(
    "text",
    [
        "Please create an issue for this failure",
        "Can you raise the issue?",
        "Please take care of that complaint",
    ],
)
def test_issue_intent_uses_native_semantic_selection(text):
    context, router = _router(text)

    assert router.route() == ALL_TOOLS
    assert context.tool_choice == "auto"


def test_active_workflow_adds_compact_state_without_hiding_tavily():
    workflow = IssueWorkflowState(
        status="collecting_fields",
        cust_id="C123456",
        mobile="9876543210",
        device_id="MSW12345678",
        description="Transactions fail on every card",
    )
    context = LLMContext(messages=[{"role": "user", "content": "Search for outages"}])
    router = _tool_router(context, issue_workflow=workflow)

    assert router.route() == ALL_TOOLS
    assert context.tool_choice == "required"
    state_messages = [
        message for message in context.messages
        if isinstance(message, dict)
        and str(message.get("content", "")).startswith("ISSUE_WORKFLOW_STATE")
    ]
    assert len(state_messages) == 1
    assert "missing=email" in state_messages[0]["content"]
    assert "Tavily remains available" in state_messages[0]["content"]

    router.route()
    assert sum(
        str(message.get("content", "")).startswith("ISSUE_WORKFLOW_STATE")
        for message in context.messages
        if isinstance(message, dict)
    ) == 1


def test_active_workflow_leaves_spoken_field_interpretation_to_native_planner():
    workflow = IssueWorkflowState(
        status="collecting_fields",
        cust_id="C123456",
        mobile="9876543210",
        device_id="MSW12345678",
        description="Transactions fail on every card",
    )
    context = LLMContext(messages=[{
        "role": "user",
        "content": "Rohan22 at the rate gmail dot com",
    }])
    router = _tool_router(context, issue_workflow=workflow)

    assert router.route() == ALL_TOOLS
    assert context.tool_choice == "required"
    assert workflow.current_user_text == "Rohan22 at the rate gmail dot com"
    assert workflow.current_field_candidates == {"email": "rohan22@gmail.com"}
    assert "normalized from voice dictation" not in context.messages[-1]["content"]


@pytest.mark.parametrize("text", ["Yes", "Proceed", "Are you done?"])
def test_active_workflow_keeps_backend_state_for_native_controller(text):
    workflow = IssueWorkflowState(
        status="awaiting_confirmation",
        cust_id="C123456",
        email="person@example.com",
        mobile="9876543210",
        device_id="MSW12345678",
        description="Transactions fail on every card",
    )
    context = LLMContext(messages=[{"role": "user", "content": text}])
    router = _tool_router(context, issue_workflow=workflow)

    assert router.route() == ALL_TOOLS
    assert context.tool_choice == "required"


def test_active_workflow_keeps_separate_web_request_in_required_tool_set():
    workflow = IssueWorkflowState(status="awaiting_confirmation")
    context = LLMContext(
        messages=[{"role": "user", "content": "Search for current outages"}]
    )
    router = _tool_router(context, issue_workflow=workflow)

    assert router.route() == ALL_TOOLS
    assert context.tool_choice == "required"


def test_submitted_workflow_returns_to_ordinary_native_planning():
    workflow = IssueWorkflowState(status="submitted", issue_id=42)
    context = LLMContext(messages=[{"role": "user", "content": "Tell me a joke"}])
    router = _tool_router(context, issue_workflow=workflow)

    assert router.route() == ALL_TOOLS
    assert context.tool_choice == "auto"


@pytest.mark.anyio
async def test_tool_availability_does_not_count_as_actual_tool_use(monkeypatch):
    context = LLMContext(messages=[{"role": "user", "content": "Tell me a joke"}])
    state = TurnLatencyState(session_id="test")
    state.start_turn()
    processor = _tool_router(context)
    delivered = []

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr(processor, "push_frame", capture)
    context_frame = LLMContextFrame(context)
    await processor.process_frame(context_frame, FrameDirection.DOWNSTREAM)

    assert delivered == [context_frame]
    assert state.tool_used is False
    assert "tool_routed" not in state.stage_times
