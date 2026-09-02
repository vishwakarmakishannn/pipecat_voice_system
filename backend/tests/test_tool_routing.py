import pytest
from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from core.processors import ToolRoutingProcessor
from core.tool_schema import tool_schema_hash
from tools.datetime_tool import get_current_datetime
from tools.mswipe_knowledge import search_mswipe_knowledge
from tools.raise_issue import IssueWorkflowState, manage_issue_draft
from tools.tavily import tavily_search


async def knowledge_tool(params):
    return None


async def issue_tool(params):
    return None


async def datetime_tool(params):
    return None


async def search_tool(params):
    return None


BASE_TOOLS = [knowledge_tool, issue_tool, datetime_tool]
WEB_TOOLS = [*BASE_TOOLS, search_tool]


def test_tool_schema_fingerprint_covers_the_real_provider_schemas():
    tools = (
        search_mswipe_knowledge,
        manage_issue_draft,
        get_current_datetime,
        tavily_search,
    )
    digest = tool_schema_hash(*tools)

    assert len(digest) == 64
    assert digest == tool_schema_hash(*tools)
    assert digest != tool_schema_hash(*tools[:-1])


def _tool_router(context, *, issue_workflow=None, with_web=False):
    return ToolRoutingProcessor(
        context,
        issue_tool,
        knowledge_tool=knowledge_tool,
        datetime_tool=datetime_tool,
        search_tool=search_tool if with_web else None,
        issue_workflow=issue_workflow,
    )


def _router(text, *, with_web=False):
    context = LLMContext(messages=[{"role": "user", "content": text}])
    return context, _tool_router(context, with_web=with_web)


@pytest.mark.parametrize(
    "text",
    [
        "Tell me a short joke",
        "How does Mswipe Soundbox work?",
        "Please create a complaint for this failure",
        "Who is the Prime Minister of India?",
        "I am currently polishing a searchlight",
    ],
)
def test_normal_turn_keeps_one_stable_semantic_planning_surface(text):
    context, router = _router(text)
    original_messages = list(context.messages)

    assert router.route() == BASE_TOOLS
    assert context.tool_choice == "auto"
    assert context.messages == original_messages


def test_web_tool_is_absent_unless_explicitly_configured():
    disabled_context, disabled_router = _router("What happened today?")
    enabled_context, enabled_router = _router(
        "What happened today?",
        with_web=True,
    )

    assert disabled_router.route() == BASE_TOOLS
    assert enabled_router.route() == WEB_TOOLS
    assert disabled_context.tool_choice == "auto"
    assert enabled_context.tool_choice == "auto"


def test_repeated_routing_never_mutates_ordinary_conversation_messages():
    context, router = _router("How does Mswipe help merchants?")
    assert router.route() == BASE_TOOLS

    context.add_message({"role": "user", "content": "What about setup?"})
    expected_messages = list(context.messages)
    assert router.route() == BASE_TOOLS

    assert context.messages == expected_messages
    assert context.tool_choice == "auto"


def test_active_workflow_exposes_only_the_state_machine_tool():
    workflow = IssueWorkflowState(
        status="collecting_fields",
        cust_id="C123456",
        mobile="9876543210",
        device_id="MSW12345678",
        description="Transactions fail on every card",
    )
    context = LLMContext(messages=[{"role": "user", "content": "What next?"}])
    router = _tool_router(context, issue_workflow=workflow, with_web=True)

    assert router.route() == [issue_tool]
    assert context.tool_choice == "required"
    state_messages = [
        message
        for message in context.messages
        if isinstance(message, dict)
        and str(message.get("content", "")).startswith("ISSUE_WORKFLOW_STATE")
    ]
    assert len(state_messages) == 1
    assert "missing=email" in state_messages[0]["content"]
    assert "unrelated turn must use defer" in state_messages[0]["content"]
    assert "Tavily" not in state_messages[0]["content"]

    router.route()
    assert sum(
        str(message.get("content", "")).startswith("ISSUE_WORKFLOW_STATE")
        for message in context.messages
        if isinstance(message, dict)
    ) == 1


def test_active_workflow_keeps_authoritative_turn_without_router_field_parsing():
    workflow = IssueWorkflowState(
        status="collecting_fields",
        cust_id="C123456",
        mobile="9876543210",
        device_id="MSW12345678",
        description="Transactions fail on every card",
    )
    context = LLMContext(
        messages=[{
            "role": "user",
            "content": "Rohan22 at the rate gmail dot com",
        }]
    )
    router = _tool_router(context, issue_workflow=workflow)

    assert router.route() == [issue_tool]
    assert workflow.current_user_text == "Rohan22 at the rate gmail dot com"
    assert workflow.candidate_for_missing_field() is None


@pytest.mark.parametrize("text", ["Yes", "Proceed", "Are you done?", "Tell me a joke"])
def test_active_workflow_always_returns_control_to_backend_state_machine(text):
    workflow = IssueWorkflowState(
        status="awaiting_confirmation",
        cust_id="C123456",
        email="person@example.com",
        mobile="9876543210",
        device_id="MSW12345678",
        description="Transactions fail on every card",
    )
    context = LLMContext(messages=[{"role": "user", "content": text}])
    router = _tool_router(context, issue_workflow=workflow, with_web=True)

    assert router.route() == [issue_tool]
    assert context.tool_choice == "required"


def test_submitted_workflow_returns_to_ordinary_native_planning():
    workflow = IssueWorkflowState(status="submitted", issue_id=42)
    context = LLMContext(messages=[{"role": "user", "content": "Tell me a joke"}])
    router = _tool_router(context, issue_workflow=workflow)

    assert router.route() == BASE_TOOLS
    assert context.tool_choice == "auto"


@pytest.mark.anyio
async def test_tool_availability_does_not_count_as_actual_tool_use(monkeypatch):
    context = LLMContext(messages=[{"role": "user", "content": "Tell me a joke"}])
    processor = _tool_router(context)
    delivered = []

    async def capture(frame, _direction):
        delivered.append(frame)

    monkeypatch.setattr(processor, "push_frame", capture)
    context_frame = LLMContextFrame(context)
    await processor.process_frame(context_frame, FrameDirection.DOWNSTREAM)

    assert delivered == [context_frame]
