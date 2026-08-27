import asyncio
import importlib
from types import SimpleNamespace

import pytest
from pipecat.frames.frames import OutputTransportMessageFrame, TTSSpeakFrame
from pipecat.processors.aggregators.llm_context import LLMContext

from core.processors import ContextRetrievalProcessor, TurnLatencyState


raise_issue_module = importlib.import_module("tools.raise_issue")


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("It is rohan22@gmail.com", "rohan22@gmail.com"),
        ("Rohan22 at the rate gmail.com", "rohan22@gmail.com"),
        ("rohan22 at gmail dot com", "rohan22@gmail.com"),
        (
            "The email address is Rohan at the red Gmail.com",
            "rohan@gmail.com",
        ),
        ("Use one@example.com or two@example.com", None),
        ("That is not an email", None),
    ],
)
def test_spoken_email_normalization_is_unambiguous(spoken, expected):
    assert raise_issue_module.normalize_spoken_email(spoken) == expected


def test_missing_email_is_assembled_from_adjacent_finalized_fragments():
    workflow = raise_issue_module.IssueWorkflowState(
        status="collecting_fields",
        cust_id="C001122",
        mobile="9876543210",
        device_id="MSW12345678",
        description="Card transactions fail for debit and credit cards.",
    )

    workflow.observe_user_turn("The email address is Rohan at the red")
    assert workflow.current_field_candidates == {}

    workflow.observe_user_turn("Gmail.com")
    assert workflow.current_user_text == (
        "The email address is Rohan at the red Gmail.com"
    )
    assert workflow.current_field_candidates == {"email": "rohan@gmail.com"}


@pytest.mark.anyio
async def test_router_parser_does_not_write_a_field_without_an_llm_argument():
    delivered = []
    workflow = raise_issue_module.IssueWorkflowState(
        status="collecting_fields",
        cust_id="C001122",
        mobile="9876543210",
        device_id="MSW12345678",
        description="Card transactions fail for debit and credit cards.",
    )
    workflow.observe_user_turn("Rohan22 at the rate gmail dot com")

    async def capture(result, *, properties=None):
        delivered.append((result, properties))

    params = SimpleNamespace(
        function_name="manage_issue_draft",
        tool_call_id="spoken-email",
        arguments={"operation": "update"},
        pipeline_worker=None,
        # Reproduces the live failure mode: the tool context can lag behind the
        # finalized user turn, while router-owned workflow state is current.
        context=LLMContext(messages=[{"role": "user", "content": "Proceed"}]),
        app_resources={"issue_workflow": workflow},
        result_callback=capture,
    )

    await raise_issue_module.manage_issue_draft(params, operation="update")

    result, properties = delivered[0]
    assert result["status"] == "collecting_fields"
    assert result["draft"]["email"] is None
    assert "email" not in result["provenance"]
    assert result["missing_fields"] == ["email"]
    assert properties.run_llm is False


@pytest.mark.anyio
async def test_schema_valid_llm_interpretation_is_not_vetoed_by_text_reparsing():
    delivered = []
    workflow = raise_issue_module.IssueWorkflowState(
        status="collecting_fields",
        cust_id="C001122",
        mobile="9876543210",
        device_id="MSW12345678",
        description="Card transactions fail for debit and credit cards.",
    )
    workflow.observe_user_turn(
        "Yes, the email is rohan22 at the rate gmail.com. Thank you."
    )

    async def capture(result, *, properties=None):
        delivered.append((result, properties))

    params = SimpleNamespace(
        function_name="manage_issue_draft",
        tool_call_id="llm-interpreted-email",
        arguments={"operation": "update", "email": "rohan22@gmail.com"},
        pipeline_worker=None,
        context=LLMContext(messages=[{
            "role": "user",
            "content": "Yes, the email is rohan22 at the rate gmail.com. Thank you.",
        }]),
        app_resources={"issue_workflow": workflow},
        result_callback=capture,
    )

    await raise_issue_module.manage_issue_draft(
        params,
        operation="update",
        email="rohan22@gmail.com",
    )

    result, properties = delivered[0]
    assert result["status"] == "awaiting_confirmation"
    assert result["draft"]["email"] == "rohan22@gmail.com"
    assert result["provenance"]["email"] == "llm_interpreted"
    assert result["invalid_fields"] == []
    assert properties.run_llm is False


@pytest.mark.anyio
async def test_raise_issue_uses_flush_without_post_commit_refresh(monkeypatch):
    events = []
    results = []
    frames = []

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def add(self, issue):
            self.issue = issue
            events.append("add")

        async def flush(self):
            self.issue.id = 42
            events.append("flush")

        async def commit(self):
            events.append("commit")

        async def refresh(self, _issue):
            raise AssertionError("post-commit refresh must not be used")

    class Worker:
        @staticmethod
        async def queue_frame(frame):
            frames.append(frame)

        @staticmethod
        async def queue_frames(items):
            frames.extend(items)

    class Params:
        function_name = "raise_issue"
        tool_call_id = "issue-call-1"
        arguments = {
            "cust_id": "C123456",
            "email": "person@example.com",
            "mobile": "9876543210",
            "device_id": "MSW12345678",
            "description": "Intermittent connection",
        }
        pipeline_worker = Worker()
        app_resources = {"latency_state": TurnLatencyState(session_id="test")}

        @staticmethod
        async def result_callback(result):
            results.append(result)

    monkeypatch.setattr(raise_issue_module, "VoiceSessionLocal", FakeSession)

    await raise_issue_module.raise_issue(
        Params(),
        cust_id="C123456",
        email="person@example.com",
        mobile="9876543210",
        device_id="MSW12345678",
        description="Intermittent connection",
    )

    assert events == ["add", "flush", "commit"]
    assert results == [{"status": "success", "message": "Issue #42 has been successfully raised."}]
    assert isinstance(frames[0], OutputTransportMessageFrame)
    assert frames[0].message["data"]["type"] == "tool_call"
    assert not any(isinstance(frame, TTSSpeakFrame) for frame in frames)
    messages = [
        frame.message["data"]
        for frame in frames
        if isinstance(frame, OutputTransportMessageFrame)
        and frame.message["data"]["type"] == "tool_call"
    ]
    assert [message["payload"]["status"] for message in messages] == [
        "in_progress",
        "completed",
    ]
    assert messages[0]["payload"]["tool_call_id"] == "issue-call-1"
    assert messages[-1]["payload"]["result"] == results[-1]


@pytest.mark.anyio
async def test_semantic_issue_draft_uses_rag_evidence_then_requires_confirmation(
    monkeypatch,
):
    inserts = []
    results = []
    frames = []
    workflow = raise_issue_module.IssueWorkflowState()

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def add(self, issue):
            self.issue = issue
            inserts.append(issue)

        async def flush(self):
            self.issue.id = 73

        async def commit(self):
            return None

    class Worker:
        @staticmethod
        async def queue_frame(frame):
            frames.append(frame)

        @staticmethod
        async def queue_frames(items):
            frames.extend(items)

    evidence = {
        "result": {
            "chunks": [
                {
                    "id": 12,
                    "file_id": 4,
                    "source_type": "pdf",
                    "filename": "issue.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "content": (
                        "Rohan Sharma customer ID C001122 mobile 9876543210 "
                        "device MSW12345678. Card transactions fail with a "
                        "Transaction Failed error."
                    ),
                }
            ]
        }
    }
    retrieval = SimpleNamespace(latest_rag_evidence=evidence)
    context = LLMContext(messages=[
        {"role": "user", "content": "Can you raise the issue?"},
    ])

    class Params:
        function_name = "manage_issue_draft"
        tool_call_id = "workflow-call"
        arguments = {"operation": "start"}
        pipeline_worker = Worker()
        app_resources = {
            "issue_workflow": workflow,
            "context_retrieval": retrieval,
        }

        @staticmethod
        async def result_callback(result, *, properties=None):
            results.append((result, properties))

    Params.context = context
    monkeypatch.setattr(raise_issue_module, "VoiceSessionLocal", FakeSession)

    await raise_issue_module.manage_issue_draft(
        Params(),
        operation="start",
        customer_name="Rohan Sharma",
        cust_id="C001122",
        mobile="9876543210",
        device_id="MSW12345678",
        description="Card transactions fail with a Transaction Failed error.",
    )

    first_result, first_properties = results[-1]
    assert workflow.status == "collecting_fields"
    assert first_result["missing_fields"] == ["email"]
    assert first_result["provenance"]["cust_id"] == "rag"
    assert first_result["source_refs"][0]["id"] == 12
    assert "email address" in first_result["message"]
    assert first_properties.run_llm is False
    assert inserts == []

    context.add_message({"role": "user", "content": "Use rohan@example.com"})
    Params.arguments = {"operation": "update", "email": "rohan@example.com"}
    await raise_issue_module.manage_issue_draft(
        Params(),
        operation="update",
        email="rohan@example.com",
    )

    second_result, second_properties = results[-1]
    assert workflow.status == "awaiting_confirmation"
    assert second_result["missing_fields"] == []
    assert second_result["provenance"]["email"] == "user"
    assert "Would you like me to submit it?" in second_result["message"]
    assert second_properties.run_llm is False
    assert inserts == []

    context.add_message({"role": "user", "content": "Yes, submit it"})
    Params.arguments = {"operation": "confirm"}
    await asyncio.gather(
        raise_issue_module.manage_issue_draft(Params(), operation="confirm"),
        raise_issue_module.manage_issue_draft(Params(), operation="confirm"),
    )

    assert workflow.status == "submitted"
    assert workflow.issue_id == 73
    assert len(inserts) == 1
    assert {result["status"] for result, _properties in results[-2:]} == {
        "success",
        "already_submitted",
    }
    assert all(properties.run_llm is False for _result, properties in results[-2:])

    await raise_issue_module.manage_issue_draft(Params(), operation="confirm")
    assert len(inserts) == 1
    assert results[-1][0]["status"] == "already_submitted"
    assert any(
        isinstance(frame, TTSSpeakFrame) and frame.append_to_context
        for frame in frames
    )


@pytest.mark.anyio
async def test_confirmation_cannot_skip_collection():
    delivered = []

    async def capture(result, *, properties=None):
        delivered.append((result, properties))

    params = SimpleNamespace(
        function_name="manage_issue_draft",
        tool_call_id="premature-confirm",
        arguments={"operation": "confirm"},
        pipeline_worker=None,
        context=LLMContext(messages=[{"role": "user", "content": "Do it"}]),
        app_resources={"issue_workflow": raise_issue_module.IssueWorkflowState()},
        result_callback=capture,
    )

    await raise_issue_module.manage_issue_draft(params, operation="confirm")

    assert delivered[0][0]["status"] == "invalid_transition"
    assert delivered[0][1].run_llm is False


@pytest.mark.anyio
@pytest.mark.parametrize("operation", ["status", "defer"])
async def test_active_workflow_status_is_authoritative_and_never_submits(operation):
    delivered = []
    workflow = raise_issue_module.IssueWorkflowState(
        status="awaiting_confirmation",
        cust_id="C001122",
        email="rohan@example.com",
        mobile="9876543210",
        device_id="MSW12345678",
        description="Transactions fail.",
    )

    async def capture(result, *, properties=None):
        delivered.append((result, properties))

    params = SimpleNamespace(
        function_name="manage_issue_draft",
        tool_call_id=f"workflow-{operation}",
        arguments={"operation": operation},
        pipeline_worker=None,
        context=LLMContext(messages=[{"role": "user", "content": "Are you done?"}]),
        app_resources={"issue_workflow": workflow},
        result_callback=capture,
    )

    await raise_issue_module.manage_issue_draft(params, operation=operation)

    result, properties = delivered[0]
    assert result["status"] == "awaiting_confirmation"
    assert result["issue_id"] is None
    assert "not been submitted" in result["message"]
    assert workflow.status == "awaiting_confirmation"
    assert properties.run_llm is False


@pytest.mark.anyio
async def test_stale_rag_evidence_never_autofills_a_new_user_supplied_draft():
    delivered = []
    workflow = raise_issue_module.IssueWorkflowState()
    retrieval = SimpleNamespace(latest_rag_evidence={
        "result": {
            "chunks": [{
                "id": 9,
                "file_id": 2,
                "content": "Old complaint C001122 9876543210 MSW12345678",
            }]
        }
    })

    async def capture(result, *, properties=None):
        delivered.append((result, properties))

    params = SimpleNamespace(
        function_name="manage_issue_draft",
        tool_call_id="new-draft",
        arguments={"operation": "start"},
        pipeline_worker=None,
        context=LLMContext(messages=[{
            "role": "user",
            "content": "I need help with a different device that will not start",
        }]),
        app_resources={
            "issue_workflow": workflow,
            "context_retrieval": retrieval,
        },
        result_callback=capture,
    )

    await raise_issue_module.manage_issue_draft(
        params,
        operation="start",
        description="different device that will not start",
    )

    result = delivered[0][0]
    assert result["draft"]["cust_id"] is None
    assert result["draft"]["mobile"] is None
    assert result["draft"]["device_id"] is None
    assert result["source_refs"] == []
    assert result["provenance"]["description"] == "user"


@pytest.mark.anyio
async def test_immediate_grounded_followup_hydrates_unique_issue_fields():
    delivered = []
    context = LLMContext(messages=[
        {"role": "user", "content": "Can you raise the issue?"},
    ])
    retrieval = ContextRetrievalProcessor(1, 1, context)
    retrieval._record_grounded_evidence(
        "Tell me about Rohan Sharma from my file",
        {
            "rag_call_id": "rag-rohan",
            "result": {
                "chunks": [{
                    "id": 12,
                    "file_id": 4,
                    "source_type": "pdf",
                    "filename": "issue.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "content": (
                        "Rohan Sharma customer ID C001122 mobile 9876543210 "
                        "device MSW12345678. Card transactions consistently "
                        "fail with a Transaction Failed error."
                    ),
                }],
            },
        },
    )
    retrieval.start_user_turn()

    async def capture(result, *, properties=None):
        delivered.append((result, properties))

    params = SimpleNamespace(
        function_name="manage_issue_draft",
        tool_call_id="grounded-followup",
        arguments={"operation": "start", "customer_name": "Rohan Sharma"},
        pipeline_worker=None,
        context=context,
        app_resources={
            "issue_workflow": raise_issue_module.IssueWorkflowState(),
            "context_retrieval": retrieval,
        },
        result_callback=capture,
    )

    await raise_issue_module.manage_issue_draft(
        params,
        operation="start",
        customer_name="Rohan Sharma",
    )

    result = delivered[0][0]
    assert result["missing_fields"] == ["email"]
    assert result["draft"]["cust_id"] == "C001122"
    assert result["draft"]["mobile"] == "9876543210"
    assert result["draft"]["device_id"] == "MSW12345678"
    assert "Transaction Failed" in result["draft"]["description"]
    assert result["evidence_id"] == "rag-rohan"
    assert result["hydrated_fields"] == [
        "cust_id",
        "description",
        "device_id",
        "mobile",
    ]
    assert result["source_refs"][0]["id"] == 12


@pytest.mark.anyio
async def test_multiple_grounded_identifiers_are_not_guessed():
    delivered = []
    context = LLMContext(messages=[{"role": "user", "content": "Open a draft"}])
    retrieval = ContextRetrievalProcessor(1, 1, context)
    retrieval._record_grounded_evidence(
        "Compare the two records in my file",
        {
            "rag_call_id": "rag-ambiguous",
            "result": {"chunks": [
                {"id": 1, "content": "C001122 9876543210 MSW12345678 first failure"},
                {"id": 2, "content": "C009999 9123456789 MSW87654321 second failure"},
            ]},
        },
    )
    retrieval.start_user_turn()

    async def capture(result, *, properties=None):
        delivered.append(result)

    params = SimpleNamespace(
        function_name="manage_issue_draft",
        tool_call_id="ambiguous-followup",
        arguments={"operation": "start"},
        pipeline_worker=None,
        context=context,
        app_resources={
            "issue_workflow": raise_issue_module.IssueWorkflowState(),
            "context_retrieval": retrieval,
        },
        result_callback=capture,
    )

    await raise_issue_module.manage_issue_draft(params, operation="start")

    assert delivered[0]["draft"]["cust_id"] is None
    assert delivered[0]["draft"]["mobile"] is None
    assert delivered[0]["draft"]["device_id"] is None
    assert delivered[0]["hydrated_fields"] == []


@pytest.mark.anyio
async def test_expired_grounded_anchor_cannot_hydrate_later_draft():
    delivered = []
    context = LLMContext(messages=[{"role": "user", "content": "Start a draft"}])
    retrieval = ContextRetrievalProcessor(1, 1, context)
    retrieval._record_grounded_evidence(
        "old question",
        {"rag_call_id": "rag-old", "result": {"chunks": [{
            "id": 1,
            "content": "C001122 9876543210 MSW12345678 old complaint",
        }]}},
    )
    retrieval._observe_completed_user_message(
        {"role": "user", "content": "First unrelated completed turn"}
    )
    retrieval._observe_completed_user_message(
        {"role": "user", "content": "Second unrelated completed turn"}
    )

    async def capture(result, *, properties=None):
        delivered.append(result)

    params = SimpleNamespace(
        function_name="manage_issue_draft",
        tool_call_id="expired-anchor",
        arguments={"operation": "start"},
        pipeline_worker=None,
        context=context,
        app_resources={
            "issue_workflow": raise_issue_module.IssueWorkflowState(),
            "context_retrieval": retrieval,
        },
        result_callback=capture,
    )

    await raise_issue_module.manage_issue_draft(params, operation="start")

    assert delivered[0]["draft"]["cust_id"] is None
    assert delivered[0]["evidence_id"] is None
    assert delivered[0]["hydrated_fields"] == []
