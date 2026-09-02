import asyncio
import importlib
from types import SimpleNamespace

import pytest
from pipecat.frames.frames import OutputTransportMessageFrame, TTSSpeakFrame
from pipecat.processors.aggregators.llm_context import LLMContext

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


def test_workflow_keeps_current_turn_and_bounded_protected_history():
    workflow = raise_issue_module.IssueWorkflowState(
        status="collecting_fields",
        cust_id="C001122",
        mobile="9876543210",
        device_id="MSW12345678",
        description="Card transactions fail for debit and credit cards.",
    )

    workflow.observe_user_turn("The email address is Rohan at the red")
    assert workflow.current_user_text == "The email address is Rohan at the red"

    workflow.observe_user_turn("Gmail.com")
    assert workflow.current_user_text == "Gmail.com"
    assert workflow.recent_user_turns == [
        (None, "The email address is Rohan at the red"),
        (None, "Gmail.com"),
    ]
    assert workflow.candidate_for_missing_field() is None


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
    assert result["fields"]["email"]["present"] is False
    assert "email" not in result["provenance"]
    assert result["missing_fields"] == ["email"]
    assert properties.run_llm is False

@pytest.mark.anyio
async def test_spoken_email_is_aligned_to_the_authoritative_user_turn():
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
    assert result["fields"]["email"] == {
        "present": True,
        "state": "unverified",
        "masked": "r***@gmail.com",
    }
    assert result["provenance"]["email"]["source_type"] == "user_spoken_email"
    assert result["invalid_fields"] == []
    assert properties.run_llm is False


@pytest.mark.anyio
async def test_semantic_issue_draft_collects_fields_then_requires_confirmation(
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

    context = LLMContext(messages=[{
        "role": "user",
        "content": (
            "Customer ID is C zero zero one one two two, mobile is nine eight "
            "seven six five four three two one zero, and device ID is M S W "
            "one two three four five six seven eight. Card transactions fail "
            "with a Transaction Failed error."
        ),
    }])

    class Params:
        function_name = "manage_issue_draft"
        tool_call_id = "workflow-call"
        arguments = {"operation": "start"}
        pipeline_worker = Worker()
        app_resources = {"issue_workflow": workflow}

        @staticmethod
        async def result_callback(result, *, properties=None):
            results.append((result, properties))

    Params.context = context
    monkeypatch.setattr(raise_issue_module, "VoiceSessionLocal", FakeSession)

    await raise_issue_module.manage_issue_draft(
        Params(),
        operation="start",
        cust_id="C001122",
        mobile="9876543210",
        device_id="MSW12345678",
        description="Card transactions fail with a Transaction Failed error.",
    )

    first_result, first_properties = results[-1]
    assert workflow.status == "collecting_fields"
    assert first_result["missing_fields"] == ["email"]
    assert first_result["provenance"]["cust_id"]["source_type"] == "user_spoken_digits"
    assert first_result["contract_version"] == "demo-v1-unverified"
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
    assert second_result["provenance"]["email"]["source_type"] == "user_spoken_email"
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
async def test_cancel_closes_draft_without_submission():
    delivered = []
    workflow = raise_issue_module.IssueWorkflowState(
        status="collecting_fields",
        cust_id="C001122",
    )

    async def capture(result, *, properties=None):
        delivered.append((result, properties))

    params = SimpleNamespace(
        function_name="manage_issue_draft",
        tool_call_id="cancel-draft",
        arguments={"operation": "cancel"},
        pipeline_worker=None,
        context=LLMContext(messages=[{"role": "user", "content": "Cancel it"}]),
        app_resources={"issue_workflow": workflow},
        result_callback=capture,
    )

    await raise_issue_module.manage_issue_draft(params, operation="cancel")

    assert workflow.status == "cancelled"
    assert workflow.issue_id is None
    assert delivered[0][0]["status"] == "cancelled"
    assert delivered[0][1].run_llm is False


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [(TimeoutError(), "timeout"), (RuntimeError("database unavailable"), "error")],
)
async def test_submission_failure_restores_confirmation_state(
    monkeypatch,
    failure,
    expected_status,
):
    delivered = []
    workflow = raise_issue_module.IssueWorkflowState(
        status="awaiting_confirmation",
        cust_id="C001122",
        email="rohan@example.com",
        mobile="9876543210",
        device_id="MSW12345678",
        description="Transactions keep failing.",
    )

    async def fail_submission(_state):
        raise failure

    async def capture(result, *, properties=None):
        delivered.append((result, properties))

    monkeypatch.setattr(raise_issue_module, "_create_issue_record", fail_submission)
    params = SimpleNamespace(
        function_name="manage_issue_draft",
        tool_call_id="failed-submit",
        arguments={"operation": "confirm"},
        pipeline_worker=None,
        context=LLMContext(messages=[{"role": "user", "content": "Submit it"}]),
        app_resources={"issue_workflow": workflow},
        result_callback=capture,
    )

    await raise_issue_module.manage_issue_draft(params, operation="confirm")

    assert workflow.status == "awaiting_confirmation"
    assert workflow.issue_id is None
    assert delivered[0][0]["status"] == expected_status
    assert delivered[0][1].run_llm is False


@pytest.mark.anyio
async def test_invalid_structured_fields_are_rejected_by_backend_schema():
    delivered = []
    workflow = raise_issue_module.IssueWorkflowState()

    async def capture(result, *, properties=None):
        delivered.append((result, properties))

    params = SimpleNamespace(
        function_name="manage_issue_draft",
        tool_call_id="invalid-fields",
        arguments={"operation": "start", "cust_id": "merchant one"},
        pipeline_worker=None,
        context=LLMContext(messages=[{"role": "user", "content": "My ID is merchant one"}]),
        app_resources={"issue_workflow": workflow},
        result_callback=capture,
    )

    await raise_issue_module.manage_issue_draft(
        params,
        operation="start",
        cust_id="merchant one",
    )

    result, properties = delivered[0]
    assert result["status"] == "collecting_fields"
    assert result["fields"]["cust_id"]["present"] is False
    assert result["invalid_fields"] == ["cust_id"]
    assert result["validation_outcomes"] == {"cust_id": "incomplete_speech"}
    assert properties.run_llm is False


@pytest.mark.anyio
async def test_ten_digit_mobile_with_invalid_prefix_gets_actionable_format_result():
    delivered = []
    workflow = raise_issue_module.IssueWorkflowState(status="collecting_fields")
    workflow.observe_user_turn(
        "My mobile is one two three four five six seven eight nine zero",
        turn_id=12,
    )

    async def capture(result, *, properties=None):
        delivered.append((result, properties))

    params = SimpleNamespace(
        function_name="manage_issue_draft",
        tool_call_id="invalid-prefix",
        arguments={"operation": "update", "mobile": "1234567890"},
        pipeline_worker=None,
        context=LLMContext(messages=[]),
        app_resources={"issue_workflow": workflow},
        result_callback=capture,
    )
    await raise_issue_module.manage_issue_draft(
        params,
        operation="update",
        mobile="1234567890",
    )

    result = delivered[0][0]
    assert result["validation_outcomes"] == {"mobile": "invalid_format"}
    assert result["fields"]["mobile"]["present"] is False
    assert "exactly 10 digits" in result["message"]
    assert "beginning with 6, 7, 8, or 9" in result["message"]


@pytest.mark.anyio
async def test_spoken_double_zero_overrides_lossy_llm_numeric_candidate():
    delivered = []
    workflow = raise_issue_module.IssueWorkflowState(status="collecting_fields")
    workflow.observe_user_turn(
        "My mobile is nine double zero four eight zero one eight zero six",
        turn_id=13,
    )

    async def capture(result, *, properties=None):
        delivered.append((result, properties))

    params = SimpleNamespace(
        function_name="manage_issue_draft",
        tool_call_id="double-zero",
        arguments={"operation": "update", "mobile": "990040801860"},
        pipeline_worker=None,
        context=LLMContext(messages=[]),
        app_resources={"issue_workflow": workflow},
        result_callback=capture,
    )
    await raise_issue_module.manage_issue_draft(
        params,
        operation="update",
        mobile="990040801860",
    )

    assert workflow.mobile == "9004801806"
    assert delivered[0][0]["fields"]["mobile"]["masked"] == "***1806"
    assert delivered[0][0]["provenance"]["mobile"] == {
        "source_type": "user_spoken_digits",
        "turn_id": 13,
        "source_hash": workflow.provenance["mobile"]["source_hash"],
        "normalization_version": "spoken-digits-en-hi-v1",
        "verification": "not_configured",
    }


@pytest.mark.anyio
async def test_schema_valid_identity_without_user_evidence_cannot_mutate_draft():
    delivered = []
    workflow = raise_issue_module.IssueWorkflowState(status="collecting_fields")
    workflow.observe_user_turn("Please continue with the complaint", turn_id=14)

    async def capture(result, *, properties=None):
        delivered.append((result, properties))

    params = SimpleNamespace(
        function_name="manage_issue_draft",
        tool_call_id="fabricated-id",
        arguments={"operation": "update", "cust_id": "C123456"},
        pipeline_worker=None,
        context=LLMContext(messages=[]),
        app_resources={"issue_workflow": workflow},
        result_callback=capture,
    )
    await raise_issue_module.manage_issue_draft(
        params,
        operation="update",
        cust_id="C123456",
    )

    assert workflow.cust_id is None
    assert "cust_id" not in workflow.provenance
    assert delivered[0][0]["validation_outcomes"] == {
        "cust_id": "invalid_format"
    }


def test_customer_name_is_not_part_of_demo_tool_or_workflow_contract():
    schema = raise_issue_module.openai_manage_issue_tool_schema()

    assert "customer_name" not in schema["function"]["parameters"]["properties"]


@pytest.mark.anyio
async def test_retry_reuses_exact_protected_authoritative_turn():
    delivered = []
    workflow = raise_issue_module.IssueWorkflowState(
        status="collecting_fields",
        cust_id="C001122",
        email="rohan@example.com",
        device_id="MSW12345678",
        description="The device does not announce payments.",
    )
    workflow.observe_user_turn(
        "My mobile is nine double zero four eight zero one eight zero six",
        turn_id=21,
    )
    workflow.observe_user_turn("Please try that again", turn_id=22)

    async def capture(result, *, properties=None):
        delivered.append((result, properties))

    params = SimpleNamespace(
        function_name="manage_issue_draft",
        tool_call_id="retry-protected-turn",
        arguments={"operation": "retry"},
        pipeline_worker=None,
        context=LLMContext(messages=[]),
        app_resources={"issue_workflow": workflow},
        result_callback=capture,
    )

    await raise_issue_module.manage_issue_draft(params, operation="retry")

    assert workflow.mobile == "9004801806"
    assert workflow.provenance["mobile"]["turn_id"] == 21
    assert delivered[0][0]["fields"]["mobile"]["masked"] == "***1806"
    assert delivered[0][0]["status"] == "awaiting_confirmation"
    assert not hasattr(raise_issue_module.IssueWorkflowState(), "customer_name")
