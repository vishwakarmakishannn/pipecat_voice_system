import re
import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper
from pipecat.frames.frames import OutputTransportMessageFrame, TTSSpeakFrame
from pipecat.services.llm_service import (
    FunctionCallParams,
    FunctionCallResultProperties,
)
from core.database import VoiceSessionLocal
from core.issue_contract import (
    ISSUE_CONTRACT_VERSION,
    ISSUE_FIELDS,
    ISSUE_REQUIRED_FIELDS,
    ValidationCode,
    safe_issue_field_states,
)
from core.models import Issue
from core.task_queue import task_queue
from core.tool_privacy import sanitize_tool_data
from core.tool_config import issue_tool_timeout_seconds
from services.calls import save_transcript_entry
from services.structured_digits import (
    NORMALIZATION_VERSION,
    DigitSequence,
    extract_digit_sequences,
    unique_sequence_for_length,
)


_SPOKEN_EMAIL_AT = re.compile(
    r"\b(?:at\s+the\s+rate|at\s+sign|at)\b",
    re.IGNORECASE,
)
_SPOKEN_EMAIL_DOT = re.compile(r"\b(?:dot|point)\b", re.IGNORECASE)
_EMAIL_CANDIDATE = re.compile(
    r"(?<![a-z0-9._%+-])"
    r"([a-z0-9][a-z0-9._%+-]*)\s*@\s*"
    r"([a-z0-9-]+(?:\.[a-z0-9-]+)+)",
    re.IGNORECASE,
)
_EMAIL_DOMAIN_AFTER_AT = re.compile(
    r"\b([a-z0-9-]+(?:\s*(?:\.|dot|point)\s*[a-z0-9-]+)+)\b",
    re.IGNORECASE,
)
_EMAIL_LOCAL_TOKEN = re.compile(r"[a-z0-9][a-z0-9._%+-]*", re.IGNORECASE)


def normalize_spoken_email(text: str) -> str | None:
    """Extract one schema-valid email from finalized text or voice dictation."""
    spoken = _SPOKEN_EMAIL_AT.sub("@", text or "")
    spoken = re.sub(
        r"\s*\b(?:dot|point)\b\s*",
        ".",
        spoken,
        flags=re.IGNORECASE,
    )
    matches = {
        f"{match.group(1)}@{re.sub(r'\s+', '', match.group(2))}".casefold()
        for match in _EMAIL_CANDIDATE.finditer(spoken)
    }
    if len(matches) != 1:
        # Streaming STT commonly inserts harmless words between a spoken "at"
        # marker and a later domain fragment (for example when the local part
        # and domain land in different finals). During explicit email dictation,
        # recover the field structurally instead of enumerating ASR mistakes.
        loose_matches: set[str] = set()
        raw = re.sub(
            r"\s*\b(?:dot|point)\b\s*",
            ".",
            text or "",
            flags=re.IGNORECASE,
        )
        if re.search(r"\bemail\b", raw, re.IGNORECASE):
            for at_match in _SPOKEN_EMAIL_AT.finditer(raw):
                local_tokens = _EMAIL_LOCAL_TOKEN.findall(raw[: at_match.start()])
                domain_match = _EMAIL_DOMAIN_AFTER_AT.search(raw, at_match.end())
                if not local_tokens or domain_match is None:
                    continue
                domain = _SPOKEN_EMAIL_DOT.sub(".", domain_match.group(1))
                domain = re.sub(r"\s+", "", domain)
                candidate = f"{local_tokens[-1]}@{domain}".casefold()
                if _valid_field("email", candidate):
                    loose_matches.add(candidate)
        return next(iter(loose_matches)) if len(loose_matches) == 1 else None
    candidate = next(iter(matches))
    return candidate if _valid_field("email", candidate) else None


@dataclass
class IssueWorkflowState:
    """Per-call complaint draft controlled by semantic tool calls, not text rules."""

    status: str = "idle"
    cust_id: str | None = None
    email: str | None = None
    mobile: str | None = None
    device_id: str | None = None
    description: str | None = None
    provenance: dict[str, dict[str, Any]] = field(default_factory=dict)
    validation_states: dict[str, ValidationCode] = field(
        default_factory=lambda: {
            name: "missing" for name in ISSUE_REQUIRED_FIELDS
        }
    )
    issue_id: int | None = None
    current_user_text: str = ""
    current_turn_id: int | None = None
    protected_corrections: dict[str, dict[str, Any]] = field(
        default_factory=dict,
        repr=False,
    )
    recent_user_turns: list[tuple[int | None, str]] = field(
        default_factory=list,
        repr=False,
    )
    confirmation_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def reset(self) -> None:
        self.status = "idle"
        self.cust_id = None
        self.email = None
        self.mobile = None
        self.device_id = None
        self.description = None
        self.provenance.clear()
        self.validation_states = {
            name: "missing" for name in ISSUE_REQUIRED_FIELDS
        }
        self.issue_id = None
        self.protected_corrections.clear()
        self.recent_user_turns = (
            [(self.current_turn_id, self.current_user_text)]
            if self.current_user_text
            else []
        )

    def draft(self) -> dict[str, Any]:
        return {
            "cust_id": self.cust_id,
            "email": self.email,
            "mobile": self.mobile,
            "device_id": self.device_id,
            "description": self.description,
        }

    def public_fields(self) -> dict[str, dict[str, str | bool | None]]:
        return safe_issue_field_states(self.draft(), self.validation_states)

    def missing_fields(self) -> list[str]:
        return [name for name in ISSUE_REQUIRED_FIELDS if not getattr(self, name)]

    def prompt_context(self) -> str | None:
        if self.status in {"idle", "cancelled", "submitted"}:
            return None
        present = [name for name in ISSUE_REQUIRED_FIELDS if getattr(self, name)]
        missing = self.missing_fields()
        return (
            "ISSUE_WORKFLOW_STATE: A complaint draft is active. "
            f"status={self.status}; present={','.join(present) or 'none'}; "
            f"missing={','.join(missing) or 'none'}. "
            "The backend is the only authority for workflow outcomes. Call "
            "manage_issue_draft exactly once: update for supplied fields, confirm "
            "or cancel for the user's decision, status for a progress question, "
            "retry when the caller asks to retry protected recent field dictation, "
            "or defer when the turn does not decide or change the draft. Never "
            "state that an issue was submitted unless the tool result contains "
            "status=success and an issue_id. An unrelated turn must use defer and "
            "leave the complaint draft unchanged."
        )

    def observe_user_turn(self, text: str, *, turn_id: int | None = None) -> None:
        """Bind evidence to the authoritative assembled turn seen by the router."""
        self.current_user_text = " ".join((text or "").split())
        self.current_turn_id = turn_id
        if self.current_user_text:
            entry = (turn_id, self.current_user_text)
            if not self.recent_user_turns or self.recent_user_turns[-1] != entry:
                self.recent_user_turns.append(entry)
                del self.recent_user_turns[:-4]

    def candidate_for_missing_field(self) -> None:
        # Field parsing is deliberately kept out of the semantic router.  The
        # tool aligns structured candidates with this authoritative turn.
        return None

    def sensitive_values(self) -> dict[str, str]:
        return {
            name: value
            for name in ("cust_id", "email", "mobile", "device_id")
            if (value := getattr(self, name))
        }


async def _publish_tool_event(
    params: FunctionCallParams,
    status: str,
    result: dict | None = None,
) -> None:
    """Publish issue-tool lifecycle independently of provider frame direction."""
    worker = getattr(params, "pipeline_worker", None)
    if worker is None:
        return
    payload = {
        "tool_call_id": getattr(params, "tool_call_id", None) or "manage-issue-draft",
        "function_name": (
            getattr(params, "function_name", None) or "manage_issue_draft"
        ),
        "arguments": sanitize_tool_data(
            "manage_issue_draft",
            dict(getattr(params, "arguments", {}) or {}),
        ),
        "status": status,
    }
    if result is not None:
        payload["result"] = sanitize_tool_data("manage_issue_draft", result)
    await worker.queue_frame(OutputTransportMessageFrame({
        "label": "rtvi-ai",
        "type": "server-message",
        "data": {"type": "tool_call", "payload": payload},
    }))
def _workflow_state(params: FunctionCallParams) -> IssueWorkflowState:
    resources = getattr(params, "app_resources", None)
    if not isinstance(resources, dict):
        resources = {}
        params.app_resources = resources
    state = resources.get("issue_workflow")
    if not isinstance(state, IssueWorkflowState):
        state = IssueWorkflowState()
        resources["issue_workflow"] = state
    return state


def _latest_user_text(
    params: FunctionCallParams,
    state: IssueWorkflowState | None = None,
) -> str:
    if state is not None and state.current_user_text:
        return state.current_user_text
    context = getattr(params, "context", None)
    for message in reversed(getattr(context, "messages", []) or []):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""




def _valid_field(name: str, value: str) -> bool:
    contract = ISSUE_FIELDS.get(name)
    return bool(contract and contract.validate(value)[0] == "unverified")


def _source_hash(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _numeric_evidence(
    name: str,
    latest_user_text: str,
) -> tuple[str | None, ValidationCode, DigitSequence | None]:
    contract = ISSUE_FIELDS[name]
    assert contract.digit_count is not None
    exact = unique_sequence_for_length(latest_user_text, contract.digit_count)
    if exact is not None:
        canonical = contract.canonicalize(exact.value)
        code, canonical = contract.validate(canonical)
        return (canonical if code == "unverified" else None), code, exact

    sequences = extract_digit_sequences(latest_user_text)
    if len(sequences) != 1:
        return None, "invalid_format", None
    sequence = sequences[0]
    canonical = contract.canonicalize(sequence.value)
    code, canonical = contract.validate(canonical)
    return None, code, sequence


def _aligned_candidate(
    name: str,
    raw_value: str,
    *,
    latest_user_text: str,
    turn_id: int | None,
) -> tuple[str | None, ValidationCode, dict[str, Any]]:
    contract = ISSUE_FIELDS[name]
    source_hash = _source_hash(latest_user_text)
    if contract.digit_count is not None:
        value, code, sequence = _numeric_evidence(name, latest_user_text)
        provenance = {
            "source_type": "user_spoken_digits",
            "turn_id": turn_id,
            "source_hash": source_hash,
            "source_span": (
                [sequence.start, sequence.end] if sequence is not None else None
            ),
            "normalization_version": NORMALIZATION_VERSION,
            "verification": "not_configured",
        }
        return value, code, provenance

    if contract.kind == "email":
        value = normalize_spoken_email(latest_user_text)
        code, canonical = contract.validate(value)
        return (
            canonical if code == "unverified" else None,
            code,
            {
                "source_type": "user_spoken_email",
                "turn_id": turn_id,
                "source_hash": source_hash,
                "normalization_version": "spoken-email-v1",
                "verification": "not_configured",
            },
        )

    code, canonical = contract.validate(raw_value)
    # A problem description may be summarized by the LLM, but its source turn
    # remains attributable and the confirmation never invents a person's name.
    if code == "unverified" and latest_user_text:
        return (
            canonical,
            code,
            {
                "source_type": "llm_summary_of_user_report",
                "turn_id": turn_id,
                "source_hash": source_hash,
                "normalization_version": "description-summary-v1",
                "verification": "not_applicable",
            },
        )
    return None, code, {
        "source_type": "unattributed",
        "turn_id": turn_id,
        "source_hash": source_hash,
        "normalization_version": "none",
        "verification": "not_configured",
    }


def _merge_draft_fields(
    state: IssueWorkflowState,
    candidates: dict[str, str | None],
    *,
    latest_user_text: str,
    turn_id: int | None = None,
) -> dict[str, ValidationCode]:
    rejected: dict[str, ValidationCode] = {}
    for name, raw_value in candidates.items():
        if raw_value is None:
            continue
        contract = ISSUE_FIELDS.get(name)
        if contract is None:
            continue
        raw_text = str(raw_value).strip()
        argument_canonical = contract.canonicalize(raw_text)
        if (
            getattr(state, name) == argument_canonical
            and state.validation_states.get(name) == "unverified"
        ):
            # Models commonly repeat already accepted values.  Repetition does
            # not change their original evidence or verification state.
            continue
        value, code, provenance = _aligned_candidate(
            name,
            raw_text,
            latest_user_text=latest_user_text,
            turn_id=turn_id,
        )
        state.validation_states[name] = code
        if value is None or code != "unverified":
            setattr(state, name, None)
            state.provenance.pop(name, None)
            state.protected_corrections[name] = {
                "state": code,
                "candidate_hash": _source_hash(argument_canonical),
                "turn_id": turn_id,
            }
            rejected[name] = code
            continue
        setattr(state, name, value)
        state.provenance[name] = provenance
        state.protected_corrections.pop(name, None)
    return rejected


def _retry_recent_evidence(
    state: IssueWorkflowState,
) -> dict[str, ValidationCode]:
    """Re-run deterministic extraction over a short protected in-memory window."""
    rejected: dict[str, ValidationCode] = {}
    for name in ("cust_id", "email", "mobile", "device_id"):
        if getattr(state, name):
            continue
        matches: dict[str, dict[str, Any]] = {}
        for turn_id, user_text in reversed(state.recent_user_turns):
            value, code, provenance = _aligned_candidate(
                name,
                "",
                latest_user_text=user_text,
                turn_id=turn_id,
            )
            if value is not None and code == "unverified":
                matches[value] = provenance
        if len(matches) == 1:
            value, provenance = next(iter(matches.items()))
            setattr(state, name, value)
            state.validation_states[name] = "unverified"
            state.provenance[name] = provenance
            state.protected_corrections.pop(name, None)
        elif len(matches) > 1:
            state.validation_states[name] = "invalid_format"
            rejected[name] = "invalid_format"
    return rejected


def _draft_result(
    state: IssueWorkflowState,
    message: str,
    rejected: dict[str, ValidationCode],
) -> dict[str, Any]:
    return {
        "status": state.status,
        "message": message,
        "contract_version": ISSUE_CONTRACT_VERSION,
        "missing_fields": state.missing_fields(),
        "invalid_fields": list(rejected),
        "validation_outcomes": dict(rejected),
        "fields": state.public_fields(),
        "provenance": {
            name: {
                "source_type": value.get("source_type"),
                "turn_id": value.get("turn_id"),
                "source_hash": value.get("source_hash"),
                "normalization_version": value.get("normalization_version"),
                "verification": value.get("verification"),
            }
            for name, value in state.provenance.items()
        },
    }


def _field_list(names: list[str]) -> str:
    labels = [ISSUE_FIELDS[name].label if name in ISSUE_FIELDS else name for name in names]
    if len(labels) <= 1:
        return labels[0] if labels else ""
    return ", ".join(labels[:-1]) + f" and {labels[-1]}"


def _workflow_message(
    state: IssueWorkflowState,
    *,
    rejected: dict[str, ValidationCode] | None = None,
) -> str:
    rejected = rejected or {}
    missing = state.missing_fields()
    if rejected:
        corrections = [
            ISSUE_FIELDS[name].correction(code)
            for name, code in rejected.items()
        ]
        other_missing = [name for name in missing if name not in rejected]
        if other_missing:
            corrections.append(
                f"I also still need {_field_list(other_missing)} before I can prepare the issue."
            )
        return " ".join(corrections)
    if missing:
        return f"I still need {_field_list(missing)} before I can prepare the issue."
    masked = {
        name: ISSUE_FIELDS[name].masked(getattr(state, name))
        for name in ("cust_id", "email", "mobile", "device_id")
    }
    return (
        "I have the complaint details ready: customer ID ending "
        f"{masked['cust_id'][-4:]}, email {masked['email']}, mobile ending "
        f"{masked['mobile'][-4:]}, device ID ending {masked['device_id'][-4:]}, "
        "and the issue description you provided. "
        "Would you like me to submit it?"
    )


async def _finish_workflow_call(
    params: FunctionCallParams,
    result: dict[str, Any],
    spoken_text: str,
) -> None:
    """Finish locally so collection turns do not require a second LLM request."""
    await _publish_tool_event(params, "completed", result)
    resources = getattr(params, "app_resources", None)
    call_id = resources.get("call_id") if isinstance(resources, dict) else None
    latency_state = resources.get("latency_state") if isinstance(resources, dict) else None
    if call_id:
        task_queue.enqueue(
            save_transcript_entry,
            call_id,
            "Aura",
            spoken_text,
            source="issue_workflow",
            turn_id=getattr(latency_state, "turn_id", None),
            audio_offset_ms=(
                latency_state.audio_offset_getter()
                if getattr(latency_state, "audio_offset_getter", None)
                else None
            ),
            key=str(call_id),
        )
    worker = getattr(params, "pipeline_worker", None)
    if worker is not None:
        await worker.queue_frames([
            OutputTransportMessageFrame({
                "label": "rtvi-ai",
                "type": "server-message",
                "data": {
                    "type": "assistant_transcript",
                    "payload": {
                        "id": f"issue-workflow-{getattr(params, 'tool_call_id', 'call')}",
                        "text": spoken_text,
                        "source": "issue_workflow",
                    },
                },
            }),
            TTSSpeakFrame(spoken_text, append_to_context=True),
        ])
    await params.result_callback(
        result,
        properties=FunctionCallResultProperties(run_llm=False),
    )


async def _create_issue_record(state: IssueWorkflowState) -> int:
    """Persist the demo issue; replace this boundary with the production API."""
    async with asyncio.timeout(issue_tool_timeout_seconds()):
        async with VoiceSessionLocal() as session:
            new_issue = Issue(
                cust_id=state.cust_id,
                email=state.email,
                mobile=state.mobile,
                device_id=state.device_id,
                description=state.description,
            )
            session.add(new_issue)
            await session.flush()
            issue_id = new_issue.id
            await session.commit()
            return issue_id


async def manage_issue_draft(
    params: FunctionCallParams,
    operation: Literal[
        "start", "update", "confirm", "cancel", "status", "retry", "defer"
    ],
    cust_id: str | None = None,
    email: str | None = None,
    mobile: str | None = None,
    device_id: str | None = None,
    description: str | None = None,
):
    """Manage a complaint draft using the meaning of the full conversation.

    Use this tool only when the user semantically wants to begin, update,
    confirm, or cancel complaint processing. It is non-destructive until a
    separate confirmation turn calls ``operation=confirm`` while the backend
    state is ``awaiting_confirmation``.

    Args:
        operation: One of start, update, confirm, cancel, status, retry, or defer. Use start for the
            initial action request, update when the user supplies or corrects
            details, confirm only after the assistant asked for confirmation,
            cancel when the user declines or abandons the draft, status for a
            progress question, retry when the caller asks to retry recent field
            dictation after a transient failure, and defer when the turn does not
            change the draft.
        cust_id: Demo customer ID candidate. The temporary demo-v1-unverified
            contract expects C followed by exactly 6 digits; this is format
            validation, not customer verification.
        email: Customer email address.
        mobile: Demo mobile candidate. The temporary demo contract expects ten
            digits beginning with 6, 7, 8, or 9.
        device_id: Demo device ID candidate. The temporary demo contract expects
            MSW followed by exactly 8 digits.
        description: Concise, grounded description of the reported problem.
    """
    state = _workflow_state(params)
    operation = (operation or "").strip().casefold()
    await _publish_tool_event(params, "in_progress")

    if operation in {"status", "defer"}:
        if state.status == "submitted" and state.issue_id is not None:
            message = f"Issue #{state.issue_id} has been successfully raised."
        elif state.status == "awaiting_confirmation":
            message = (
                "The complaint has not been submitted yet. Its details are ready "
                "and it is awaiting your confirmation."
            )
        elif state.status == "collecting_fields":
            message = (
                f"The complaint has not been submitted yet. I still need "
                f"{_field_list(state.missing_fields())}."
            )
        elif state.status == "submitting":
            message = "The complaint submission is currently in progress."
        elif state.status == "cancelled":
            message = "The complaint draft was cancelled and was not submitted."
        else:
            message = "There is no active complaint draft."
        await _finish_workflow_call(
            params,
            {
                "status": state.status,
                "issue_id": state.issue_id,
                "message": message,
            },
            message,
        )
        return

    if operation == "cancel":
        if state.status in {"idle", "cancelled", "submitted"}:
            message = "There is no active complaint draft to cancel."
        else:
            state.status = "cancelled"
            message = "Okay, I cancelled the complaint draft."
        await _finish_workflow_call(
            params,
            {"status": state.status, "message": message},
            message,
        )
        return

    if operation == "confirm":
        async with state.confirmation_lock:
            if state.status == "submitted" and state.issue_id is not None:
                message = f"Issue #{state.issue_id} has already been submitted."
                await _finish_workflow_call(
                    params,
                    {
                        "status": "already_submitted",
                        "issue_id": state.issue_id,
                        "message": message,
                    },
                    message,
                )
                return
            if state.status != "awaiting_confirmation":
                message = "The complaint is not ready for confirmation yet."
                await _finish_workflow_call(
                    params,
                    {"status": "invalid_transition", "message": message},
                    message,
                )
                return
            state.status = "submitting"
            try:
                state.issue_id = await _create_issue_record(state)
            except TimeoutError:
                state.status = "awaiting_confirmation"
                message = (
                    "Issue creation timed out and was not confirmed. "
                    "Please try again later."
                )
                await _finish_workflow_call(
                    params,
                    {"status": "timeout", "message": message},
                    message,
                )
                return
            except asyncio.CancelledError:
                state.status = "awaiting_confirmation"
                await _publish_tool_event(params, "cancelled")
                raise
            except Exception:
                state.status = "awaiting_confirmation"
                message = (
                    "Issue creation failed and was not confirmed. "
                    "Please try again later."
                )
                await _finish_workflow_call(
                    params,
                    {"status": "error", "message": message},
                    message,
                )
                return
            state.status = "submitted"
            message = f"Issue #{state.issue_id} has been successfully raised."
            await _finish_workflow_call(
                params,
                {
                    "status": "success",
                    "issue_id": state.issue_id,
                    "message": message,
                },
                message,
            )
        return

    if operation == "retry":
        if state.status not in {"collecting_fields", "awaiting_confirmation"}:
            message = "There is no active complaint draft to retry."
            await _finish_workflow_call(
                params,
                {"status": "invalid_transition", "message": message},
                message,
            )
            return
        rejected = _retry_recent_evidence(state)
        state.status = (
            "collecting_fields"
            if state.missing_fields() or rejected
            else "awaiting_confirmation"
        )
        message = _workflow_message(state, rejected=rejected)
        await _finish_workflow_call(
            params,
            _draft_result(state, message, rejected),
            message,
        )
        return

    if operation not in {"start", "update"}:
        message = "I couldn't determine the requested complaint action."
        await _finish_workflow_call(
            params,
            {"status": "invalid_operation", "message": message},
            message,
        )
        return

    if operation == "start" and state.status in {"idle", "cancelled", "submitted"}:
        state.reset()
        state.status = "collecting_fields"
    elif operation == "update" and state.status not in {
        "collecting_fields",
        "awaiting_confirmation",
    }:
        message = "There is no active complaint draft to update."
        await _finish_workflow_call(
            params,
            {"status": "invalid_transition", "message": message},
            message,
        )
        return

    rejected = _merge_draft_fields(
        state,
        {
            "cust_id": cust_id,
            "email": email,
            "mobile": mobile,
            "device_id": device_id,
            "description": description,
        },
        latest_user_text=_latest_user_text(params, state),
        turn_id=state.current_turn_id,
    )
    state.status = (
        "collecting_fields"
        if state.missing_fields() or rejected
        else "awaiting_confirmation"
    )
    message = _workflow_message(state, rejected=rejected)
    result = _draft_result(state, message, rejected)
    await _finish_workflow_call(params, result, message)


def openai_manage_issue_tool_schema() -> dict:
    """Return the provider schema used for stable local-LLM warmup."""
    schema = DirectFunctionWrapper(manage_issue_draft).to_function_schema()
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": {
                "type": "object",
                "properties": schema.properties,
                "required": schema.required,
            },
        },
    }
