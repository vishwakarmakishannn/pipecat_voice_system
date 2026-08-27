import re
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper
from pipecat.frames.frames import OutputTransportMessageFrame, TTSSpeakFrame
from pipecat.processors.aggregators.llm_context import NOT_GIVEN
from pipecat.services.llm_service import (
    FunctionCallParams,
    FunctionCallResultProperties,
)
from core.database import VoiceSessionLocal
from core.models import Issue
from core.task_queue import task_queue
from core.tool_config import issue_tool_timeout_seconds, tool_filler_enabled
from services.calls import save_transcript_entry


ISSUE_REQUIRED_FIELDS = ("cust_id", "email", "mobile", "device_id", "description")
ISSUE_FIELD_LABELS = {
    "cust_id": "customer ID",
    "email": "email address",
    "mobile": "mobile number",
    "device_id": "device ID",
    "description": "issue description",
}

_SPOKEN_EMAIL_AT = re.compile(
    r"\b(?:at\s+the\s+rate|at\s+sign|at)\b",
    re.IGNORECASE,
)
_SPOKEN_EMAIL_DOT = re.compile(r"\b(?:dot|point)\b", re.IGNORECASE)
_EMAIL_CANDIDATE = re.compile(
    r"(?<![a-z0-9._%+-])"
    r"([a-z0-9][a-z0-9._%+-]*)\s*@\s*"
    r"([a-z0-9-]+(?:\s*\.\s*[a-z0-9-]+)+)",
    re.IGNORECASE,
)
_EMAIL_DOMAIN_AFTER_AT = re.compile(
    r"\b([a-z0-9-]+(?:\s*(?:\.|dot|point)\s*[a-z0-9-]+)+)\b",
    re.IGNORECASE,
)
_EMAIL_LOCAL_TOKEN = re.compile(r"[a-z0-9][a-z0-9._%+-]*", re.IGNORECASE)
_FIELD_FRAGMENT_TTL_SECONDS = 8.0
_FIELD_FRAGMENT_MAX_TURNS = 3
_FIELD_FRAGMENT_MAX_CHARS = 320


def normalize_spoken_email(text: str) -> str | None:
    """Extract one schema-valid email from finalized text or voice dictation."""
    spoken = _SPOKEN_EMAIL_AT.sub("@", text or "")
    spoken = _SPOKEN_EMAIL_DOT.sub(".", spoken)
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
        raw = text or ""
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
    customer_name: str | None = None
    cust_id: str | None = None
    email: str | None = None
    mobile: str | None = None
    device_id: str | None = None
    description: str | None = None
    provenance: dict[str, str] = field(default_factory=dict)
    source_refs: list[dict[str, Any]] = field(default_factory=list)
    evidence_id: str | None = None
    issue_id: int | None = None
    current_user_text: str = ""
    current_field_candidates: dict[str, str] = field(default_factory=dict)
    field_fragment_buffer: list[tuple[float, str]] = field(
        default_factory=list,
        repr=False,
    )
    confirmation_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def reset(self) -> None:
        self.status = "idle"
        self.customer_name = None
        self.cust_id = None
        self.email = None
        self.mobile = None
        self.device_id = None
        self.description = None
        self.provenance.clear()
        self.source_refs.clear()
        self.evidence_id = None
        self.issue_id = None
        self.field_fragment_buffer.clear()

    def draft(self) -> dict[str, Any]:
        return {
            "customer_name": self.customer_name,
            "cust_id": self.cust_id,
            "email": self.email,
            "mobile": self.mobile,
            "device_id": self.device_id,
            "description": self.description,
        }

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
            "or defer when the turn does not decide or change the draft. Never "
            "state that an issue was submitted unless the tool result contains "
            "status=success and an issue_id. "
            "Tavily remains available for a separate public web-search request, and a "
            "web search must not change or close this draft."
        )

    def observe_user_turn(self, text: str) -> None:
        """Bind field candidates to the exact finalized turn seen by the router."""
        raw_text = " ".join((text or "").split())
        self.current_user_text = raw_text
        self.current_field_candidates.clear()
        if self.status != "collecting_fields" or "email" not in self.missing_fields():
            self.field_fragment_buffer.clear()
            return

        now = time.monotonic()
        self.field_fragment_buffer = [
            (seen_at, fragment)
            for seen_at, fragment in self.field_fragment_buffer
            if now - seen_at <= _FIELD_FRAGMENT_TTL_SECONDS
        ]
        if raw_text:
            self.field_fragment_buffer.append((now, raw_text))
        self.field_fragment_buffer = self.field_fragment_buffer[
            -_FIELD_FRAGMENT_MAX_TURNS:
        ]

        fragments = [fragment for _seen_at, fragment in self.field_fragment_buffer]
        candidates = [raw_text]
        candidates.extend(
            " ".join(fragments[-count:])[-_FIELD_FRAGMENT_MAX_CHARS:]
            for count in range(2, len(fragments) + 1)
        )
        for candidate_text in candidates:
            if email := normalize_spoken_email(candidate_text):
                self.current_user_text = candidate_text
                self.current_field_candidates["email"] = email
                return

    def candidate_for_missing_field(self) -> tuple[str, str] | None:
        for name in self.missing_fields():
            if value := self.current_field_candidates.get(name):
                return name, value
        return None

    def sensitive_values(self) -> dict[str, str]:
        return {
            name: value
            for name in ("cust_id", "email", "mobile", "device_id")
            if (value := getattr(self, name))
        }


async def _publish_tool_filler(params: FunctionCallParams) -> None:
    worker = getattr(params, "pipeline_worker", None)
    resources = getattr(params, "app_resources", None)
    state = resources.get("latency_state") if isinstance(resources, dict) else None
    if worker is None or not tool_filler_enabled():
        return
    if state is not None and state.tool_filler_spoken:
        return
    if state is not None:
        state.tool_filler_spoken = True
    tool_call_id = getattr(params, "tool_call_id", None) or "raise-issue"
    filler_text = "Let me check that."
    await worker.queue_frames([
        OutputTransportMessageFrame({
            "label": "rtvi-ai",
            "type": "server-message",
            "data": {
                "type": "assistant_transcript",
                "payload": {
                    "id": f"tool-filler-{tool_call_id}",
                    "text": filler_text,
                    "source": "tool_filler",
                },
            },
        }),
        TTSSpeakFrame(filler_text, append_to_context=False),
    ])


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
        "tool_call_id": getattr(params, "tool_call_id", None) or "raise-issue",
        "function_name": getattr(params, "function_name", None) or "raise_issue",
        "arguments": dict(getattr(params, "arguments", {}) or {}),
        "status": status,
    }
    if result is not None:
        payload["result"] = result
    await worker.queue_frame(OutputTransportMessageFrame({
        "label": "rtvi-ai",
        "type": "server-message",
        "data": {"type": "tool_call", "payload": payload},
    }))


async def _return_tool_result(params: FunctionCallParams, result: dict) -> None:
    await _publish_tool_event(params, "completed", result)
    context = getattr(params, "context", None)
    if context is not None:
        context.set_tools([])
        context.set_tool_choice(NOT_GIVEN)
    await params.result_callback(result)


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


def _latest_rag_payload(params: FunctionCallParams) -> dict[str, Any] | None:
    resources = getattr(params, "app_resources", None)
    retrieval = resources.get("context_retrieval") if isinstance(resources, dict) else None
    payload = getattr(retrieval, "latest_rag_evidence", None)
    return payload if isinstance(payload, dict) else None


def _grounded_anchor(params: FunctionCallParams, evidence_id: str | None = None):
    resources = getattr(params, "app_resources", None)
    retrieval = resources.get("context_retrieval") if isinstance(resources, dict) else None
    resolver = getattr(retrieval, "grounded_evidence", None)
    if not callable(resolver):
        return None
    return resolver(evidence_id, immediate_only=True)


def _available_rag_payload(params: FunctionCallParams, anchor):
    """Return fresh anchored evidence, retaining compatibility with old callers."""
    resources = getattr(params, "app_resources", None)
    retrieval = resources.get("context_retrieval") if isinstance(resources, dict) else None
    if callable(getattr(retrieval, "grounded_evidence", None)):
        return getattr(anchor, "payload", None) if anchor is not None else None
    return _latest_rag_payload(params)


def _rag_evidence(payload: dict[str, Any] | None) -> tuple[str, list[dict[str, Any]]]:
    chunks = (
        payload.get("result", {}).get("chunks", [])
        if isinstance(payload, dict)
        else []
    )
    usable = [chunk for chunk in chunks if isinstance(chunk, dict)]
    text = "\n".join(
        str(chunk.get("content") or "")
        for chunk in usable
        if chunk.get("content")
    )
    refs = [
        {
            key: chunk.get(key)
            for key in (
                "id",
                "file_id",
                "source_type",
                "filename",
                "title",
                "url",
                "page_start",
                "page_end",
                "heading_path",
            )
        }
        for chunk in usable
    ]
    return text, refs


_EVIDENCE_FIELD_PATTERNS = {
    "cust_id": re.compile(r"\bC\d{6}\b", re.IGNORECASE),
    "email": re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b"),
    "mobile": re.compile(r"\b[6-9]\d{9}\b"),
    "device_id": re.compile(r"\bMSW\d{8}\b", re.IGNORECASE),
}


def _unique_grounded_fields(
    payload: dict[str, Any],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Extract only unambiguous schema-valid values from supporting chunks."""
    chunks = payload.get("result", {}).get("chunks", [])
    values: dict[str, set[str]] = {name: set() for name in _EVIDENCE_FIELD_PATTERNS}
    chunk_values: list[tuple[dict[str, Any], dict[str, set[str]]]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        content = str(chunk.get("content") or "")
        found: dict[str, set[str]] = {}
        for name, pattern in _EVIDENCE_FIELD_PATTERNS.items():
            matches = {
                match.upper() if name in {"cust_id", "device_id"} else match
                for match in pattern.findall(content)
            }
            if matches:
                found[name] = matches
                values[name].update(matches)
        chunk_values.append((chunk, found))

    unique = {
        name: next(iter(found))
        for name, found in values.items()
        if len(found) == 1
    }
    supporting = [
        chunk
        for chunk, found in chunk_values
        if any(value in found.get(name, set()) for name, value in unique.items())
    ]

    # A single complaint-like record can safely supply its own description
    # when at least two independent structured identifiers co-occur. Multiple
    # equally plausible records remain unresolved and are never guessed.
    description_rows = [
        chunk
        for chunk, found in chunk_values
        if sum(
            value in found.get(name, set())
            for name, value in unique.items()
        ) >= 2
    ]
    if len(description_rows) == 1:
        description = re.sub(
            r"\s+", " ", str(description_rows[0].get("content") or "")
        ).strip()
        if _valid_field("description", description):
            unique["description"] = description[:1000]
            if description_rows[0] not in supporting:
                supporting.append(description_rows[0])
    return unique, supporting


def _source_refs(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: chunk.get(key)
            for key in (
                "id",
                "file_id",
                "source_type",
                "filename",
                "title",
                "url",
                "page_start",
                "page_end",
                "heading_path",
            )
        }
        for chunk in chunks
    ]


def _valid_field(name: str, value: str) -> bool:
    if name == "cust_id":
        return bool(re.fullmatch(r"C\d{6}", value))
    if name == "email":
        return bool(re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value))
    if name == "mobile":
        return bool(re.fullmatch(r"[6-9]\d{9}", value))
    if name == "device_id":
        return bool(re.fullmatch(r"MSW\d{8}", value))
    if name == "description":
        return len(value.strip()) >= 8
    if name == "customer_name":
        return bool(value.strip())
    return False


def _description_supported(value: str, evidence_text: str) -> bool:
    candidate = {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) > 2
    }
    evidence = set(re.findall(r"[a-z0-9]+", evidence_text.casefold()))
    if not candidate:
        return False
    overlap = len(candidate & evidence)
    return overlap >= min(4, max(2, len(candidate) // 2))


def _candidate_provenance(
    name: str,
    value: str,
    latest_user_text: str,
    evidence_text: str,
) -> str | None:
    normalized = value.casefold()
    if normalized in latest_user_text.casefold():
        return "user"
    if name == "description":
        return "rag" if _description_supported(value, evidence_text) else None
    if normalized in evidence_text.casefold():
        return "rag"
    return None


def _merge_draft_fields(
    state: IssueWorkflowState,
    candidates: dict[str, str | None],
    *,
    latest_user_text: str,
    evidence_text: str,
    accept_structured_arguments: bool = False,
) -> list[str]:
    invalid: list[str] = []
    for name, raw_value in candidates.items():
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if name in {"cust_id", "device_id"}:
            value = value.upper()
        if not _valid_field(name, value):
            invalid.append(name)
            continue
        provenance = _candidate_provenance(
            name,
            value,
            latest_user_text,
            evidence_text,
        )
        # Existing validated values can be repeated by the model without having
        # to appear verbatim in the latest user utterance again.
        if provenance is None and getattr(state, name) == value:
            provenance = state.provenance.get(name, "workflow")
        # Explicit function arguments are the LLM's structured interpretation of
        # the current conversation. Once they satisfy the deterministic field
        # schema, retain them for the mandatory confirmation turn instead of
        # requiring a second, independent text parser to reach the same value.
        # RAG hydration does not use this path and remains evidence-validated.
        if provenance is None and accept_structured_arguments:
            provenance = "llm_interpreted"
        if provenance is None:
            invalid.append(name)
            continue
        setattr(state, name, value)
        state.provenance[name] = provenance
    return invalid


def _field_list(names: list[str]) -> str:
    labels = [ISSUE_FIELD_LABELS.get(name, name) for name in names]
    if len(labels) <= 1:
        return labels[0] if labels else ""
    return ", ".join(labels[:-1]) + f" and {labels[-1]}"


def _workflow_message(
    state: IssueWorkflowState,
    *,
    invalid: list[str] | None = None,
) -> str:
    invalid = invalid or []
    missing = state.missing_fields()
    if invalid:
        invalid_text = _field_list(invalid)
        if missing:
            return (
                f"I couldn't verify the {invalid_text}. I still need "
                f"{_field_list(missing)} before I can prepare the issue."
            )
        return f"I couldn't verify the {invalid_text}. Please provide it again."
    if missing:
        return f"I still need {_field_list(missing)} before I can prepare the issue."
    return (
        "I have the complaint details ready: customer ID "
        f"{state.cust_id}, email {state.email}, mobile {state.mobile}, device ID "
        f"{state.device_id}, and description: {state.description}. "
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
    operation: Literal["start", "update", "confirm", "cancel", "status", "defer"],
    evidence_id: str | None = None,
    customer_name: str | None = None,
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
        operation: One of start, update, confirm, cancel, status, or defer. Use start for the
            initial action request, update when the user supplies or corrects
            details, confirm only after the assistant asked for confirmation,
            cancel when the user declines or abandons the draft, status for a
            progress question, and defer when the turn does not change the draft.
        evidence_id: Grounded evidence anchor from the immediately preceding
            retrieved turn when this action continues that turn.
        customer_name: Customer name when established by the conversation.
        cust_id: Customer ID, C followed by exactly 6 digits.
        email: Customer email address.
        mobile: Ten-digit Indian mobile number beginning with 6, 7, 8, or 9.
        device_id: Device ID, MSW followed by exactly 8 digits.
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

    explicit_required_values = any(
        value is not None
        for value in (cust_id, email, mobile, device_id, description)
    )
    anchor = _grounded_anchor(params, evidence_id)
    implicitly_bound = (
        operation == "start"
        and anchor is not None
        and not explicit_required_values
    )
    bound_anchor = anchor if evidence_id or implicitly_bound else None
    payload = _available_rag_payload(params, bound_anchor or anchor)
    evidence_text, all_source_refs = _rag_evidence(payload)
    hydrated: dict[str, str] = {}
    source_refs: list[dict[str, Any]] = []
    if bound_anchor is not None and isinstance(payload, dict):
        hydrated, supporting_chunks = _unique_grounded_fields(payload)
        source_refs = _source_refs(supporting_chunks)
        state.evidence_id = getattr(bound_anchor, "evidence_id", evidence_id)
        _merge_draft_fields(
            state,
            hydrated,
            latest_user_text=_latest_user_text(params, state),
            evidence_text=evidence_text,
        )
    invalid = _merge_draft_fields(
        state,
        {
            "customer_name": customer_name,
            "cust_id": cust_id,
            "email": email,
            "mobile": mobile,
            "device_id": device_id,
            "description": description,
        },
        latest_user_text=_latest_user_text(params, state),
        evidence_text=evidence_text,
        accept_structured_arguments=True,
    )
    if state.email:
        state.field_fragment_buffer.clear()
    if (
        (source_refs or all_source_refs)
        and not state.source_refs
        and any(source == "rag" for source in state.provenance.values())
    ):
        state.source_refs = source_refs or all_source_refs
    state.status = (
        "collecting_fields"
        if state.missing_fields() or invalid
        else "awaiting_confirmation"
    )
    message = _workflow_message(state, invalid=invalid)
    result = {
        "status": state.status,
        "message": message,
        "missing_fields": state.missing_fields(),
        "invalid_fields": invalid,
        "draft": state.draft(),
        "provenance": dict(state.provenance),
        "evidence_id": state.evidence_id,
        "hydrated_fields": sorted(hydrated),
        "source_refs": list(state.source_refs),
    }
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


async def raise_issue(
    params: FunctionCallParams,
    cust_id: str,
    email: str,
    mobile: str,
    device_id: str,
    description: str
):
    """Raise a complaint issue and save it to the database.
    
    Args:
        cust_id: Customer ID. Must start with 'C' followed by 6 digits (e.g. C123456).
        email: Customer's email address.
        mobile: Customer's mobile number. Must be a 10-digit Indian number starting with 6, 7, 8, or 9 (exclude +91).
        device_id: Device ID. Must start with 'MSW' followed by 8 digits (e.g. MSW12345678).
        description: A brief description of the issue.
    """
    await _publish_tool_filler(params)
    await _publish_tool_event(params, "in_progress")
    errors = []
    
    if not re.match(r"^C\d{6}$", cust_id):
        errors.append("Invalid cust_id format. Must start with 'C' followed by 6 digits.")
        
    if not re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email):
        errors.append("Invalid email format.")
        
    if not re.match(r"^[6-9]\d{9}$", mobile):
        errors.append("Invalid mobile format. Must be a 10-digit number starting with 6, 7, 8, or 9.")
        
    if not re.match(r"^MSW\d{8}$", device_id):
        errors.append("Invalid device_id format. Must start with 'MSW' followed by 8 digits.")
        
    if errors:
        error_msg = "Validation failed: " + "; ".join(errors) + " Please ask the user for correct information."
        await _return_tool_result(params, {"status": "error", "message": error_msg})
        return

    try:
        async with asyncio.timeout(issue_tool_timeout_seconds()):
            async with VoiceSessionLocal() as session:
                new_issue = Issue(
                    cust_id=cust_id,
                    email=email,
                    mobile=mobile,
                    device_id=device_id,
                    description=description
                )
                session.add(new_issue)
                # Flush performs the INSERT and populates the generated primary
                # key. A post-commit refresh would add a second database round
                # trip solely to read data we already have.
                await session.flush()
                issue_id = new_issue.id
                await session.commit()
    except TimeoutError:
        await _return_tool_result(params, {
            "status": "timeout",
            "message": "Issue creation timed out and was not confirmed. Ask the user to retry later.",
        })
        return
    except asyncio.CancelledError:
        await _publish_tool_event(params, "cancelled")
        raise
    except Exception:
        await _return_tool_result(params, {
            "status": "error",
            "message": "Issue creation failed and was not confirmed. Ask the user to retry later.",
        })
        return
    
    await _return_tool_result(params, {
        "status": "success",
        "message": f"Issue #{issue_id} has been successfully raised."
    })
