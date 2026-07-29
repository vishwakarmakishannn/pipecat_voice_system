import re
import asyncio
from pipecat.frames.frames import OutputTransportMessageFrame, TTSSpeakFrame
from pipecat.services.llm_service import FunctionCallParams
from core.database import VoiceSessionLocal
from core.models import Issue
from core.tool_config import issue_tool_timeout_seconds, tool_filler_enabled


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
    await params.result_callback(result)


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
