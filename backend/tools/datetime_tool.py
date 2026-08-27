"""Deterministic current date/time tool backed by the host clock."""

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper
from pipecat.frames.frames import OutputTransportMessageUrgentFrame, TTSSpeakFrame
from pipecat.processors.aggregators.llm_context import NOT_GIVEN
from pipecat.services.llm_service import (
    FunctionCallParams,
    FunctionCallResultProperties,
)

from core.datetime_context import configured_timezone_name


def _utc_offset(value: datetime) -> str:
    offset = value.utcoffset()
    if offset is None:
        return "+00:00"
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    hours, minutes = divmod(abs(total_minutes), 60)
    return f"{sign}{hours:02d}:{minutes:02d}"


def current_datetime_result(
    timezone: str | None = None,
    *,
    now: datetime | None = None,
) -> dict:
    """Return structured current time for one valid IANA timezone."""
    timezone_name = (timezone or configured_timezone_name()).strip()
    try:
        zone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        return {
            "status": "error",
            "message": (
                f"Unknown IANA timezone {timezone_name!r}. Ask for a valid "
                "timezone such as Asia/Kolkata or America/New_York."
            ),
        }

    local_now = datetime.now(zone) if now is None else now.astimezone(zone)
    return {
        "status": "ok",
        "iso8601": local_now.isoformat(timespec="seconds"),
        "local_datetime": local_now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": local_now.date().isoformat(),
        "time": local_now.strftime("%H:%M:%S"),
        "timezone": timezone_name,
        "utc_offset": _utc_offset(local_now),
    }


async def get_current_datetime(
    params: FunctionCallParams,
    timezone: str | None = None,
):
    """Get the exact current date and time in an IANA timezone.

    Use this for exact clock-time questions, timezone conversions, and deadline
    calculations that depend on the current time. Do not use it for news,
    prices, office hours, schedules, product releases, political officeholders,
    or other current external facts; use web search for those when available.

    Args:
        timezone: Optional IANA timezone such as Asia/Kolkata or
            America/New_York. Omit it to use the assistant's configured timezone.
    """
    result = current_datetime_result(timezone)
    context = getattr(params, "context", None)
    if context is not None:
        # The result pass must answer the user instead of selecting the clock
        # tool repeatedly under automatic tool choice.
        context.set_tools([])
        context.set_tool_choice(NOT_GIVEN)
    worker = getattr(params, "pipeline_worker", None)
    if result.get("status") == "ok" and worker is not None:
        spoken_text = (
            f"It is {result.get('time', result.get('local_datetime', 'now'))} "
            f"on {result.get('date', '')} in {result['timezone']}."
        )
        tool_call_id = getattr(params, "tool_call_id", None) or "datetime"
        await worker.queue_frames([
                # Queue speech before the dashboard transcript so a local
                # deterministic answer needs no second LLM request.
                TTSSpeakFrame(spoken_text, append_to_context=True),
                OutputTransportMessageUrgentFrame({
                    "label": "rtvi-ai",
                    "type": "server-message",
                    "data": {
                        "type": "assistant_transcript",
                        "payload": {
                            "id": f"datetime-{tool_call_id}",
                            "text": spoken_text,
                            "source": "datetime_tool",
                        },
                    },
                }),
        ])
        await params.result_callback(
            result,
            properties=FunctionCallResultProperties(run_llm=False),
        )
        return
    await params.result_callback(
        result,
        properties=FunctionCallResultProperties(run_llm=True),
    )


def openai_datetime_tool_schema() -> dict:
    """Return the same OpenAI schema Pipecat sends for the direct function."""
    schema = DirectFunctionWrapper(get_current_datetime).to_function_schema()
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
