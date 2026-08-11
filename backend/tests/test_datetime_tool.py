from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper
from pipecat.processors.aggregators.llm_context import LLMContext

from core.datetime_context import (
    SESSION_DATE_CONTEXT_MARKER,
    configured_timezone_name,
    session_date_context,
)
from core.prompt_config import load_system_prompt
from tools.datetime_tool import current_datetime_result, get_current_datetime


FIXED_UTC = datetime(2026, 8, 9, 20, 0, 5, tzinfo=timezone.utc)


def test_configured_timezone_defaults_to_india_and_validates_iana(monkeypatch):
    monkeypatch.delenv("VOICE_TIMEZONE", raising=False)
    assert configured_timezone_name() == "Asia/Kolkata"

    monkeypatch.setenv("VOICE_TIMEZONE", "Not/A_Zone")
    with pytest.raises(ValueError, match="valid IANA timezone"):
        configured_timezone_name()


def test_session_context_contains_date_and_timezone_without_clock_time():
    context = session_date_context(
        timezone_name="Asia/Kolkata",
        now=FIXED_UTC,
    )

    assert SESSION_DATE_CONTEXT_MARKER in context
    assert "August 10, 2026 (2026-08-10)" in context
    assert "Configured timezone: Asia/Kolkata" in context
    assert "01:30" not in context


def test_system_prompt_appends_one_session_date_and_new_session_gets_new_date():
    first_session = load_system_prompt(
        timezone_name="Asia/Kolkata",
        now=FIXED_UTC,
    )
    next_session = load_system_prompt(
        timezone_name="Asia/Kolkata",
        now=datetime(2026, 8, 10, 20, 0, tzinfo=timezone.utc),
    )

    assert first_session.count(SESSION_DATE_CONTEXT_MARKER) == 1
    assert "August 10, 2026" in first_session
    assert next_session.count(SESSION_DATE_CONTEXT_MARKER) == 1
    assert "August 11, 2026" in next_session


def test_current_datetime_result_returns_structured_zoned_time():
    result = current_datetime_result("Asia/Kolkata", now=FIXED_UTC)

    assert result == {
        "status": "ok",
        "iso8601": "2026-08-10T01:30:05+05:30",
        "local_datetime": "2026-08-10 01:30:05",
        "date": "2026-08-10",
        "time": "01:30:05",
        "timezone": "Asia/Kolkata",
        "utc_offset": "+05:30",
    }


def test_current_datetime_result_supports_other_iana_zones_and_invalid_input():
    new_york = current_datetime_result("America/New_York", now=FIXED_UTC)
    invalid = current_datetime_result("Mars/Olympus_Mons", now=FIXED_UTC)

    assert new_york["iso8601"] == "2026-08-09T16:00:05-04:00"
    assert new_york["utc_offset"] == "-04:00"
    assert invalid["status"] == "error"
    assert "IANA timezone" in invalid["message"]


def test_datetime_tool_description_separates_clock_from_current_facts():
    schema = DirectFunctionWrapper(get_current_datetime).to_function_schema()

    assert "deadline" in schema.description
    assert "news" in schema.description
    assert "IANA timezone" in schema.properties["timezone"]["description"]


@pytest.mark.anyio
async def test_datetime_tool_returns_result_and_disables_repeat_selection(monkeypatch):
    delivered = []

    async def capture(result, *, properties=None):
        delivered.append((result, properties))

    monkeypatch.setattr(
        "tools.datetime_tool.current_datetime_result",
        lambda requested: {
            "status": "ok",
            "timezone": requested,
            "iso8601": "2026-08-10T01:30:05+05:30",
        },
    )
    context = LLMContext(
        messages=[{"role": "user", "content": "What time is it?"}],
        tools=[get_current_datetime],
        tool_choice="auto",
    )
    params = SimpleNamespace(result_callback=capture, context=context)

    await get_current_datetime(params, timezone="Asia/Kolkata")

    assert delivered[0][0]["timezone"] == "Asia/Kolkata"
    assert delivered[0][1].run_llm is True
    assert repr(context.tools) == "NOT_GIVEN"
    assert repr(context.tool_choice) == "NOT_GIVEN"
