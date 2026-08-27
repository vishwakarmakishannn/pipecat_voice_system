import asyncio
from types import SimpleNamespace

import pytest
from tavily.errors import TimeoutError as TavilyTimeoutError

import tools.tavily as tavily_module
from core.tool_config import (
    issue_tool_timeout_seconds,
    tool_timeout_seconds,
    web_search_attempt_timeout_seconds,
    web_search_max_attempts,
    web_search_timeout_seconds,
    web_search_tool_timeout_seconds,
)


def test_tool_timeout_config_is_validated(monkeypatch):
    monkeypatch.setenv("VOICE_TOOL_TIMEOUT_SECONDS", "0.1")
    with pytest.raises(ValueError, match="VOICE_TOOL_TIMEOUT_SECONDS"):
        tool_timeout_seconds()


def test_web_search_default_is_bounded_for_live_voice(monkeypatch):
    monkeypatch.delenv("VOICE_WEB_SEARCH_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("VOICE_WEB_SEARCH_ATTEMPT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("VOICE_WEB_SEARCH_MAX_ATTEMPTS", raising=False)

    assert web_search_timeout_seconds() == 4.0
    assert web_search_attempt_timeout_seconds() == 3.5
    assert web_search_max_attempts() == 1


@pytest.mark.anyio
async def test_tavily_timeout_returns_fallback(monkeypatch):
    results = []

    async def callback(result, *, properties=None):
        results.append((result, properties))

    class SlowClient:
        async def search(self, **_kwargs):
            await asyncio.sleep(0.05)

    monkeypatch.setenv("TAVILY_API_KEY", "test")
    monkeypatch.setenv("VOICE_TOOL_TIMEOUT_SECONDS", "0.25")
    monkeypatch.setattr(tavily_module, "web_search_timeout_seconds", lambda: 0.01)
    monkeypatch.setattr(
        tavily_module, "web_search_attempt_timeout_seconds", lambda: 0.01
    )
    monkeypatch.setattr(tavily_module, "web_search_max_attempts", lambda: 2)
    monkeypatch.setattr(tavily_module, "_get_tavily_client", lambda: SlowClient())

    await tavily_module.tavily_search(
        SimpleNamespace(result_callback=callback),
        "current news",
    )

    assert results[0][0] == {
        "status": "timeout",
        "message": "Web search timed out. Give a brief fallback answer and disclose that live results were unavailable.",
        "attempts": 1,
    }
    assert results[0][1].run_llm is True


def test_web_search_outer_deadline_exceeds_provider_deadline(monkeypatch):
    monkeypatch.setenv("VOICE_TOOL_TIMEOUT_SECONDS", "3")
    monkeypatch.setenv("VOICE_WEB_SEARCH_TIMEOUT_SECONDS", "8")

    assert web_search_tool_timeout_seconds() == 9.0
    # The decorator captures the production default at import time; runtime
    # environment overrides are still owned by run_web_search itself.
    assert tavily_module.tavily_search._pipecat_timeout_secs == 5.0


def test_per_tool_timeouts_are_independent(monkeypatch):
    monkeypatch.setenv("VOICE_WEB_SEARCH_TIMEOUT_SECONDS", "1.5")
    monkeypatch.setenv("VOICE_ISSUE_TOOL_TIMEOUT_SECONDS", "0.6")
    assert web_search_timeout_seconds() == 1.5
    assert issue_tool_timeout_seconds() == 0.6


@pytest.mark.anyio
async def test_tavily_retries_one_transient_timeout_with_explicit_deadline(monkeypatch):
    calls = []

    class FlakyClient:
        async def search(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise TavilyTimeoutError(kwargs["timeout"])
            return {"query": kwargs["query"], "results": []}

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(tavily_module, "web_search_timeout_seconds", lambda: 1.0)
    monkeypatch.setattr(
        tavily_module, "web_search_attempt_timeout_seconds", lambda: 0.2
    )
    monkeypatch.setattr(tavily_module, "web_search_max_attempts", lambda: 2)
    monkeypatch.setattr(tavily_module, "_get_tavily_client", lambda: FlakyClient())

    result = await tavily_module.run_web_search("current event")

    assert result == {"query": "current event", "answer": None, "results": []}
    assert len(calls) == 2
    assert [call["timeout"] for call in calls] == [0.2, 0.2]


@pytest.mark.anyio
async def test_tavily_sdk_timeouts_exhaust_attempts_with_timeout_status(monkeypatch):
    calls = []

    class TimedOutClient:
        async def search(self, **kwargs):
            calls.append(kwargs)
            raise TavilyTimeoutError(kwargs["timeout"])

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(tavily_module, "web_search_timeout_seconds", lambda: 1.0)
    monkeypatch.setattr(tavily_module, "web_search_attempt_timeout_seconds", lambda: 0.2)
    monkeypatch.setattr(tavily_module, "web_search_max_attempts", lambda: 2)
    monkeypatch.setattr(tavily_module, "_get_tavily_client", lambda: TimedOutClient())

    result = await tavily_module.run_web_search("current event")

    assert result["status"] == "timeout"
    assert result["attempts"] == 2
    assert len(calls) == 2
