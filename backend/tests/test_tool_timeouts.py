import asyncio
from types import SimpleNamespace

import pytest

import tools.tavily as tavily_module
from core.tool_config import issue_tool_timeout_seconds, tool_timeout_seconds, web_search_timeout_seconds


def test_tool_timeout_config_is_validated(monkeypatch):
    monkeypatch.setenv("VOICE_TOOL_TIMEOUT_SECONDS", "0.1")
    with pytest.raises(ValueError, match="VOICE_TOOL_TIMEOUT_SECONDS"):
        tool_timeout_seconds()


def test_web_search_default_allows_tool_filler_window(monkeypatch):
    monkeypatch.delenv("VOICE_WEB_SEARCH_TIMEOUT_SECONDS", raising=False)

    assert web_search_timeout_seconds() == 4.0


@pytest.mark.anyio
async def test_tavily_timeout_returns_fallback(monkeypatch):
    results = []

    async def callback(result):
        results.append(result)

    class SlowClient:
        async def search(self, **_kwargs):
            await asyncio.sleep(0.05)

    monkeypatch.setenv("TAVILY_API_KEY", "test")
    monkeypatch.setenv("VOICE_TOOL_TIMEOUT_SECONDS", "0.25")
    monkeypatch.setattr(tavily_module, "web_search_timeout_seconds", lambda: 0.01)
    monkeypatch.setattr(tavily_module, "_get_tavily_client", lambda: SlowClient())

    await tavily_module.tavily_search(
        SimpleNamespace(result_callback=callback),
        "current news",
    )

    assert results == [{
        "status": "timeout",
        "message": "Web search timed out. Give a brief fallback answer and disclose that live results were unavailable.",
    }]


def test_per_tool_timeouts_are_independent(monkeypatch):
    monkeypatch.setenv("VOICE_WEB_SEARCH_TIMEOUT_SECONDS", "1.5")
    monkeypatch.setenv("VOICE_ISSUE_TOOL_TIMEOUT_SECONDS", "0.6")
    assert web_search_timeout_seconds() == 1.5
    assert issue_tool_timeout_seconds() == 0.6
