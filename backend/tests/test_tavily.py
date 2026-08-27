from types import SimpleNamespace

import pytest
from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper
from pipecat.processors.aggregators.llm_context import LLMContext

import tools.tavily as tavily_module
from tools.raise_issue import IssueWorkflowState


def test_tavily_tool_tells_llm_to_create_a_contextual_standalone_query():
    schema = DirectFunctionWrapper(tavily_module.tavily_search).to_function_schema()
    query_description = schema.properties["query"]["description"]

    assert "standalone" in query_description
    assert "conversation history" in query_description
    assert "user corrections" in query_description
    assert "assistant answer that the user rejected" in query_description
    assert "clarify" in query_description
    assert tavily_module.openai_tavily_tool_schema()["function"]["name"] == "tavily_search"


@pytest.mark.anyio
async def test_run_web_search_preserves_provider_relevance_score(monkeypatch):
    class FakeClient:
        async def search(self, **kwargs):
            assert kwargs["query"] == "Samsung Galaxy A30s camera specifications"
            return {
                "query": kwargs["query"],
                "answer": None,
                "results": [
                    {
                        "title": "Galaxy A30s specifications",
                        "url": "https://example.com/a30s",
                        "content": "Camera specifications",
                        "score": 0.91,
                    }
                ],
            }

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(tavily_module, "_get_tavily_client", lambda: FakeClient())

    result = await tavily_module.run_web_search(
        "Samsung Galaxy A30s camera specifications"
    )

    assert result["results"][0]["score"] == 0.91


@pytest.mark.anyio
async def test_tavily_search_returns_result_through_pipecat_callback(monkeypatch):
    delivered = []

    async def fake_search(query):
        return {"query": query, "results": []}

    async def capture(result, *, properties=None):
        delivered.append((result, properties))

    monkeypatch.setattr(tavily_module, "run_web_search", fake_search)
    context = LLMContext(
        messages=[{"role": "user", "content": "Search for it"}],
        tools=[tavily_module.tavily_search],
        tool_choice="auto",
    )
    params = SimpleNamespace(result_callback=capture, context=context)

    await tavily_module.tavily_search(
        params,
        query="Samsung Galaxy A30s camera specifications",
    )

    assert delivered[0][0] == {
        "query": "Samsung Galaxy A30s camera specifications",
        "results": [],
    }
    assert delivered[0][1].run_llm is True
    assert repr(context.tools) == "NOT_GIVEN"
    assert repr(context.tool_choice) == "NOT_GIVEN"
    assert context.messages == [{"role": "user", "content": "Search for it"}]


@pytest.mark.anyio
async def test_tavily_search_converts_empty_provider_result_to_terminal_error(monkeypatch):
    delivered = []

    async def fake_search(_query):
        return None

    async def capture(result, *, properties=None):
        delivered.append((result, properties))

    monkeypatch.setattr(tavily_module, "run_web_search", fake_search)

    await tavily_module.tavily_search(
        SimpleNamespace(result_callback=capture),
        query="current information",
    )

    assert delivered[0][0]["status"] == "error"
    assert delivered[0][0]["message"]
    assert delivered[0][1].run_llm is True


@pytest.mark.anyio
async def test_tavily_remains_available_but_redacts_active_draft_values(monkeypatch):
    searched = []
    delivered = []

    async def fake_search(query):
        searched.append(query)
        return {"query": query, "results": []}

    async def capture(result, *, properties=None):
        delivered.append((result, properties))

    workflow = IssueWorkflowState(
        status="collecting_fields",
        cust_id="C001122",
        mobile="9876543210",
        device_id="MSW12345678",
    )
    monkeypatch.setattr(tavily_module, "run_web_search", fake_search)
    params = SimpleNamespace(
        result_callback=capture,
        context=None,
        app_resources={"issue_workflow": workflow},
    )

    await tavily_module.tavily_search(
        params,
        query="current Mswipe outages for MSW12345678 and customer C001122",
    )

    assert searched == ["current Mswipe outages for [redacted] and customer [redacted]"]
    assert delivered[0][0]["query_sanitized"] is True
    assert set(delivered[0][0]["redacted_fields"]) == {"cust_id", "device_id"}
    assert delivered[0][1].run_llm is True
