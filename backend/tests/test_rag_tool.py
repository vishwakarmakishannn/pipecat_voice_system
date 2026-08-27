from types import SimpleNamespace

import pytest
from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper
from pipecat.processors.aggregators.llm_context import LLMContext

from tools.rag import openai_rag_tool_schema, search_uploaded_content


def test_rag_tool_requires_a_contextual_standalone_query():
    schema = DirectFunctionWrapper(search_uploaded_content).to_function_schema()
    query_description = schema.properties["query"]["description"]

    assert "standalone" in query_description
    assert "conversation history" in query_description
    assert "corrections" in query_description
    assert "underspecified" in query_description
    assert openai_rag_tool_schema()["function"]["name"] == "search_uploaded_content"


@pytest.mark.anyio
async def test_rag_tool_returns_grounded_result_and_preserves_tool_schema():
    delivered = []
    queued = []
    rag_call = {
        "rag_call_id": "rag-tool-test",
        "result": {"chunk_count": 1, "chunks": [{"filename": "rohan.pdf"}]},
    }

    class Retrieval:
        async def retrieve_for_tool(self, query):
            assert query == "Rohan Sharma from the uploaded PDF"
            return {
                "status": "ok",
                "query": query,
                "chunk_count": 1,
                "chunks": [{"filename": "rohan.pdf", "content": "Rohan Sharma"}],
                "rag_call": rag_call,
            }

    class Worker:
        async def queue_frames(self, frames):
            queued.extend(frames)

    async def capture(result, *, properties=None):
        delivered.append((result, properties))

    context = LLMContext(
        messages=[{"role": "user", "content": "It is uploaded, check again"}],
        tools=[search_uploaded_content],
        tool_choice="auto",
    )
    original_tools = context.tools
    params = SimpleNamespace(
        result_callback=capture,
        context=context,
        app_resources={"context_retrieval": Retrieval()},
        pipeline_worker=Worker(),
    )

    await search_uploaded_content(
        params,
        query="Rohan Sharma from the uploaded PDF",
    )

    assert delivered[0][0] == {
        "status": "ok",
        "query": "Rohan Sharma from the uploaded PDF",
        "chunk_count": 1,
        "chunks": [{"filename": "rohan.pdf", "content": "Rohan Sharma"}],
    }
    assert delivered[0][1].run_llm is True
    assert context.tools is original_tools
    assert context.tool_choice == "auto"
    assert queued[0].message["data"]["type"] == "rag_call"
    assert queued[0].message["data"]["payload"] == rag_call
    assert "rag_call" not in delivered[0][0]


@pytest.mark.anyio
async def test_rag_tool_timeout_does_not_claim_upload_is_missing():
    delivered = []

    class Retrieval:
        async def retrieve_for_tool(self, _query):
            return {
                "status": "timeout",
                "message": (
                    "Uploaded content is available, but document retrieval timed out."
                ),
            }

    async def capture(result, *, properties=None):
        delivered.append(result)

    await search_uploaded_content(
        SimpleNamespace(
            result_callback=capture,
            app_resources={"context_retrieval": Retrieval()},
        ),
        query="Rohan Sharma from the uploaded PDF",
    )

    assert delivered == [
        {
            "status": "timeout",
            "message": "Uploaded content is available, but document retrieval timed out.",
        }
    ]
