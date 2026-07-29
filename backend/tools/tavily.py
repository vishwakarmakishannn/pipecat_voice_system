import os
import asyncio
from pipecat.services.llm_service import FunctionCallParams
from tavily import AsyncTavilyClient
from core.tool_config import web_search_timeout_seconds

_tavily_client = None


def _get_tavily_client():
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return _tavily_client

async def run_web_search(query: str) -> dict:
    """Execute one bounded web search and return a compact provider-neutral result."""
    if not os.getenv("TAVILY_API_KEY"):
        return {
            "status": "unavailable",
            "message": "Web search is not configured. Answer without search or tell the user it is unavailable.",
        }

    client = _get_tavily_client()
    try:
        async with asyncio.timeout(web_search_timeout_seconds()):
            result = await client.search(
            query=query,
            search_depth="fast",
            max_results=5,
            include_answer=True,
            include_raw_content=False,
            )
    except TimeoutError:
        return {
            "status": "timeout",
            "message": "Web search timed out. Give a brief fallback answer and disclose that live results were unavailable.",
        }
    except asyncio.CancelledError:
        raise
    except Exception:
        return {
            "status": "error",
            "message": "Web search failed. Continue without live results.",
        }
    return {
        "query": result.get("query", query),
        "answer": result.get("answer"),
        "results": [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": (item.get("content") or "")[:600],
            }
            for item in result.get("results", [])[:3]
        ],
    }


async def tavily_search(params: FunctionCallParams, query: str):
    """Search the web using Tavily through Pipecat's function-call contract."""
    await params.result_callback(await run_web_search(query))
