import os
import asyncio
import time

from loguru import logger
from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper
from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.services.llm_service import (
    FunctionCallParams,
    FunctionCallResultProperties,
)
from pipecat.processors.aggregators.llm_context import NOT_GIVEN
from tavily import AsyncTavilyClient

from core.tool_config import (
    web_search_attempt_timeout_seconds,
    web_search_max_attempts,
    web_search_timeout_seconds,
    web_search_tool_timeout_seconds,
)
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
    started_at = time.monotonic()
    total_timeout = web_search_timeout_seconds()
    attempt_timeout = web_search_attempt_timeout_seconds()
    max_attempts = web_search_max_attempts()
    result = None
    logger.info(
        "web_search status=started provider=tavily total_timeout_ms={} "
        "attempt_timeout_ms={} max_attempts={} query={!r}",
        round(total_timeout * 1000),
        round(attempt_timeout * 1000),
        max_attempts,
        query[:160],
    )
    try:
        async with asyncio.timeout(total_timeout):
            for attempt in range(1, max_attempts + 1):
                attempt_started_at = time.monotonic()
                try:
                    result = await client.search(
                        query=query,
                        search_depth="basic",
                        max_results=3,
                        chunks_per_source=1,
                        include_answer=False,
                        include_raw_content=False,
                        include_images=False,
                        include_favicon=False,
                        auto_parameters=False,
                        timeout=attempt_timeout,
                    )
                    logger.info(
                        "web_search status=attempt_completed provider=tavily "
                        "attempt={} duration_ms={}",
                        attempt,
                        round((time.monotonic() - attempt_started_at) * 1000, 1),
                    )
                    break
                except TimeoutError:
                    logger.warning(
                        "web_search status=attempt_timeout provider=tavily "
                        "attempt={} max_attempts={} duration_ms={}",
                        attempt,
                        max_attempts,
                        round((time.monotonic() - attempt_started_at) * 1000, 1),
                    )
                    if attempt == max_attempts:
                        raise
    except TimeoutError:
        logger.warning(
            "web_search status=timeout provider=tavily duration_ms={} attempts={}",
            round((time.monotonic() - started_at) * 1000, 1),
            max_attempts,
        )
        return {
            "status": "timeout",
            "message": "Web search timed out. Give a brief fallback answer and disclose that live results were unavailable.",
        }
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            "web_search status=error provider=tavily duration_ms={} error_type={}",
            round((time.monotonic() - started_at) * 1000, 1),
            type(exc).__name__,
        )
        return {
            "status": "error",
            "message": "Web search failed. Continue without live results.",
        }
    try:
        response = {
            "query": result.get("query", query),
            "answer": result.get("answer"),
            "results": [
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "content": (item.get("content") or ""),
                    "score": item.get("score"),
                }
                for item in result.get("results", [])[:3]
            ],
        }
        logger.info(
            "web_search status=completed provider=tavily duration_ms={} results={}",
            round((time.monotonic() - started_at) * 1000, 1),
            len(response["results"]),
        )
        return response
    except (AttributeError, TypeError):
        return {
            "status": "error",
            "message": "Web search returned an invalid response. Continue without live results.",
        }


@tool_options(timeout_secs=web_search_tool_timeout_seconds())
async def tavily_search(params: FunctionCallParams, query: str):
    """Search the web for information needed to answer the current request.

    Args:
        query: A concise, standalone web-search query formulated from the user's
            intent and relevant conversation history. Resolve references and
            follow-up wording, remove conversational filler and search commands,
            preserve exact constraints and user corrections, and never include
            claims from an assistant answer that the user rejected. If the
            subject is genuinely ambiguous, ask the user to clarify instead of
            calling this tool.
    """
    try:
        result = await run_web_search(query)
    except asyncio.CancelledError:
        raise
    except Exception:
        result = {
            "status": "error",
            "message": "Web search failed unexpectedly. Continue without live results.",
        }

    if not isinstance(result, dict) or not result:
        result = {
            "status": "error",
            "message": "Web search returned no usable result. Continue without live results.",
        }

    # The result pass must answer the user, not select another tool.
    context = getattr(params, "context", None)
    if context is not None:
        context.set_tools([])
        context.set_tool_choice(NOT_GIVEN)

    await params.result_callback(
        result,
        properties=FunctionCallResultProperties(run_llm=True),
    )


def openai_tavily_tool_schema() -> dict:
    """Return the same OpenAI schema Pipecat sends for the direct function."""
    schema = DirectFunctionWrapper(tavily_search).to_function_schema()
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
