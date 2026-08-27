"""Live capability gate for a candidate Groq voice-orchestration model.

This command intentionally evaluates planning only; it never executes Tavily or
another write-capable tool. Run it before promoting a Groq model in production.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv

from core.assistant_output import contains_reserved_tool_markup
from core.prompt_config import load_system_prompt
from providers.llm.groq_runtime import get_shared_groq_client, shutdown_groq_runtime
from tools.datetime_tool import openai_datetime_tool_schema
from tools.tavily import openai_tavily_tool_schema


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    messages: list[dict[str, str]]
    expected_tool: str | None
    query_terms: tuple[str, ...] = ()
    clarification_terms: tuple[str, ...] = ()


CASES = (
    EvaluationCase(
        name="timeless_no_search",
        messages=[{"role": "user", "content": "What is the full form of AI?"}],
        expected_tool=None,
    ),
    EvaluationCase(
        name="current_search",
        messages=[{"role": "user", "content": "Search the current Dell G15 price in India."}],
        expected_tool="tavily_search",
        query_terms=("dell", "g15", "price"),
    ),
    EvaluationCase(
        name="ambiguous_reference_clarifies",
        messages=[{"role": "user", "content": "So should I buy them?"}],
        expected_tool=None,
        clarification_terms=("which", "what", "refer", "them"),
    ),
    EvaluationCase(
        name="correction_builds_contextual_query",
        messages=[
            {"role": "user", "content": "I was thinking of buying a Samsung Galaxy A30s."},
            {"role": "assistant", "content": "It has a 48MP main camera."},
            {"role": "user", "content": "You are wrong with the camera specs."},
        ],
        expected_tool="tavily_search",
        query_terms=("galaxy", "a30s", "camera"),
    ),
)


def _tool_call(message):
    calls = getattr(message, "tool_calls", None) or []
    return calls[0] if calls else None


def evaluate_message(case: EvaluationCase, message) -> tuple[bool, dict]:
    content = (getattr(message, "content", None) or "").strip()
    tool_call = _tool_call(message)
    tool_name = tool_call.function.name if tool_call else None
    arguments = tool_call.function.arguments if tool_call else ""
    searchable = f"{content} {arguments}".lower()
    reasons = []
    if contains_reserved_tool_markup(content):
        reasons.append("simulated tool markup appeared in assistant content")
    if tool_name != case.expected_tool:
        reasons.append(f"expected tool {case.expected_tool!r}, received {tool_name!r}")
    missing_terms = [term for term in case.query_terms if term not in searchable]
    if missing_terms:
        reasons.append(f"standalone query omitted terms: {', '.join(missing_terms)}")
    if case.clarification_terms and not any(term in content.lower() for term in case.clarification_terms):
        reasons.append("ambiguous reference did not receive a clarification question")
    return not reasons, {
        "case": case.name,
        "passed": not reasons,
        "tool": tool_name,
        "content": content[:240],
        "arguments": arguments[:300],
        "reasons": reasons,
    }


async def main() -> int:
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print("GROQ_API_KEY is required for the live capability evaluation.", file=sys.stderr)
        return 2
    model = os.getenv("GROQ_EVALUATION_MODEL", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")).strip()
    client = get_shared_groq_client(api_key=api_key)
    tools = [openai_datetime_tool_schema(), openai_tavily_tool_schema()]
    results = []
    try:
        for case in CASES:
            try:
                completion = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": load_system_prompt()},
                        *case.messages,
                    ],
                    tools=tools,
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    temperature=0,
                    max_tokens=220,
                )
                passed, result = evaluate_message(case, completion.choices[0].message)
            except Exception as exc:
                result = {
                    "case": case.name,
                    "passed": False,
                    "tool": None,
                    "content": "",
                    "arguments": "",
                    "reasons": [
                        f"provider request failed: {type(exc).__name__}"
                    ],
                    "http_status": getattr(exc, "status_code", None),
                }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))
    finally:
        await shutdown_groq_runtime()

    passed_count = sum(result["passed"] for result in results)
    print(json.dumps({"model": model, "passed": passed_count, "total": len(results)}))
    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
