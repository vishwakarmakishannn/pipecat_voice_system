"""Safety helpers for assistant text that resembles provider control syntax."""

from __future__ import annotations


RESERVED_TOOL_MARKERS = (
    "<function",
    "</function",
    "<tool_call",
    "</tool_call",
    "<tool>",
    "</tool>",
    "<|python_tag|>",
    "<|tool",
    "[tool_call",
    "assistant to=functions.",
)

NON_MEMORY_ASSISTANT_SOURCES = {
    "tool_filler",
    "spoken_recovery",
    "invalid_output_recovery",
}


def contains_reserved_tool_markup(text: str | None) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in RESERVED_TOOL_MARKERS)


def is_memory_safe_assistant_text(text: str | None, source: str | None = "llm") -> bool:
    if source in NON_MEMORY_ASSISTANT_SOURCES:
        return False
    return bool((text or "").strip()) and not contains_reserved_tool_markup(text)
