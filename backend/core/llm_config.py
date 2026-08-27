"""Latency controls for live voice LLM inference."""

import os


def _bounded_seconds(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}")
    return value


def first_token_timeout_seconds() -> float:
    return _bounded_seconds("VOICE_LLM_FIRST_TOKEN_TIMEOUT_SECONDS", 5.0, 0.5, 30.0)


def timeout_recovery_text() -> str:
    value = os.getenv(
        "VOICE_LLM_TIMEOUT_MESSAGE",
        "I'm having trouble reaching the language service. Please try that again.",
    ).strip()
    if not value:
        raise ValueError("VOICE_LLM_TIMEOUT_MESSAGE must not be empty")
    return value


def total_timeout_seconds() -> float:
    return _bounded_seconds("VOICE_LLM_TOTAL_TIMEOUT_SECONDS", 30.0, 1.0, 120.0)


def google_hedge_delay_seconds() -> float:
    """Delay before one duplicate Gemini attempt may race a silent first one."""
    return _bounded_seconds("GOOGLE_LIVE_HEDGE_DELAY_SECONDS", 2.0, 0.1, 10.0)


def google_warmup_timeout_seconds() -> float:
    return _bounded_seconds("GOOGLE_LIVE_WARMUP_TIMEOUT_SECONDS", 2.0, 0.1, 10.0)


def llm_retry_reserve_seconds() -> float:
    """Budget retained for a final attempt and a complete spoken recovery."""
    return _bounded_seconds("VOICE_LLM_RETRY_RESERVE_SECONDS", 0.35, 0.05, 2.0)


def groq_first_attempt_timeout_seconds() -> float:
    """Bound a first Groq transport attempt so one retry can fit the voice SLA."""
    return _bounded_seconds("GROQ_LIVE_FIRST_ATTEMPT_TIMEOUT_SECONDS", 2.5, 0.25, 10.0)


def groq_live_max_attempts() -> int:
    # A second attempt used to split a 3 s first-output SLA into two requests;
    # both routinely lost before Groq's first response arrived. One attempt
    # gets the entire deadline. Operators may opt back into a retry only after
    # measuring a larger budget.
    raw = os.getenv("GROQ_LIVE_MAX_ATTEMPTS", "1")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"GROQ_LIVE_MAX_ATTEMPTS must be an integer, got {raw!r}") from exc
    if not 1 <= value <= 2:
        raise ValueError(f"GROQ_LIVE_MAX_ATTEMPTS must be between 1 and 2, got {value}")
    return value
