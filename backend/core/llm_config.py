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
    return _bounded_seconds("VOICE_LLM_TOTAL_TIMEOUT_SECONDS", 20.0, 1.0, 120.0)
