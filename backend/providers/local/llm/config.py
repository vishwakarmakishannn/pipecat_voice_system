"""Validated configuration for a local llama.cpp server."""

import os
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class LocalLLMConfig:
    base_url: str
    model: str
    api_key: str
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    max_tokens: int
    warmup_timeout_seconds: float
    max_concurrent_sessions: int

    @property
    def extra_body(self) -> dict:
        return {
            "top_k": self.top_k,
            "min_p": self.min_p,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
        }


def _non_empty(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _bounded_float(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, got {value}"
        )
    return value


def _bounded_int(
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, got {value}"
        )
    return value


def _base_url() -> str:
    value = _non_empty("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080/v1")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("LOCAL_LLM_BASE_URL must be an absolute HTTP(S) URL")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            "LOCAL_LLM_BASE_URL must point to localhost for the local provider, "
            f"got host {parsed.hostname!r}"
        )
    return value.rstrip("/")


def load_local_llm_config() -> LocalLLMConfig:
    """Load settings without contacting or starting the inference server."""
    max_concurrent_sessions = _bounded_int(
        "LOCAL_LLM_MAX_CONCURRENT_SESSIONS",
        2,
        1,
        64,
    )
    voice_sessions = _bounded_int(
        "VOICE_MAX_CONCURRENT_SESSIONS",
        8,
        1,
        1000,
    )
    if voice_sessions > max_concurrent_sessions:
        raise ValueError(
            "VOICE_MAX_CONCURRENT_SESSIONS must be at most "
            f"{max_concurrent_sessions} when LLM_PROVIDER=local, "
            f"got {voice_sessions}"
        )

    return LocalLLMConfig(
        base_url=_base_url(),
        model=_non_empty("LOCAL_LLM_MODEL", "qwen3-4b-local"),
        api_key=_non_empty("LOCAL_LLM_API_KEY", "local-no-key"),
        temperature=_bounded_float("LOCAL_LLM_TEMPERATURE", 0.7, 0.0, 2.0),
        top_p=_bounded_float("LOCAL_LLM_TOP_P", 0.8, 0.0, 1.0),
        top_k=_bounded_int("LOCAL_LLM_TOP_K", 20, 0, 1000),
        min_p=_bounded_float("LOCAL_LLM_MIN_P", 0.0, 0.0, 1.0),
        presence_penalty=_bounded_float(
            "LOCAL_LLM_PRESENCE_PENALTY",
            0.0,
            -2.0,
            2.0,
        ),
        max_tokens=_bounded_int("LOCAL_LLM_MAX_TOKENS", 192, 1, 2048),
        warmup_timeout_seconds=_bounded_float(
            "LOCAL_LLM_WARMUP_TIMEOUT_SECONDS",
            30.0,
            1.0,
            300.0,
        ),
        max_concurrent_sessions=max_concurrent_sessions,
    )
