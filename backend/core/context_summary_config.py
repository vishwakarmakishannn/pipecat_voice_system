"""Validated configuration for live voice-context summarization."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class VoiceContextSummaryConfig:
    enabled: bool
    max_tokens: int
    max_messages: int
    target_tokens: int
    keep_messages: int
    timeout_seconds: float
    retry_cooldown_seconds: float
    emergency_max_messages: int
    emergency_max_chars: int
    model: str


def load_voice_context_summary_config() -> VoiceContextSummaryConfig:
    config = VoiceContextSummaryConfig(
        enabled=_env_bool("VOICE_CONTEXT_SUMMARIZATION_ENABLED", True),
        max_tokens=_env_int("VOICE_CONTEXT_SUMMARY_MAX_TOKENS", 3000),
        max_messages=_env_int("VOICE_CONTEXT_SUMMARY_MAX_MESSAGES", 20),
        target_tokens=_env_int("VOICE_CONTEXT_SUMMARY_TARGET_TOKENS", 900),
        keep_messages=_env_int("VOICE_CONTEXT_SUMMARY_KEEP_MESSAGES", 8),
        timeout_seconds=_env_float("VOICE_CONTEXT_SUMMARY_TIMEOUT_SECONDS", 6.0),
        retry_cooldown_seconds=_env_float(
            "VOICE_CONTEXT_SUMMARY_RETRY_COOLDOWN_SECONDS", 30.0
        ),
        emergency_max_messages=_env_int("VOICE_CONTEXT_EMERGENCY_MAX_MESSAGES", 40),
        emergency_max_chars=_env_int("VOICE_CONTEXT_EMERGENCY_MAX_CHARS", 24000),
        model=os.getenv(
            "GROQ_CONTEXT_SUMMARY_MODEL",
            os.getenv("GROQ_MEMORY_MODEL", "openai/gpt-oss-20b"),
        ).strip(),
    )
    errors: list[str] = []
    if config.max_tokens <= 0:
        errors.append("VOICE_CONTEXT_SUMMARY_MAX_TOKENS must be positive")
    if config.target_tokens <= 0 or config.target_tokens >= config.max_tokens:
        errors.append(
            "VOICE_CONTEXT_SUMMARY_TARGET_TOKENS must be positive and smaller than "
            "VOICE_CONTEXT_SUMMARY_MAX_TOKENS"
        )
    if config.max_messages < 2:
        errors.append("VOICE_CONTEXT_SUMMARY_MAX_MESSAGES must be at least 2")
    if config.keep_messages < 2 or config.keep_messages >= config.max_messages:
        errors.append(
            "VOICE_CONTEXT_SUMMARY_KEEP_MESSAGES must be at least 2 and smaller than "
            "VOICE_CONTEXT_SUMMARY_MAX_MESSAGES"
        )
    if not 1.0 <= config.timeout_seconds <= 30.0:
        errors.append("VOICE_CONTEXT_SUMMARY_TIMEOUT_SECONDS must be between 1 and 30")
    if config.retry_cooldown_seconds < 0:
        errors.append("VOICE_CONTEXT_SUMMARY_RETRY_COOLDOWN_SECONDS cannot be negative")
    if config.emergency_max_messages <= config.max_messages:
        errors.append(
            "VOICE_CONTEXT_EMERGENCY_MAX_MESSAGES must exceed "
            "VOICE_CONTEXT_SUMMARY_MAX_MESSAGES"
        )
    if config.emergency_max_chars < 1000:
        errors.append("VOICE_CONTEXT_EMERGENCY_MAX_CHARS must be at least 1000")
    if config.emergency_max_chars <= config.max_tokens * 4:
        errors.append(
            "VOICE_CONTEXT_EMERGENCY_MAX_CHARS must exceed the approximate "
            "VOICE_CONTEXT_SUMMARY_MAX_TOKENS character threshold"
        )
    if not config.model:
        errors.append("GROQ_CONTEXT_SUMMARY_MODEL cannot be empty")
    if errors:
        raise ValueError("; ".join(errors))
    return config
