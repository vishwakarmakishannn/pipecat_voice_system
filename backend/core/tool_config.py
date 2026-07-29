import os


def tool_timeout_seconds() -> float:
    raw = os.getenv("VOICE_TOOL_TIMEOUT_SECONDS", "3")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"VOICE_TOOL_TIMEOUT_SECONDS must be a number, got {raw!r}") from exc
    if not 0.25 <= value <= 30:
        raise ValueError(
            f"VOICE_TOOL_TIMEOUT_SECONDS must be between 0.25 and 30, got {value}"
        )
    return value


def _specific_timeout(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not 0.1 <= value <= 10:
        raise ValueError(f"{name} must be between 0.1 and 10, got {value}")
    return value


def web_search_timeout_seconds() -> float:
    return _specific_timeout("VOICE_WEB_SEARCH_TIMEOUT_SECONDS", 4.0)


def issue_tool_timeout_seconds() -> float:
    return _specific_timeout("VOICE_ISSUE_TOOL_TIMEOUT_SECONDS", 1.0)


def tool_filler_delay_seconds() -> float:
    # Do not start a second TTS context for retrievals that finish almost
    # immediately. Deepgram serializes audio contexts, so an eager filler can
    # otherwise block (and audibly collide with) the real LLM response.
    raw = os.getenv("VOICE_TOOL_FILLER_DELAY_MS", "400")
    try:
        value_ms = float(raw)
    except ValueError as exc:
        raise ValueError(f"VOICE_TOOL_FILLER_DELAY_MS must be a number, got {raw!r}") from exc
    if not 0 <= value_ms <= 5000:
        raise ValueError(
            f"VOICE_TOOL_FILLER_DELAY_MS must be between 0 and 5000, got {value_ms}"
        )
    return value_ms / 1000


def tool_filler_enabled() -> bool:
    raw = os.getenv("VOICE_TOOL_FILLER_ENABLED", "true").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"VOICE_TOOL_FILLER_ENABLED must be a boolean, got {raw!r}")
