import os
from dataclasses import dataclass


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}")
    return value


@dataclass(frozen=True)
class EndpointingConfig:
    vad_confidence: float
    vad_start_secs: float
    vad_stop_secs: float
    vad_min_volume: float
    smart_turn_stop_secs: float
    smart_turn_pre_speech_ms: float
    smart_turn_max_duration_secs: float


def load_endpointing_config() -> EndpointingConfig:
    """Load validated endpointing controls with low-latency voice defaults."""
    return EndpointingConfig(
        vad_confidence=_bounded_float("VAD_CONFIDENCE", 0.7, 0.0, 1.0),
        vad_start_secs=_bounded_float("VAD_START_SECS", 0.15, 0.02, 2.0),
        vad_stop_secs=_bounded_float("VAD_STOP_SECS", 0.20, 0.02, 2.0),
        vad_min_volume=_bounded_float("VAD_MIN_VOLUME", 0.5, 0.0, 1.0),
        # The turn analyzer is the authoritative end-of-turn gate. A 600 ms
        # default consumed most of the direct-turn latency budget before the
        # LLM could even start. 300 ms retains semantic turn validation while
        # leaving enough budget for LLM, TTS, and transport startup.
        smart_turn_stop_secs=_bounded_float("SMART_TURN_STOP_SECS", 0.3, 0.2, 5.0),
        smart_turn_pre_speech_ms=_bounded_float("SMART_TURN_PRE_SPEECH_MS", 300.0, 0.0, 2000.0),
        smart_turn_max_duration_secs=_bounded_float("SMART_TURN_MAX_DURATION_SECS", 8.0, 1.0, 30.0),
    )


def startup_greeting() -> str:
    return os.getenv("VOICE_GREETING_TEXT", "Hello! How can I help?").strip()
