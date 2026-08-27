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


def _choice(name: str, default: str, choices: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
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
    turn_stop_strategy: str
    speech_timeout_secs: float


def load_endpointing_config() -> EndpointingConfig:
    """Load validated endpointing controls with low-latency voice defaults."""
    return EndpointingConfig(
        vad_confidence=_bounded_float("VAD_CONFIDENCE", 0.7, 0.0, 1.0),
        vad_start_secs=_bounded_float("VAD_START_SECS", 0.15, 0.02, 2.0),
        vad_stop_secs=_bounded_float("VAD_STOP_SECS", 0.15, 0.02, 2.0),
        vad_min_volume=_bounded_float("VAD_MIN_VOLUME", 0.5, 0.0, 1.0),
        # This is SmartTurn's hard silence fallback, not a delay applied to
        # every turn. Model-complete turns still release immediately; the
        # longer fallback protects incomplete turns across natural pauses.
        smart_turn_stop_secs=_bounded_float("SMART_TURN_STOP_SECS", 0.7, 0.2, 5.0),
        smart_turn_pre_speech_ms=_bounded_float("SMART_TURN_PRE_SPEECH_MS", 300.0, 0.0, 2000.0),
        smart_turn_max_duration_secs=_bounded_float("SMART_TURN_MAX_DURATION_SECS", 8.0, 1.0, 30.0),
        # SmartTurn remains the production-safe default. The timeout strategy
        # is an explicit low-latency mode for deployments that accept a higher
        # chance of ending a turn during a natural pause.
        turn_stop_strategy=_choice(
            "TURN_STOP_STRATEGY", "smart_turn", {"smart_turn", "speech_timeout"}
        ),
        speech_timeout_secs=_bounded_float(
            "SPEECH_TIMEOUT_SECS", 0.2, 0.05, 2.0
        ),
    )


def startup_greeting() -> str:
    return os.getenv("VOICE_GREETING_TEXT", "Hello! How can I help?").strip()
