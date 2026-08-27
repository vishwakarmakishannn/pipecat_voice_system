"""Configuration for the local Moonshine streaming STT provider."""

import os
from dataclasses import dataclass
from pathlib import Path

from core.audio_config import audio_input_sample_rate


MOONSHINE_SAMPLE_RATE = 16000
MOONSHINE_MODEL = "medium-streaming"
_MODEL_ALIASES = {
    MOONSHINE_MODEL: MOONSHINE_MODEL,
    "moonshine-v2-medium": MOONSHINE_MODEL,
}


@dataclass(frozen=True)
class MoonshineConfig:
    model: str
    language: str
    update_interval_seconds: float
    vad_window_duration_seconds: float
    finalize_grace_seconds: float
    ttfs_p99_latency_seconds: float
    model_dir: Path | None
    cache_dir: Path | None


def _non_empty(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _model() -> str:
    raw = _non_empty("MOONSHINE_MODEL", MOONSHINE_MODEL).lower()
    try:
        return _MODEL_ALIASES[raw]
    except KeyError as exc:
        raise ValueError(
            "MOONSHINE_MODEL must be 'medium-streaming' for the English "
            f"245M streaming model, got {raw!r}"
        ) from exc


def _language() -> str:
    language = _non_empty("MOONSHINE_LANGUAGE", "en").lower()
    if language != "en":
        raise ValueError(
            "MOONSHINE_LANGUAGE must be 'en' because medium-streaming is "
            f"English-only, got {language!r}"
        )
    return language


def _update_interval() -> float:
    # Moonshine only publishes implicit stream updates on this cadence. A
    # quarter-second interval was directly visible in endpoint-to-final-STT
    # latency, so use the provider's supported 100 ms floor for live voice.
    raw = os.getenv("MOONSHINE_UPDATE_INTERVAL_SECONDS", "0.10").strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            "MOONSHINE_UPDATE_INTERVAL_SECONDS must be a number, "
            f"got {raw!r}"
        ) from exc
    if not 0.1 <= value <= 2.0:
        raise ValueError(
            "MOONSHINE_UPDATE_INTERVAL_SECONDS must be between 0.1 and 2.0, "
            f"got {value}"
        )
    return value


def _vad_window_duration() -> float:
    # Moonshine defaults to a 500 ms native VAD averaging window. External
    # Pipecat VAD has already identified speech end, so use a shorter native
    # window to avoid holding the final transcript for another half second.
    raw = os.getenv(
        "MOONSHINE_VAD_WINDOW_DURATION_SECONDS",
        "0.25",
    ).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            "MOONSHINE_VAD_WINDOW_DURATION_SECONDS must be a number, "
            f"got {raw!r}"
        ) from exc
    if not 0.1 <= value <= 2.0:
        raise ValueError(
            "MOONSHINE_VAD_WINDOW_DURATION_SECONDS must be between 0.1 "
            f"and 2.0, got {value}"
        )
    return value


def _finalize_grace() -> float:
    """Maximum wait for native line completion before a safety flush.

    This is not an unconditional endpointing delay. Moonshine finals are
    forwarded immediately when its native VAD completes the line; the grace
    only bounds a stalled native stream after Pipecat has observed speech end.
    """
    raw = os.getenv("MOONSHINE_FINALIZE_GRACE_SECONDS", "0.35").strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            "MOONSHINE_FINALIZE_GRACE_SECONDS must be a number, "
            f"got {raw!r}"
        ) from exc
    if not 0.1 <= value <= 2.0:
        raise ValueError(
            "MOONSHINE_FINALIZE_GRACE_SECONDS must be between 0.1 and 2.0, "
            f"got {value}"
        )
    return value


def _ttfs_p99_latency() -> float:
    raw = os.getenv("MOONSHINE_TTFS_P99_SECONDS", "0.75").strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"MOONSHINE_TTFS_P99_SECONDS must be a number, got {raw!r}"
        ) from exc
    if not 0.05 <= value <= 5.0:
        raise ValueError(
            "MOONSHINE_TTFS_P99_SECONDS must be between 0.05 and 5.0, "
            f"got {value}"
        )
    return value


def _optional_directory(name: str) -> Path | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.exists() and not path.is_dir():
        raise ValueError(f"{name} must be a directory, got {str(path)!r}")
    return path


def load_moonshine_config() -> MoonshineConfig:
    """Load and validate Moonshine settings without loading the model."""
    sample_rate = audio_input_sample_rate()
    if sample_rate != MOONSHINE_SAMPLE_RATE:
        raise ValueError(
            "AUDIO_INPUT_SAMPLE_RATE must be 16000 when STT_PROVIDER=moonshine, "
            f"got {sample_rate}"
        )
    return MoonshineConfig(
        model=_model(),
        language=_language(),
        update_interval_seconds=_update_interval(),
        vad_window_duration_seconds=_vad_window_duration(),
        finalize_grace_seconds=_finalize_grace(),
        ttfs_p99_latency_seconds=_ttfs_p99_latency(),
        model_dir=_optional_directory("MOONSHINE_MODEL_DIR"),
        cache_dir=_optional_directory("MOONSHINE_VOICE_CACHE"),
    )
