"""Configuration for the local Whisper speech-to-text provider."""

import os
from dataclasses import dataclass
from pathlib import Path

from core.audio_config import audio_input_sample_rate


WHISPER_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class WhisperConfig:
    model: str
    language: str
    threads: int
    models_dir: Path | None


def _non_empty(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _thread_count() -> int:
    raw = os.getenv("WHISPER_THREADS", "4").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"WHISPER_THREADS must be an integer, got {raw!r}") from exc
    if not 1 <= value <= 64:
        raise ValueError(f"WHISPER_THREADS must be between 1 and 64, got {value}")
    return value


def _models_dir() -> Path | None:
    raw = os.getenv("WHISPER_MODELS_DIR", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.exists() and not path.is_dir():
        raise ValueError(
            f"WHISPER_MODELS_DIR must be a directory, got {str(path)!r}"
        )
    return path


def load_whisper_config() -> WhisperConfig:
    """Load and validate local Whisper settings without loading the model."""
    sample_rate = audio_input_sample_rate()
    if sample_rate != WHISPER_SAMPLE_RATE:
        raise ValueError(
            "AUDIO_INPUT_SAMPLE_RATE must be 16000 when STT_PROVIDER=whisper, "
            f"got {sample_rate}"
        )
    return WhisperConfig(
        model=_non_empty("WHISPER_MODEL", "small"),
        language=_non_empty("WHISPER_LANGUAGE", "auto").lower(),
        threads=_thread_count(),
        models_dir=_models_dir(),
    )
