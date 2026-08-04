"""Configuration for the Apple Silicon MLX Whisper STT provider."""

import os
import platform
from dataclasses import dataclass

from core.audio_config import audio_input_sample_rate


MLX_WHISPER_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class MLXWhisperConfig:
    model: str
    language: str
    temperature: float
    no_speech_threshold: float


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


def validate_mlx_whisper_platform() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError(
            "STT_PROVIDER=mlxwhisper requires an Apple Silicon Mac"
        )


def load_mlx_whisper_config() -> MLXWhisperConfig:
    """Load settings without downloading or initializing the MLX model."""
    validate_mlx_whisper_platform()
    sample_rate = audio_input_sample_rate()
    if sample_rate != MLX_WHISPER_SAMPLE_RATE:
        raise ValueError(
            "AUDIO_INPUT_SAMPLE_RATE must be 16000 when "
            f"STT_PROVIDER=mlxwhisper, got {sample_rate}"
        )
    return MLXWhisperConfig(
        model=_non_empty(
            "MLX_WHISPER_MODEL",
            "mlx-community/whisper-small-mlx",
        ),
        language=_non_empty("MLX_WHISPER_LANGUAGE", "auto").lower(),
        temperature=_bounded_float(
            "MLX_WHISPER_TEMPERATURE",
            0.0,
            0.0,
            1.0,
        ),
        no_speech_threshold=_bounded_float(
            "MLX_WHISPER_NO_SPEECH_THRESHOLD",
            0.6,
            0.0,
            1.0,
        ),
    )
