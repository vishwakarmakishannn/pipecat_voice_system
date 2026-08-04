"""Configuration for the local Kokoro text-to-speech provider."""

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

from pipecat.transcriptions.language import Language


_SUPPORTED_LANGUAGES = {
    "en": Language.EN,
    "en-us": Language.EN_US,
    "en-gb": Language.EN_GB,
    "es": Language.ES,
    "fr": Language.FR,
    "hi": Language.HI,
    "it": Language.IT,
    "ja": Language.JA,
    "pt": Language.PT,
    "zh": Language.ZH,
}

_LANGUAGE_CODES = {
    Language.EN: "en-us",
    Language.EN_US: "en-us",
    Language.EN_GB: "en-gb",
    Language.ES: "es",
    Language.FR: "fr",
    Language.HI: "hi",
    Language.IT: "it",
    Language.JA: "ja",
    Language.PT: "pt",
    Language.ZH: "zh",
}

KOKORO_RELEASE_BASE_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0"
)
KOKORO_MODEL_FILES = {
    "fp32": "kokoro-v1.0.onnx",
    "fp16": "kokoro-v1.0.fp16.onnx",
    "int8": "kokoro-v1.0.int8.onnx",
}
KOKORO_VOICES_FILE = "voices-v1.0.bin"


@dataclass(frozen=True)
class KokoroConfig:
    voice: str
    language: Language
    precision: str
    model_path: Path
    voices_path: Path
    low_latency_enabled: bool
    warmup_enabled: bool
    first_chunk_chars: int
    chunk_chars: int
    min_chunk_words: int
    intra_op_threads: int
    inter_op_threads: int
    allow_spinning: bool
    download_timeout_seconds: float

    @property
    def model_id(self) -> str:
        return self.model_path.name

    @property
    def language_code(self) -> str:
        return _LANGUAGE_CODES[self.language]

    @property
    def model_url(self) -> str:
        return f"{KOKORO_RELEASE_BASE_URL}/{KOKORO_MODEL_FILES[self.precision]}"

    @property
    def voices_url(self) -> str:
        return f"{KOKORO_RELEASE_BASE_URL}/{KOKORO_VOICES_FILE}"


def _non_empty(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
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


def _language() -> Language:
    raw = _non_empty("KOKORO_LANGUAGE", "en-US")
    normalized = raw.lower().replace("_", "-")
    try:
        return _SUPPORTED_LANGUAGES[normalized]
    except KeyError as exc:
        supported = ", ".join(_SUPPORTED_LANGUAGES)
        raise ValueError(
            f"KOKORO_LANGUAGE must be one of {supported}, got {raw!r}"
        ) from exc


def _precision() -> str:
    precision = _non_empty("KOKORO_MODEL_PRECISION", "fp16").lower()
    if precision not in KOKORO_MODEL_FILES:
        supported = ", ".join(KOKORO_MODEL_FILES)
        raise ValueError(
            f"KOKORO_MODEL_PRECISION must be one of {supported}, got {precision!r}"
        )
    return precision


def _cache_dir() -> Path:
    raw = os.getenv(
        "KOKORO_CACHE_DIR",
        str(Path.home() / ".cache" / "pipecat" / "kokoro-onnx"),
    ).strip()
    if not raw:
        raise ValueError("KOKORO_CACHE_DIR must not be empty")
    path = Path(raw).expanduser()
    if path.exists() and not path.is_dir():
        raise ValueError(
            f"KOKORO_CACHE_DIR must be a directory, got {str(path)!r}"
        )
    return path


def _model_paths(precision: str) -> tuple[Path, Path]:
    raw_model = os.getenv("KOKORO_MODEL_PATH", "").strip()
    raw_voices = os.getenv("KOKORO_VOICES_PATH", "").strip()
    if bool(raw_model) != bool(raw_voices):
        raise ValueError(
            "KOKORO_MODEL_PATH and KOKORO_VOICES_PATH must be configured together"
        )
    if not raw_model:
        cache_dir = _cache_dir()
        return (
            cache_dir / KOKORO_MODEL_FILES[precision],
            cache_dir / KOKORO_VOICES_FILE,
        )

    model_path = Path(raw_model).expanduser()
    voices_path = Path(raw_voices).expanduser()
    for name, path in (
        ("KOKORO_MODEL_PATH", model_path),
        ("KOKORO_VOICES_PATH", voices_path),
    ):
        if not path.is_file():
            raise ValueError(f"{name} must point to an existing file, got {str(path)!r}")
    return model_path, voices_path


def validate_kokoro_runtime() -> None:
    """Check optional runtime availability without importing or loading the model."""
    if importlib.util.find_spec("kokoro_onnx") is None:
        raise ValueError(
            'Kokoro runtime is not installed; install "pipecat-ai[kokoro]"'
        )
    if importlib.util.find_spec("onnxruntime") is None:
        raise ValueError(
            'Kokoro runtime is not installed; install "pipecat-ai[kokoro]"'
        )


def load_kokoro_config() -> KokoroConfig:
    """Load and validate Kokoro settings without downloading or loading the model."""
    precision = _precision()
    model_path, voices_path = _model_paths(precision)
    return KokoroConfig(
        voice=_non_empty("KOKORO_VOICE_ID", "af_heart"),
        language=_language(),
        precision=precision,
        model_path=model_path,
        voices_path=voices_path,
        low_latency_enabled=_boolean("KOKORO_LOW_LATENCY", True),
        warmup_enabled=_boolean("KOKORO_WARMUP_ENABLED", True),
        first_chunk_chars=_bounded_int(
            "KOKORO_FIRST_CHUNK_CHARS", 12, minimum=8, maximum=200
        ),
        chunk_chars=_bounded_int(
            "KOKORO_CHUNK_CHARS", 80, minimum=16, maximum=500
        ),
        min_chunk_words=_bounded_int(
            "KOKORO_MIN_CHUNK_WORDS", 2, minimum=1, maximum=20
        ),
        intra_op_threads=_bounded_int(
            "KOKORO_INTRA_OP_THREADS",
            min(4, os.cpu_count() or 1),
            minimum=1,
            maximum=64,
        ),
        inter_op_threads=_bounded_int(
            "KOKORO_INTER_OP_THREADS", 1, minimum=1, maximum=16
        ),
        allow_spinning=_boolean("KOKORO_ALLOW_SPINNING", False),
        download_timeout_seconds=_bounded_float(
            "KOKORO_DOWNLOAD_TIMEOUT_SECONDS",
            300.0,
            minimum=10.0,
            maximum=1800.0,
        ),
    )
