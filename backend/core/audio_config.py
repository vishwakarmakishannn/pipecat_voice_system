import os


def _sample_rate(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if not 8000 <= value <= 48000:
        raise ValueError(f"{name} must be between 8000 and 48000, got {value}")
    return value


def audio_input_sample_rate() -> int:
    return _sample_rate("AUDIO_INPUT_SAMPLE_RATE", 16000)


def audio_output_sample_rate(provider: str | None = None) -> int:
    provider = (provider or os.getenv("TTS_PROVIDER", "deepgram")).lower()
    default = 22050 if provider == "piper" else 24000
    return _sample_rate("AUDIO_OUTPUT_SAMPLE_RATE", default)


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}")
    return value


def trim_tts_leading_silence() -> bool:
    raw = os.getenv("TTS_TRIM_LEADING_SILENCE", "true").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"TTS_TRIM_LEADING_SILENCE must be a boolean, got {raw!r}")


def tts_silence_threshold() -> int:
    return _bounded_int("TTS_SILENCE_THRESHOLD", 128, 0, 32767)


def tts_silence_preroll_ms() -> int:
    return _bounded_int("TTS_SILENCE_PREROLL_MS", 20, 0, 100)


def audio_out_10ms_chunks() -> int:
    return _bounded_int("AUDIO_OUT_10MS_CHUNKS", 1, 1, 10)
