import os


def _selected_provider() -> str:
    return os.getenv("TTS_PROVIDER", "deepgram").strip().lower()


def get_tts():
    provider = _selected_provider()

    if provider == "cartesia":
        from .cartesia_tts import get_cartesia_tts
        return get_cartesia_tts()
    if provider == "piper":
        from .piper_tts import get_piper_tts
        return get_piper_tts()
    if provider == "kokoro":
        from .kokoro_tts import get_kokoro_tts
        return get_kokoro_tts()
    if provider == "deepgram":
        from .deepgram_tts import get_deepgram_tts
        return get_deepgram_tts()
    raise ValueError(
        "Unsupported TTS provider: "
        f"{provider!r}. Expected cartesia, deepgram, kokoro, or piper."
    )


async def warm_tts_provider() -> None:
    """Warm only the selected provider's process-wide runtime."""
    if _selected_provider() == "kokoro":
        from .kokoro_runtime import warm_kokoro_runtime

        await warm_kokoro_runtime()


async def shutdown_tts_provider() -> None:
    """Release process-wide resources owned by the selected provider."""
    if _selected_provider() == "kokoro":
        from .kokoro_runtime import shutdown_kokoro_runtime

        await shutdown_kokoro_runtime()
