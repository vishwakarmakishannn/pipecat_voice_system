import os


def _selected_provider() -> str:
    return os.getenv("STT_PROVIDER", "deepgram").strip().lower()


def get_stt():
    provider = _selected_provider()

    if provider == "deepgram":
        from .deepgram_stt import get_deepgram_stt
        return get_deepgram_stt()
    if provider == "whisper":
        from providers.local.stt.whisper_stt import get_whisper_stt
        return get_whisper_stt()
    if provider == "mlxwhisper":
        from providers.local.stt.mlx_whisper_stt import get_mlx_whisper_stt
        return get_mlx_whisper_stt()
    if provider == "moonshine":
        from providers.local.stt.moonshine_stt import get_moonshine_stt
        return get_moonshine_stt()
    raise ValueError(
        "Unsupported STT provider: "
        f"{provider!r}. Expected deepgram, whisper, mlxwhisper, or moonshine."
    )


async def warm_stt_provider() -> None:
    """Warm startup-only resources for the selected STT provider."""
    if _selected_provider() == "whisper":
        from providers.local.stt.whisper_stt import warm_whisper_runtime
        await warm_whisper_runtime()
    elif _selected_provider() == "mlxwhisper":
        from providers.local.stt.mlx_whisper_stt import (
            warm_mlx_whisper_runtime,
        )

        await warm_mlx_whisper_runtime()
    elif _selected_provider() == "moonshine":
        from providers.local.stt.moonshine_stt import warm_moonshine_runtime

        await warm_moonshine_runtime()


async def shutdown_stt_provider() -> None:
    """Release process-wide resources for the selected STT provider."""
    if _selected_provider() == "whisper":
        from providers.local.stt.whisper_stt import shutdown_whisper_runtime
        await shutdown_whisper_runtime()
    elif _selected_provider() == "mlxwhisper":
        from providers.local.stt.mlx_whisper_stt import (
            shutdown_mlx_whisper_runtime,
        )

        await shutdown_mlx_whisper_runtime()
    elif _selected_provider() == "moonshine":
        from providers.local.stt.moonshine_stt import shutdown_moonshine_runtime

        await shutdown_moonshine_runtime()
