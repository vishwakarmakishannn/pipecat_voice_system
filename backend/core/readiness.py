"""Fast, side-effect-free validation for voice provider readiness."""

import os


_PROVIDERS = {
    "llm": {
        "env": "LLM_PROVIDER",
        "default": "google",
        "credentials": {
            "google": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "local": None,
        },
    },
    "stt": {
        "env": "STT_PROVIDER",
        "default": "deepgram",
        "credentials": {
            "deepgram": "DEEPGRAM_API_KEY",
            "mlxwhisper": None,
            "moonshine": None,
            "whisper": None,
        },
    },
    "tts": {
        "env": "TTS_PROVIDER",
        "default": "deepgram",
        "credentials": {
            "deepgram": "DEEPGRAM_API_KEY",
            "cartesia": "CARTESIA_API_KEY",
            "kokoro": None,
            "piper": None,
        },
    },
}


def validate_voice_provider_configuration() -> dict[str, str]:
    """Return selected providers or raise with a non-secret configuration error."""
    selected = {}
    errors = []
    try:
        from core.datetime_context import configured_timezone_name

        configured_timezone_name()
    except ValueError as exc:
        errors.append(str(exc))
    try:
        from core.context_summary_config import load_voice_context_summary_config

        summary_config = load_voice_context_summary_config()
        if summary_config.enabled and not os.getenv("GROQ_API_KEY", "").strip():
            errors.append(
                "GROQ_API_KEY is not configured for live context summarization"
            )
    except ValueError as exc:
        errors.append(str(exc))
    for component, config in _PROVIDERS.items():
        provider = os.getenv(config["env"], config["default"]).strip().lower()
        selected[component] = provider
        credentials = config["credentials"]
        if provider not in credentials:
            errors.append(f"unsupported {component} provider {provider!r}")
            continue
        credential_env = credentials[provider]
        if credential_env and not os.getenv(credential_env, "").strip():
            errors.append(f"{credential_env} is not configured")
        if component == "stt" and provider == "whisper":
            try:
                from providers.local.stt.config import load_whisper_config
                load_whisper_config()
            except ValueError as exc:
                errors.append(str(exc))
        if component == "stt" and provider == "mlxwhisper":
            try:
                from providers.local.stt.mlx_config import (
                    load_mlx_whisper_config,
                )

                load_mlx_whisper_config()
            except ValueError as exc:
                errors.append(str(exc))
        if component == "stt" and provider == "moonshine":
            try:
                from providers.local.stt.moonshine_config import (
                    load_moonshine_config,
                )

                load_moonshine_config()
            except ValueError as exc:
                errors.append(str(exc))
        if component == "llm" and provider == "local":
            try:
                from providers.local.llm.config import load_local_llm_config
                load_local_llm_config()
            except ValueError as exc:
                errors.append(str(exc))
        if component == "tts" and provider == "kokoro":
            try:
                from providers.tts.kokoro_config import (
                    load_kokoro_config,
                    validate_kokoro_runtime,
                )

                validate_kokoro_runtime()
                load_kokoro_config()
            except ValueError as exc:
                errors.append(str(exc))
    if errors:
        raise ValueError("; ".join(errors))
    return selected
