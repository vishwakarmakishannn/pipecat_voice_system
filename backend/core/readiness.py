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
        },
    },
    "stt": {
        "env": "STT_PROVIDER",
        "default": "deepgram",
        "credentials": {"deepgram": "DEEPGRAM_API_KEY"},
    },
    "tts": {
        "env": "TTS_PROVIDER",
        "default": "deepgram",
        "credentials": {
            "deepgram": "DEEPGRAM_API_KEY",
            "cartesia": "CARTESIA_API_KEY",
            "piper": None,
        },
    },
}


def validate_voice_provider_configuration() -> dict[str, str]:
    """Return selected providers or raise with a non-secret configuration error."""
    selected = {}
    errors = []
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
    if errors:
        raise ValueError("; ".join(errors))
    return selected
