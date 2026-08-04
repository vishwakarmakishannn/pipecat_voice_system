import pytest

from core.readiness import validate_voice_provider_configuration


def test_voice_provider_configuration_reports_selected_providers(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "secret")
    monkeypatch.setenv("TTS_PROVIDER", "piper")

    assert validate_voice_provider_configuration() == {
        "llm": "groq",
        "stt": "deepgram",
        "tts": "piper",
    }


def test_kokoro_readiness_requires_no_tts_credential(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "secret")
    monkeypatch.setenv("TTS_PROVIDER", "kokoro")
    monkeypatch.setenv("KOKORO_VOICE_ID", "af_heart")
    monkeypatch.setenv("KOKORO_LANGUAGE", "en-US")
    monkeypatch.delenv("KOKORO_MODEL_PATH", raising=False)
    monkeypatch.delenv("KOKORO_VOICES_PATH", raising=False)

    assert validate_voice_provider_configuration() == {
        "llm": "groq",
        "stt": "deepgram",
        "tts": "kokoro",
    }


def test_kokoro_readiness_rejects_invalid_configuration(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "secret")
    monkeypatch.setenv("TTS_PROVIDER", "kokoro")
    monkeypatch.setenv("KOKORO_LANGUAGE", "unsupported")
    monkeypatch.delenv("KOKORO_MODEL_PATH", raising=False)
    monkeypatch.delenv("KOKORO_VOICES_PATH", raising=False)

    with pytest.raises(ValueError, match="KOKORO_LANGUAGE"):
        validate_voice_provider_configuration()


def test_voice_provider_configuration_rejects_missing_credentials(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.setenv("TTS_PROVIDER", "cartesia")
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)

    with pytest.raises(ValueError) as error:
        validate_voice_provider_configuration()

    message = str(error.value)
    assert "GOOGLE_API_KEY is not configured" in message
    assert "DEEPGRAM_API_KEY is not configured" in message
    assert "CARTESIA_API_KEY is not configured" in message


def test_voice_provider_configuration_rejects_unsupported_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "secret")
    monkeypatch.setenv("TTS_PROVIDER", "piper")

    with pytest.raises(ValueError, match="unsupported llm provider 'unknown'"):
        validate_voice_provider_configuration()


def test_whisper_readiness_requires_no_stt_credential(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    monkeypatch.setenv("STT_PROVIDER", "whisper")
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.setenv("TTS_PROVIDER", "piper")
    monkeypatch.setenv("AUDIO_INPUT_SAMPLE_RATE", "16000")

    assert validate_voice_provider_configuration() == {
        "llm": "groq",
        "stt": "whisper",
        "tts": "piper",
    }


def test_whisper_readiness_rejects_wrong_sample_rate(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    monkeypatch.setenv("STT_PROVIDER", "whisper")
    monkeypatch.setenv("TTS_PROVIDER", "piper")
    monkeypatch.setenv("AUDIO_INPUT_SAMPLE_RATE", "24000")

    with pytest.raises(ValueError, match="must be 16000"):
        validate_voice_provider_configuration()


def test_mlx_whisper_readiness_requires_no_stt_credential(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    monkeypatch.setenv("STT_PROVIDER", "mlxwhisper")
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.setenv("TTS_PROVIDER", "piper")
    monkeypatch.setenv("AUDIO_INPUT_SAMPLE_RATE", "16000")
    monkeypatch.setattr(
        "providers.local.stt.mlx_config.platform.system",
        lambda: "Darwin",
    )
    monkeypatch.setattr(
        "providers.local.stt.mlx_config.platform.machine",
        lambda: "arm64",
    )

    assert validate_voice_provider_configuration() == {
        "llm": "groq",
        "stt": "mlxwhisper",
        "tts": "piper",
    }


def test_mlx_whisper_readiness_rejects_unsupported_platform(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    monkeypatch.setenv("STT_PROVIDER", "mlxwhisper")
    monkeypatch.setenv("TTS_PROVIDER", "piper")
    monkeypatch.setenv("AUDIO_INPUT_SAMPLE_RATE", "16000")
    monkeypatch.setattr(
        "providers.local.stt.mlx_config.platform.system",
        lambda: "Linux",
    )

    with pytest.raises(ValueError, match="requires an Apple Silicon Mac"):
        validate_voice_provider_configuration()


def test_moonshine_readiness_requires_no_stt_credential(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    monkeypatch.setenv("STT_PROVIDER", "moonshine")
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.setenv("TTS_PROVIDER", "piper")
    monkeypatch.setenv("AUDIO_INPUT_SAMPLE_RATE", "16000")
    monkeypatch.setenv("MOONSHINE_MODEL", "medium-streaming")
    monkeypatch.setenv("MOONSHINE_LANGUAGE", "en")

    assert validate_voice_provider_configuration() == {
        "llm": "groq",
        "stt": "moonshine",
        "tts": "piper",
    }


def test_moonshine_readiness_rejects_non_english_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    monkeypatch.setenv("STT_PROVIDER", "moonshine")
    monkeypatch.setenv("TTS_PROVIDER", "piper")
    monkeypatch.setenv("AUDIO_INPUT_SAMPLE_RATE", "16000")
    monkeypatch.setenv("MOONSHINE_LANGUAGE", "hi")

    with pytest.raises(ValueError, match="English-only"):
        validate_voice_provider_configuration()


def test_local_llm_readiness_requires_no_cloud_credential(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", " LOCAL ")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen3-4b-local")
    monkeypatch.setenv("VOICE_MAX_CONCURRENT_SESSIONS", "2")
    monkeypatch.setenv("STT_PROVIDER", "whisper")
    monkeypatch.setenv("AUDIO_INPUT_SAMPLE_RATE", "16000")
    monkeypatch.setenv("TTS_PROVIDER", "piper")

    assert validate_voice_provider_configuration() == {
        "llm": "local",
        "stt": "whisper",
        "tts": "piper",
    }


def test_local_llm_readiness_rejects_more_than_two_sessions(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen3-4b-local")
    monkeypatch.setenv("LOCAL_LLM_MAX_CONCURRENT_SESSIONS", "2")
    monkeypatch.setenv("VOICE_MAX_CONCURRENT_SESSIONS", "3")
    monkeypatch.setenv("STT_PROVIDER", "whisper")
    monkeypatch.setenv("AUDIO_INPUT_SAMPLE_RATE", "16000")
    monkeypatch.setenv("TTS_PROVIDER", "piper")

    with pytest.raises(ValueError, match="must be at most 2"):
        validate_voice_provider_configuration()


def test_local_llm_readiness_rejects_remote_server(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("VOICE_MAX_CONCURRENT_SESSIONS", "2")
    monkeypatch.setenv("STT_PROVIDER", "whisper")
    monkeypatch.setenv("AUDIO_INPUT_SAMPLE_RATE", "16000")
    monkeypatch.setenv("TTS_PROVIDER", "piper")

    with pytest.raises(ValueError, match="must point to localhost"):
        validate_voice_provider_configuration()
