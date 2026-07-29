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
