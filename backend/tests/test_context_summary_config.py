import pytest

from core.context_summary_config import load_voice_context_summary_config


SUMMARY_ENV_NAMES = (
    "VOICE_CONTEXT_SUMMARIZATION_ENABLED",
    "VOICE_CONTEXT_SUMMARY_MAX_TOKENS",
    "VOICE_CONTEXT_SUMMARY_MAX_MESSAGES",
    "VOICE_CONTEXT_SUMMARY_TARGET_TOKENS",
    "VOICE_CONTEXT_SUMMARY_KEEP_MESSAGES",
    "VOICE_CONTEXT_SUMMARY_TIMEOUT_SECONDS",
    "VOICE_CONTEXT_SUMMARY_RETRY_COOLDOWN_SECONDS",
    "VOICE_CONTEXT_EMERGENCY_MAX_MESSAGES",
    "VOICE_CONTEXT_EMERGENCY_MAX_CHARS",
    "GROQ_CONTEXT_SUMMARY_MODEL",
    "GROQ_MEMORY_MODEL",
)


def test_context_summary_defaults_are_enabled_and_balanced(monkeypatch):
    for name in SUMMARY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)

    config = load_voice_context_summary_config()

    assert config.enabled is True
    assert (config.max_tokens, config.max_messages) == (3000, 20)
    assert (config.target_tokens, config.keep_messages) == (900, 8)
    assert (config.emergency_max_messages, config.emergency_max_chars) == (
        40,
        24000,
    )
    assert config.model == "llama-3.1-8b-instant"


def test_context_summary_model_falls_back_to_memory_model(monkeypatch):
    monkeypatch.delenv("GROQ_CONTEXT_SUMMARY_MODEL", raising=False)
    monkeypatch.setenv("GROQ_MEMORY_MODEL", "memory-model")

    assert load_voice_context_summary_config().model == "memory-model"


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("VOICE_CONTEXT_SUMMARY_MAX_TOKENS", "0", "MAX_TOKENS"),
        ("VOICE_CONTEXT_SUMMARY_TARGET_TOKENS", "3000", "TARGET_TOKENS"),
        ("VOICE_CONTEXT_SUMMARY_KEEP_MESSAGES", "20", "KEEP_MESSAGES"),
        ("VOICE_CONTEXT_SUMMARY_TIMEOUT_SECONDS", "31", "TIMEOUT_SECONDS"),
        ("VOICE_CONTEXT_EMERGENCY_MAX_MESSAGES", "20", "EMERGENCY_MAX_MESSAGES"),
        ("VOICE_CONTEXT_EMERGENCY_MAX_CHARS", "12000", "EMERGENCY_MAX_CHARS"),
    ],
)
def test_context_summary_rejects_unsafe_configuration(
    monkeypatch, name, value, match
):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=match):
        load_voice_context_summary_config()
