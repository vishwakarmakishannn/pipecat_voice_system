from core.voice_config import startup_greeting


def test_startup_greeting_has_static_default(monkeypatch):
    monkeypatch.delenv("VOICE_GREETING_TEXT", raising=False)
    assert startup_greeting() == "Hello! How can I help?"


def test_startup_greeting_can_be_disabled(monkeypatch):
    monkeypatch.setenv("VOICE_GREETING_TEXT", "  ")
    assert startup_greeting() == ""
