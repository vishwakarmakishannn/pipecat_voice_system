import pytest

from providers.tts.cartesia_tts import _max_buffer_delay_ms, get_cartesia_tts


def test_cartesia_latency_defaults_are_explicit(monkeypatch):
    monkeypatch.setenv("CARTESIA_API_KEY", "test")
    monkeypatch.delenv("CARTESIA_MODEL", raising=False)
    monkeypatch.delenv("CARTESIA_MAX_BUFFER_DELAY_MS", raising=False)

    service = get_cartesia_tts()

    assert service._settings.model == "sonic-3"
    assert service._max_buffer_delay_ms == 150


def test_cartesia_buffer_delay_is_validated(monkeypatch):
    monkeypatch.setenv("CARTESIA_MAX_BUFFER_DELAY_MS", "invalid")
    with pytest.raises(ValueError, match="must be an integer"):
        _max_buffer_delay_ms()

    monkeypatch.setenv("CARTESIA_MAX_BUFFER_DELAY_MS", "5001")
    with pytest.raises(ValueError, match="between 0 and 5000"):
        _max_buffer_delay_ms()
