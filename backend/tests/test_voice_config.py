import pytest

from core.voice_config import load_endpointing_config


def test_endpointing_defaults_are_low_latency(monkeypatch):
    for name in (
        "VAD_CONFIDENCE",
        "VAD_START_SECS",
        "VAD_STOP_SECS",
        "VAD_MIN_VOLUME",
        "SMART_TURN_STOP_SECS",
        "SMART_TURN_PRE_SPEECH_MS",
        "SMART_TURN_MAX_DURATION_SECS",
        "TURN_STOP_STRATEGY",
        "SPEECH_TIMEOUT_SECS",
    ):
        monkeypatch.delenv(name, raising=False)

    config = load_endpointing_config()

    assert config.vad_stop_secs == 0.15
    assert config.smart_turn_stop_secs == 0.7
    assert config.turn_stop_strategy == "smart_turn"
    assert config.speech_timeout_secs == 0.2


def test_endpointing_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("SMART_TURN_STOP_SECS", "20")
    with pytest.raises(ValueError, match="SMART_TURN_STOP_SECS"):
        load_endpointing_config()


def test_endpointing_accepts_low_latency_speech_timeout(monkeypatch):
    monkeypatch.setenv("TURN_STOP_STRATEGY", "speech_timeout")
    monkeypatch.setenv("SPEECH_TIMEOUT_SECS", "0.12")

    config = load_endpointing_config()

    assert config.turn_stop_strategy == "speech_timeout"
    assert config.speech_timeout_secs == 0.12


def test_endpointing_rejects_unknown_turn_stop_strategy(monkeypatch):
    monkeypatch.setenv("TURN_STOP_STRATEGY", "regex")

    with pytest.raises(ValueError, match="TURN_STOP_STRATEGY"):
        load_endpointing_config()
