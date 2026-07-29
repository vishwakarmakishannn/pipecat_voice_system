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
    ):
        monkeypatch.delenv(name, raising=False)

    config = load_endpointing_config()

    assert config.vad_stop_secs == 0.20
    assert config.smart_turn_stop_secs == 0.3


def test_endpointing_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("SMART_TURN_STOP_SECS", "20")
    with pytest.raises(ValueError, match="SMART_TURN_STOP_SECS"):
        load_endpointing_config()
