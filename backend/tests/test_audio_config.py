import pytest

from core.audio_config import (
    audio_input_sample_rate,
    audio_out_10ms_chunks,
    audio_output_sample_rate,
    trim_tts_leading_silence,
    tts_silence_preroll_ms,
    tts_silence_threshold,
)


def test_provider_aligned_sample_rate_defaults(monkeypatch):
    monkeypatch.delenv("AUDIO_INPUT_SAMPLE_RATE", raising=False)
    monkeypatch.delenv("AUDIO_OUTPUT_SAMPLE_RATE", raising=False)
    assert audio_input_sample_rate() == 16000
    assert audio_output_sample_rate("deepgram") == 24000
    assert audio_output_sample_rate("cartesia") == 24000
    assert audio_output_sample_rate("piper") == 22050


def test_invalid_sample_rate_is_rejected(monkeypatch):
    monkeypatch.setenv("AUDIO_INPUT_SAMPLE_RATE", "96000")
    with pytest.raises(ValueError, match="AUDIO_INPUT_SAMPLE_RATE"):
        audio_input_sample_rate()


def test_tts_silence_trim_defaults(monkeypatch):
    monkeypatch.delenv("TTS_TRIM_LEADING_SILENCE", raising=False)
    monkeypatch.delenv("TTS_SILENCE_THRESHOLD", raising=False)
    monkeypatch.delenv("TTS_SILENCE_PREROLL_MS", raising=False)
    assert trim_tts_leading_silence() is True
    assert tts_silence_threshold() == 128
    assert tts_silence_preroll_ms() == 20


def test_invalid_tts_silence_configuration_is_rejected(monkeypatch):
    monkeypatch.setenv("TTS_TRIM_LEADING_SILENCE", "sometimes")
    with pytest.raises(ValueError, match="TTS_TRIM_LEADING_SILENCE"):
        trim_tts_leading_silence()


def test_output_transport_defaults_to_one_10ms_chunk(monkeypatch):
    monkeypatch.delenv("AUDIO_OUT_10MS_CHUNKS", raising=False)
    assert audio_out_10ms_chunks() == 1


def test_invalid_output_chunk_count_is_rejected(monkeypatch):
    monkeypatch.setenv("AUDIO_OUT_10MS_CHUNKS", "0")
    with pytest.raises(ValueError, match="AUDIO_OUT_10MS_CHUNKS"):
        audio_out_10ms_chunks()
