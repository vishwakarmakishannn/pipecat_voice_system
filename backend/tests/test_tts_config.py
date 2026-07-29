import pytest
from pipecat.services.tts_service import TextAggregationMode

from providers.tts.config import get_text_aggregation_mode


def test_remote_tts_defaults_to_token_streaming(monkeypatch):
    monkeypatch.delenv("TTS_TEXT_AGGREGATION_MODE", raising=False)
    assert get_text_aggregation_mode("deepgram") is TextAggregationMode.TOKEN
    assert get_text_aggregation_mode("cartesia") is TextAggregationMode.TOKEN


def test_local_piper_defaults_to_sentence_aggregation(monkeypatch):
    monkeypatch.delenv("TTS_TEXT_AGGREGATION_MODE", raising=False)
    assert get_text_aggregation_mode("piper") is TextAggregationMode.SENTENCE


def test_tts_aggregation_mode_can_be_overridden(monkeypatch):
    monkeypatch.setenv("TTS_TEXT_AGGREGATION_MODE", "sentence")
    assert get_text_aggregation_mode("deepgram") is TextAggregationMode.SENTENCE

    monkeypatch.setenv("TTS_TEXT_AGGREGATION_MODE", "invalid")
    with pytest.raises(ValueError, match="TTS_TEXT_AGGREGATION_MODE"):
        get_text_aggregation_mode("deepgram")
