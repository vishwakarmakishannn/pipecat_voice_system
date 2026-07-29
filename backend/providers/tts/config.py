import os

from pipecat.services.tts_service import TextAggregationMode


def get_text_aggregation_mode(provider: str) -> TextAggregationMode:
    """Select token streaming for remote streaming TTS and sentence mode for Piper."""
    configured = os.getenv("TTS_TEXT_AGGREGATION_MODE", "auto").strip().lower()
    if configured == "auto":
        return TextAggregationMode.SENTENCE if provider == "piper" else TextAggregationMode.TOKEN
    try:
        return TextAggregationMode(configured)
    except ValueError as exc:
        supported = "auto, token, sentence"
        raise ValueError(
            f"TTS_TEXT_AGGREGATION_MODE must be one of {supported}, got {configured!r}"
        ) from exc
