import os
import time
from loguru import logger
from pipecat.frames.frames import TTSAudioRawFrame, TTSStoppedFrame
from pipecat.services.deepgram.tts import DeepgramTTSService
from .config import get_text_aggregation_mode
from core.audio_config import audio_output_sample_rate

class InstrumentedDeepgramTTSService(DeepgramTTSService):
    """Expose provider TTFB separately from downstream pipeline latency."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._request_started_at: dict[str, float] = {}
        self._first_audio_seen: set[str] = set()

    async def run_tts(self, text: str, context_id: str):
        if context_id not in self._request_started_at:
            self._request_started_at[context_id] = time.monotonic()
            logger.info(
                "voice_tts stage=request_started provider=deepgram context={} chars={}",
                context_id,
                len(text),
            )
        async for frame in super().run_tts(text, context_id):
            yield frame

    async def append_to_audio_context(self, context_id, frame):
        if isinstance(frame, TTSAudioRawFrame) and context_id not in self._first_audio_seen:
            self._first_audio_seen.add(context_id)
            started = self._request_started_at.get(context_id)
            logger.info(
                "voice_tts stage=provider_first_audio provider=deepgram context={} elapsed_ms={}",
                context_id,
                round((time.monotonic() - started) * 1000, 1) if started else None,
            )
        await super().append_to_audio_context(context_id, frame)
        if isinstance(frame, TTSStoppedFrame):
            self._request_started_at.pop(context_id, None)
            self._first_audio_seen.discard(context_id)

    async def on_audio_context_interrupted(self, context_id: str):
        self._request_started_at.pop(context_id, None)
        self._first_audio_seen.discard(context_id)
        await super().on_audio_context_interrupted(context_id)


def get_deepgram_tts():
    voice = os.getenv("DEEPGRAM_VOICE_ID", "aura-2-thalia-en")
    return InstrumentedDeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        sample_rate=audio_output_sample_rate("deepgram"),
        settings=DeepgramTTSService.Settings(voice=voice),
        text_aggregation_mode=get_text_aggregation_mode("deepgram"),
    )
