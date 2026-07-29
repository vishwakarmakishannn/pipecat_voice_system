import os
import time
from loguru import logger
from pipecat.frames.frames import TTSAudioRawFrame, TTSStoppedFrame
from pipecat.services.cartesia.tts import CartesiaTTSService
from .config import get_text_aggregation_mode
from core.audio_config import audio_output_sample_rate


class InstrumentedCartesiaTTSService(CartesiaTTSService):
    """Expose WebSocket startup and per-context provider TTFB."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._request_started_at: dict[str, float] = {}
        self._first_audio_seen: set[str] = set()

    async def start(self, frame):
        started = time.monotonic()
        try:
            await super().start(frame)
        except BaseException:
            logger.warning(
                "voice_startup stage=provider_connected service=cartesia "
                "status=failed duration_ms={}",
                round((time.monotonic() - started) * 1000, 1),
            )
            raise
        logger.info(
            "voice_startup stage=provider_connected service=cartesia "
            "status=ready duration_ms={}",
            round((time.monotonic() - started) * 1000, 1),
        )

    async def run_tts(self, text: str, context_id: str):
        if context_id not in self._request_started_at:
            self._request_started_at[context_id] = time.monotonic()
            logger.info(
                "voice_tts stage=request_started provider=cartesia context={} chars={}",
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
                "voice_tts stage=provider_first_audio provider=cartesia "
                "context={} elapsed_ms={}",
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


def _max_buffer_delay_ms() -> int:
    # A small buffer lets Cartesia combine streamed LLM tokens into natural
    # phrases without waiting for a complete sentence. Zero remains available
    # for callers that implement their own phrase buffer.
    raw = os.getenv("CARTESIA_MAX_BUFFER_DELAY_MS", "150")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"CARTESIA_MAX_BUFFER_DELAY_MS must be an integer, got {raw!r}"
        ) from exc
    if not 0 <= value <= 5000:
        raise ValueError(
            f"CARTESIA_MAX_BUFFER_DELAY_MS must be between 0 and 5000, got {value}"
        )
    return value


def get_cartesia_tts():
    return InstrumentedCartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        sample_rate=audio_output_sample_rate("cartesia"),
        max_buffer_delay_ms=_max_buffer_delay_ms(),
        settings=InstrumentedCartesiaTTSService.Settings(
            model=os.getenv("CARTESIA_MODEL", "sonic-3"),
            voice=os.getenv("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"),
        ),
        text_aggregation_mode=get_text_aggregation_mode("cartesia"),
    )
