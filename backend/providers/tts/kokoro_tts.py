"""Pipecat adapter for the process-wide local Kokoro runtime."""

from collections.abc import AsyncGenerator
from dataclasses import dataclass

from loguru import logger
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.kokoro.tts import language_to_kokoro_language
from pipecat.services.settings import TTSSettings, assert_given
from pipecat.services.tts_service import TTSService, TextAggregationMode
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

from core.audio_config import audio_output_sample_rate
from providers.tts.config import get_text_aggregation_mode
from providers.tts.kokoro_config import load_kokoro_config
from providers.tts.kokoro_runtime import KokoroRuntime, get_kokoro_runtime
from providers.tts.kokoro_text_aggregator import KokoroTextAggregator


@dataclass
class LocalKokoroTTSSettings(TTSSettings):
    """Runtime-updatable settings for the local Kokoro adapter."""


class LowLatencyKokoroTTSService(TTSService):
    """Per-call Pipecat state backed by one process-wide Kokoro model."""

    Settings = LocalKokoroTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        runtime: KokoroRuntime,
        low_latency_enabled: bool,
        first_chunk_chars: int,
        chunk_chars: int,
        min_chunk_words: int,
        **kwargs,
    ):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            **kwargs,
        )
        self._runtime = runtime
        self._resampler = create_stream_resampler()

        if (
            low_latency_enabled
            and self._text_aggregation_mode is TextAggregationMode.SENTENCE
        ):
            self._text_aggregator = KokoroTextAggregator(
                first_chunk_chars=first_chunk_chars,
                chunk_chars=chunk_chars,
                min_chunk_words=min_chunk_words,
            )

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str:
        return language_to_kokoro_language(language)

    @traced_tts
    async def run_tts(
        self,
        text: str,
        context_id: str,
    ) -> AsyncGenerator[Frame, None]:
        """Synthesize one eager phrase without blocking the asyncio loop."""
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            await self.start_tts_usage_metrics(text)
            voice = assert_given(self._settings.voice)
            language = assert_given(self._settings.language)
            if voice is None or language is None:
                raise ValueError("Kokoro voice and language are required")

            audio, source_sample_rate = await self._runtime.synthesize(
                text,
                voice=voice,
                language=language,
                speed=1.0,
            )
            await self.stop_ttfb_metrics()
            audio = await self._resampler.resample(
                audio,
                source_sample_rate,
                self.sample_rate,
            )
            if audio:
                yield TTSAudioRawFrame(
                    audio=audio,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
        except Exception as exc:
            logger.exception("Local Kokoro synthesis failed")
            yield ErrorFrame(error=f"Local Kokoro synthesis failed: {exc}")
        finally:
            await self.stop_ttfb_metrics()


def get_kokoro_tts() -> LowLatencyKokoroTTSService:
    config = load_kokoro_config()
    runtime = get_kokoro_runtime(config)
    aggregation_mode = get_text_aggregation_mode("kokoro")
    return LowLatencyKokoroTTSService(
        runtime=runtime,
        sample_rate=audio_output_sample_rate("kokoro"),
        settings=LowLatencyKokoroTTSService.Settings(
            model=config.model_id,
            voice=config.voice,
            language=config.language,
        ),
        text_aggregation_mode=aggregation_mode,
        low_latency_enabled=config.low_latency_enabled,
        first_chunk_chars=config.first_chunk_chars,
        chunk_chars=config.chunk_chars,
        min_chunk_words=config.min_chunk_words,
    )
