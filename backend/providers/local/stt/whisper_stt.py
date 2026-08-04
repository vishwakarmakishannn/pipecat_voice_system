"""VAD-segmented local Whisper STT backed by whisper.cpp."""

import asyncio
import threading
import time
from collections.abc import AsyncGenerator, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import WHISPER_TTFS_P99
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601

from .config import WHISPER_SAMPLE_RATE, WhisperConfig, load_whisper_config


ModelFactory = Callable[[WhisperConfig], Any]
_MIN_AUDIO_PEAK = 1e-5
_NON_SPEECH_SEGMENTS = {
    "[blank_audio]",
    "[silence]",
    "(silence)",
}


def _engine_language(language: str) -> str:
    """Translate the public auto value to whisper.cpp's auto-detect sentinel."""
    return "" if language == "auto" else language


def _create_model(config: WhisperConfig):
    try:
        from pywhispercpp.model import Model
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Local Whisper requires pywhispercpp. Install backend dependencies "
            "before setting STT_PROVIDER=whisper."
        ) from exc

    kwargs: dict[str, Any] = {
        "n_threads": config.threads,
        "language": _engine_language(config.language),
        "print_realtime": False,
        "print_progress": False,
        "redirect_whispercpp_logs_to": None,
    }
    if config.models_dir is not None:
        kwargs["models_dir"] = str(config.models_dir)
    return Model(config.model, **kwargs)


class WhisperRuntime:
    """Own one Whisper context and serialize all inference on its worker thread."""

    def __init__(
        self,
        config: WhisperConfig,
        *,
        model_factory: ModelFactory = _create_model,
    ):
        self.config = config
        self._model_factory = model_factory
        self._model = None
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="local-whisper-stt",
        )

    def _ensure_model(self):
        if self._closed:
            raise RuntimeError("Whisper runtime is closed")
        if self._model is None:
            started = time.perf_counter()
            self._model = self._model_factory(self.config)
            logger.info(
                "voice_startup stage=model_loaded service=whisper_stt "
                "model={} duration_ms={}",
                self.config.model,
                round((time.perf_counter() - started) * 1000, 1),
            )
        return self._model

    async def warm(self) -> None:
        """Load or download the configured model before the server is ready."""
        if self._closed:
            raise RuntimeError("Whisper runtime is closed")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._ensure_model)

    async def transcribe(self, audio: np.ndarray) -> list[Any]:
        """Queue one transcription without blocking the asyncio event loop."""
        if self._closed:
            raise RuntimeError("Whisper runtime is closed")
        queued_at = time.perf_counter()
        audio_duration_ms = round(
            (audio.size / WHISPER_SAMPLE_RATE) * 1000,
            1,
        )

        def run() -> list[Any]:
            started = time.perf_counter()
            queue_wait_ms = round((started - queued_at) * 1000, 1)
            model = self._ensure_model()
            segments = model.transcribe(audio)
            inference_ms = round((time.perf_counter() - started) * 1000, 1)
            real_time_factor = (
                round(inference_ms / audio_duration_ms, 3)
                if audio_duration_ms > 0
                else None
            )
            logger.info(
                "local_stt provider=whisper model={} audio_ms={} "
                "queue_wait_ms={} inference_ms={} real_time_factor={}",
                self.config.model,
                audio_duration_ms,
                queue_wait_ms,
                inference_ms,
                real_time_factor,
            )
            return list(segments)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, run)

    async def close(self) -> None:
        """Stop accepting inference and release the dedicated worker."""
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(
            self._executor.shutdown,
            wait=True,
            cancel_futures=True,
        )
        self._model = None


_runtime: WhisperRuntime | None = None
_runtime_lock = threading.Lock()


def get_whisper_runtime(config: WhisperConfig | None = None) -> WhisperRuntime:
    """Return the one process-wide runtime for the selected configuration."""
    global _runtime
    config = config or load_whisper_config()
    with _runtime_lock:
        if _runtime is None or _runtime._closed:
            _runtime = WhisperRuntime(config)
        elif _runtime.config != config:
            raise RuntimeError(
                "Whisper runtime is already initialized with different settings; "
                "restart the backend after changing Whisper environment variables"
            )
        return _runtime


async def warm_whisper_runtime() -> None:
    await get_whisper_runtime().warm()


async def shutdown_whisper_runtime() -> None:
    global _runtime
    with _runtime_lock:
        runtime = _runtime
        _runtime = None
    if runtime is not None:
        await runtime.close()


class WhisperSTTService(SegmentedSTTService):
    """Decode one complete VAD-delimited utterance with local Whisper."""

    def __init__(
        self,
        *,
        runtime: WhisperRuntime,
        config: WhisperConfig,
        **kwargs,
    ):
        super().__init__(
            sample_rate=WHISPER_SAMPLE_RATE,
            settings=STTSettings(
                model=config.model,
                language=None if config.language == "auto" else config.language,
            ),
            ttfs_p99_latency=WHISPER_TTFS_P99,
            **kwargs,
        )
        self._runtime = runtime

    @property
    def wants_wav_segments(self) -> bool:
        return False

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not audio:
            return

        await self.start_processing_metrics()
        try:
            pcm = (
                np.frombuffer(audio, dtype=np.int16).astype(np.float32)
                / 32768.0
            )
            if pcm.size == 0 or float(np.max(np.abs(pcm))) < _MIN_AUDIO_PEAK:
                return
            segments = await self._runtime.transcribe(pcm)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "local_stt provider=whisper model={} status=failed",
                self._runtime.config.model,
            )
            yield ErrorFrame(
                error="Local Whisper transcription failed",
                fatal=True,
                processor=self,
                exception=exc,
            )
            return
        finally:
            await self.stop_processing_metrics()

        parts = []
        for segment in segments:
            part = getattr(segment, "text", "").strip()
            if part and part.lower() not in _NON_SPEECH_SEGMENTS:
                parts.append(part)
        text = " ".join(parts).strip()
        if text:
            yield TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                finalized=True,
            )


def get_whisper_stt() -> WhisperSTTService:
    config = load_whisper_config()
    return WhisperSTTService(
        runtime=get_whisper_runtime(config),
        config=config,
    )
