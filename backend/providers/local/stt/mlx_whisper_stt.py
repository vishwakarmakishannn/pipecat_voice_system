"""VAD-segmented local Whisper STT accelerated by Apple MLX."""

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

from .mlx_config import (
    MLX_WHISPER_SAMPLE_RATE,
    MLXWhisperConfig,
    load_mlx_whisper_config,
)


Transcriber = Callable[[np.ndarray, MLXWhisperConfig], dict[str, Any]]
_MIN_AUDIO_PEAK = 1e-5
_NON_SPEECH_SEGMENTS = {
    "[blank_audio]",
    "[silence]",
    "(silence)",
}


def _engine_language(language: str) -> str | None:
    return None if language == "auto" else language


def _transcribe_with_mlx(
    audio: np.ndarray,
    config: MLXWhisperConfig,
) -> dict[str, Any]:
    try:
        import mlx_whisper
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MLX Whisper requires mlx-whisper. Run `uv sync` before "
            "setting STT_PROVIDER=mlxwhisper."
        ) from exc

    return mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=config.model,
        verbose=False,
        temperature=config.temperature,
        no_speech_threshold=config.no_speech_threshold,
        condition_on_previous_text=False,
        language=_engine_language(config.language),
    )


class MLXWhisperRuntime:
    """Own one cached MLX model and serialize inference on a worker thread."""

    def __init__(
        self,
        config: MLXWhisperConfig,
        *,
        transcriber: Transcriber = _transcribe_with_mlx,
    ):
        self.config = config
        self._transcriber = transcriber
        self._warmed = False
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="local-mlx-whisper-stt",
        )

    @property
    def warmed(self) -> bool:
        return self._warmed

    def _warm_sync(self) -> None:
        if self._closed:
            raise RuntimeError("MLX Whisper runtime is closed")
        if self._warmed:
            return
        started = time.perf_counter()
        self._transcriber(
            np.zeros(1600, dtype=np.float32),
            self.config,
        )
        self._warmed = True
        logger.info(
            "voice_startup stage=model_loaded service=mlx_whisper_stt "
            "model={} duration_ms={}",
            self.config.model,
            round((time.perf_counter() - started) * 1000, 1),
        )

    async def warm(self) -> None:
        """Download, load, and exercise the model before accepting traffic."""
        if self._closed:
            raise RuntimeError("MLX Whisper runtime is closed")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._warm_sync)

    async def transcribe(self, audio: np.ndarray) -> dict[str, Any]:
        """Queue one transcription without blocking the asyncio event loop."""
        if self._closed:
            raise RuntimeError("MLX Whisper runtime is closed")
        queued_at = time.perf_counter()
        audio_duration_ms = round(
            (audio.size / MLX_WHISPER_SAMPLE_RATE) * 1000,
            1,
        )

        def run() -> dict[str, Any]:
            started = time.perf_counter()
            queue_wait_ms = round((started - queued_at) * 1000, 1)
            result = self._transcriber(audio, self.config)
            self._warmed = True
            inference_ms = round((time.perf_counter() - started) * 1000, 1)
            real_time_factor = (
                round(inference_ms / audio_duration_ms, 3)
                if audio_duration_ms > 0
                else None
            )
            logger.info(
                "local_stt provider=mlxwhisper model={} audio_ms={} "
                "queue_wait_ms={} inference_ms={} real_time_factor={}",
                self.config.model,
                audio_duration_ms,
                queue_wait_ms,
                inference_ms,
                real_time_factor,
            )
            return result

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, run)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(
            self._executor.shutdown,
            wait=True,
            cancel_futures=True,
        )


_runtime: MLXWhisperRuntime | None = None
_runtime_lock = threading.Lock()


def get_mlx_whisper_runtime(
    config: MLXWhisperConfig | None = None,
) -> MLXWhisperRuntime:
    """Return the process-wide runtime for the selected MLX model."""
    global _runtime
    config = config or load_mlx_whisper_config()
    with _runtime_lock:
        if _runtime is None or _runtime._closed:
            _runtime = MLXWhisperRuntime(config)
        elif _runtime.config != config:
            raise RuntimeError(
                "MLX Whisper runtime is already initialized with different "
                "settings; restart the backend after changing MLX_WHISPER_*"
            )
        return _runtime


async def warm_mlx_whisper_runtime() -> None:
    await get_mlx_whisper_runtime().warm()


async def shutdown_mlx_whisper_runtime() -> None:
    global _runtime
    with _runtime_lock:
        runtime = _runtime
        _runtime = None
    if runtime is not None:
        await runtime.close()


class MLXWhisperSTTService(SegmentedSTTService):
    """Decode one complete VAD-delimited utterance with MLX Whisper."""

    def __init__(
        self,
        *,
        runtime: MLXWhisperRuntime,
        config: MLXWhisperConfig,
        **kwargs,
    ):
        super().__init__(
            sample_rate=MLX_WHISPER_SAMPLE_RATE,
            settings=STTSettings(
                model=config.model,
                language=_engine_language(config.language),
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
            result = await self._runtime.transcribe(pcm)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "local_stt provider=mlxwhisper model={} status=failed",
                self._runtime.config.model,
            )
            yield ErrorFrame(
                error="Local MLX Whisper transcription failed",
                fatal=True,
                processor=self,
                exception=exc,
            )
            return
        finally:
            await self.stop_processing_metrics()

        parts = []
        for segment in result.get("segments", []):
            part = str(segment.get("text", "")).strip()
            if part and part.lower() not in _NON_SPEECH_SEGMENTS:
                parts.append(part)
        if not parts:
            fallback = str(result.get("text", "")).strip()
            if fallback and fallback.lower() not in _NON_SPEECH_SEGMENTS:
                parts.append(fallback)
        text = " ".join(parts).strip()
        if text:
            yield TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                finalized=True,
            )


def get_mlx_whisper_stt() -> MLXWhisperSTTService:
    config = load_mlx_whisper_config()
    return MLXWhisperSTTService(
        runtime=get_mlx_whisper_runtime(config),
        config=config,
    )
