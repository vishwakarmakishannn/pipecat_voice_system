"""True streaming local STT backed by Moonshine Voice."""

import asyncio
import threading
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601

from .moonshine_config import (
    MOONSHINE_SAMPLE_RATE,
    MoonshineConfig,
    load_moonshine_config,
)


ModelResolver = Callable[[MoonshineConfig], tuple[str | Path, Any]]
TranscriberFactory = Callable[[str | Path, Any, float], Any]


def _resolve_model(config: MoonshineConfig) -> tuple[str | Path, Any]:
    try:
        from moonshine_voice import ModelArch, get_model_for_language
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Moonshine STT requires moonshine-voice. Install backend "
            "dependencies before setting STT_PROVIDER=moonshine."
        ) from exc

    architecture = ModelArch.MEDIUM_STREAMING
    if config.model_dir is not None:
        return config.model_dir, architecture
    return get_model_for_language(
        config.language,
        architecture,
        cache_root=config.cache_dir,
    )


def _create_transcriber(
    model_path: str | Path,
    model_arch: Any,
    update_interval_seconds: float,
):
    try:
        from moonshine_voice import Transcriber
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Moonshine STT requires moonshine-voice. Install backend "
            "dependencies before setting STT_PROVIDER=moonshine."
        ) from exc
    return Transcriber(
        model_path,
        model_arch,
        update_interval=update_interval_seconds,
    )


class MoonshineRuntime:
    """Own one process-wide model while each call session owns a stream."""

    def __init__(
        self,
        config: MoonshineConfig,
        *,
        model_resolver: ModelResolver = _resolve_model,
        transcriber_factory: TranscriberFactory = _create_transcriber,
    ):
        self.config = config
        self._model_resolver = model_resolver
        self._transcriber_factory = transcriber_factory
        self._transcriber = None
        self._closed = False
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="moonshine-model",
        )

    def _ensure_transcriber(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("Moonshine runtime is closed")
            if self._transcriber is None:
                started = time.perf_counter()
                model_path, model_arch = self._model_resolver(self.config)
                self._transcriber = self._transcriber_factory(
                    model_path,
                    model_arch,
                    self.config.update_interval_seconds,
                )
                logger.info(
                    "voice_startup stage=model_loaded service=moonshine_stt "
                    "model={} duration_ms={}",
                    self.config.model,
                    round((time.perf_counter() - started) * 1000, 1),
                )
            return self._transcriber

    async def warm(self) -> None:
        """Download and load the selected model before reporting ready."""
        if self._closed:
            raise RuntimeError("Moonshine runtime is closed")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._ensure_transcriber)

    def create_stream(self):
        """Create a native stream that shares the process-wide transcriber."""
        with self._lock:
            transcriber = self._ensure_transcriber()
            return transcriber.create_stream(
                update_interval=self.config.update_interval_seconds
            )

    async def close(self) -> None:
        """Release the shared native transcriber and its loader thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            transcriber = self._transcriber
            self._transcriber = None

        try:
            if transcriber is not None:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._executor, transcriber.close)
        finally:
            await asyncio.to_thread(
                self._executor.shutdown,
                wait=True,
                cancel_futures=True,
            )


_runtime: MoonshineRuntime | None = None
_runtime_lock = threading.Lock()


def get_moonshine_runtime(
    config: MoonshineConfig | None = None,
) -> MoonshineRuntime:
    """Return the process-wide runtime for the configured 245M model."""
    global _runtime
    config = config or load_moonshine_config()
    with _runtime_lock:
        if _runtime is None or _runtime._closed:
            _runtime = MoonshineRuntime(config)
        elif _runtime.config != config:
            raise RuntimeError(
                "Moonshine runtime is already initialized with different "
                "settings; restart the backend after changing Moonshine "
                "environment variables"
            )
        return _runtime


async def warm_moonshine_runtime() -> None:
    await get_moonshine_runtime().warm()


async def shutdown_moonshine_runtime() -> None:
    global _runtime
    with _runtime_lock:
        runtime = _runtime
        _runtime = None
    if runtime is not None:
        await runtime.close()


@dataclass(frozen=True)
class _TranscriptEvent:
    kind: str
    line_id: int
    text: str
    latency_ms: int
    error: Exception | None = None


class MoonshineSTTService(STTService):
    """Feed live PCM to Moonshine and translate native stream events to Pipecat."""

    def __init__(
        self,
        *,
        runtime: MoonshineRuntime,
        config: MoonshineConfig,
        **kwargs,
    ):
        super().__init__(
            sample_rate=MOONSHINE_SAMPLE_RATE,
            settings=STTSettings(model=config.model, language=config.language),
            ttfs_p99_latency=config.ttfs_p99_latency_seconds,
            **kwargs,
        )
        self._runtime = runtime
        self._config = config
        self._stream = None
        self._events: deque[_TranscriptEvent] = deque()
        self._events_lock = threading.Lock()
        self._last_interim: dict[int, str] = {}
        self._completed: set[int] = set()
        self._closed = False
        self._executor_closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="moonshine-stream",
        )

    def can_generate_metrics(self) -> bool:
        return True

    def _on_transcript_event(self, event: Any) -> None:
        event_name = type(event).__name__
        if event_name == "Error":
            error = getattr(event, "error", None)
            if not isinstance(error, Exception):
                error = RuntimeError("Moonshine stream reported an error")
            item = _TranscriptEvent(
                kind="error",
                line_id=-1,
                text="",
                latency_ms=0,
                error=error,
            )
        else:
            line = getattr(event, "line", None)
            if line is None:
                return
            text = str(getattr(line, "text", "")).strip()
            if not text:
                return
            is_complete = bool(getattr(line, "is_complete", False))
            if event_name == "LineCompleted":
                kind = "final"
            elif event_name == "LineTextChanged" and not is_complete:
                kind = "interim"
            else:
                return
            item = _TranscriptEvent(
                kind=kind,
                line_id=int(getattr(line, "line_id", 0)),
                text=text,
                latency_ms=int(
                    getattr(line, "last_transcription_latency_ms", 0) or 0
                ),
            )
        with self._events_lock:
            self._events.append(item)

    def _ensure_stream_sync(self) -> None:
        if self._closed:
            raise RuntimeError("Moonshine STT service is closed")
        if self._stream is None:
            self._stream = self._runtime.create_stream()
            self._stream.add_listener(self._on_transcript_event)
            self._stream.start()

    def _add_audio_sync(self, samples: list[float]) -> None:
        self._ensure_stream_sync()
        self._stream.add_audio(samples, MOONSHINE_SAMPLE_RATE)

    def _finish_stream_sync(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.remove_listener(self._on_transcript_event)
        self._stream.close()
        self._stream = None

    def _close_stream_sync(self) -> None:
        if self._stream is None:
            return
        self._stream.remove_listener(self._on_transcript_event)
        self._stream.close()
        self._stream = None

    async def _run_native(self, function: Callable[..., Any], *args) -> Any:
        if self._executor_closed:
            raise RuntimeError("Moonshine STT service is closed")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, function, *args)

    def _take_events(self) -> list[_TranscriptEvent]:
        with self._events_lock:
            events = list(self._events)
            self._events.clear()
        return events

    async def _pending_frames(self) -> list[Frame]:
        frames: list[Frame] = []
        for event in self._take_events():
            key = event.line_id
            if event.kind == "error":
                await self.stop_all_metrics()
                frames.append(
                    ErrorFrame(
                        error="Local Moonshine transcription failed",
                        fatal=True,
                        processor=self,
                        exception=event.error,
                    )
                )
            elif event.kind == "interim":
                if key in self._completed or self._last_interim.get(key) == event.text:
                    continue
                self._last_interim[key] = event.text
                frames.append(
                    InterimTranscriptionFrame(
                        event.text,
                        self._user_id,
                        time_now_iso8601(),
                        self._settings.language,
                        result={
                            "line_id": event.line_id,
                            "latency_ms": event.latency_ms,
                        },
                    )
                )
            elif event.kind == "final":
                if key in self._completed:
                    continue
                self._completed.add(key)
                await self.stop_processing_metrics()
                logger.info(
                    "local_stt provider=moonshine model={} final_latency_ms={}",
                    self._config.model,
                    event.latency_ms,
                )
                frames.append(
                    TranscriptionFrame(
                        event.text,
                        self._user_id,
                        time_now_iso8601(),
                        self._settings.language,
                        result={
                            "line_id": event.line_id,
                            "latency_ms": event.latency_ms,
                        },
                        finalized=True,
                    )
                )
        return frames

    async def _push_pending_frames(self) -> None:
        """Push native transcript events toward the user aggregator."""
        for frame in await self._pending_frames():
            await self.push_frame(frame, FrameDirection.DOWNSTREAM)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._run_native(self._ensure_stream_sync)

    async def stop(self, frame: EndFrame):
        try:
            await self._run_native(self._finish_stream_sync)
            await self._push_pending_frames()
        finally:
            await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        self._closed = True
        try:
            await self._run_native(self._close_stream_sync)
        finally:
            await super().cancel(frame)

    async def cleanup(self):
        self._closed = True
        try:
            if not self._executor_closed:
                await self._run_native(self._close_stream_sync)
        finally:
            await super().cleanup()
            if not self._executor_closed:
                self._executor_closed = True
                await asyncio.to_thread(
                    self._executor.shutdown,
                    wait=True,
                    cancel_futures=True,
                )

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not audio:
            return
        try:
            samples = (
                np.frombuffer(audio, dtype=np.int16).astype(np.float32)
                / 32768.0
            ).tolist()
            await self._run_native(self._add_audio_sync, samples)
            for frame in await self._pending_frames():
                yield frame
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "local_stt provider=moonshine model={} status=failed",
                self._config.model,
            )
            await self.stop_all_metrics()
            yield ErrorFrame(
                error="Local Moonshine transcription failed",
                fatal=True,
                processor=self,
                exception=exc,
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self.start_processing_metrics()


def get_moonshine_stt() -> MoonshineSTTService:
    config = load_moonshine_config()
    return MoonshineSTTService(
        runtime=get_moonshine_runtime(config),
        config=config,
    )
