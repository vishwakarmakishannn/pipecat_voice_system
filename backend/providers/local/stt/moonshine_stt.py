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
    VADUserStoppedSpeakingFrame,
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
TranscriberFactory = Callable[[str | Path, Any, float, dict[str, str]], Any]


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
    options: dict[str, str],
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
        options=options,
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
                    {
                        "vad_window_duration": str(
                            self.config.vad_window_duration_seconds
                        ),
                        # Keep partial-line decoding deterministic across
                        # Moonshine releases; 0.1.3+ defaults this on.
                        "decode_incomplete_lines": "true",
                    },
                )
                logger.info(
                    "voice_startup stage=model_loaded service=moonshine_stt "
                    "model={} update_interval_seconds={} "
                    "vad_window_duration_seconds={} duration_ms={}",
                    self.config.model,
                    self.config.update_interval_seconds,
                    self.config.vad_window_duration_seconds,
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
    received_at: float
    error: Exception | None = None


@dataclass
class _FinalizationTiming:
    """Monotonic timing for one externally observed speech endpoint."""

    vad_received_at: float
    vad_frame_age_ms: float
    vad_silence_ms: float
    vad_forwarded_at: float | None = None
    delivery_released_at: float | None = None
    native_completed_at: float | None = None
    fallback_submitted_at: float | None = None
    fallback_started_at: float | None = None
    fallback_completed_at: float | None = None
    fallback_forced: bool = False

    @staticmethod
    def _delta_ms(start: float | None, end: float | None) -> float | None:
        if start is None or end is None:
            return None
        return round(max(0.0, end - start) * 1000, 1)

    def payload(
        self,
        *,
        callback_received_at: float | None = None,
        frame_built_at: float | None = None,
    ) -> dict[str, float]:
        values = {
            "vad_frame_age_ms": round(max(0.0, self.vad_frame_age_ms), 1),
            "vad_silence_ms": round(max(0.0, self.vad_silence_ms), 1),
            "vad_downstream_ms": self._delta_ms(
                self.vad_received_at, self.vad_forwarded_at
            ),
            "vad_delivery_gate_ms": self._delta_ms(
                self.vad_received_at, self.delivery_released_at
            ),
            "native_wait_ms": self._delta_ms(
                self.vad_received_at,
                callback_received_at or self.native_completed_at,
            ),
            "vad_frame_to_final_callback_ms": self._delta_ms(
                self.vad_received_at, callback_received_at
            ),
            "callback_delivery_ms": self._delta_ms(
                callback_received_at, frame_built_at
            ),
            "fallback_queue_ms": self._delta_ms(
                self.fallback_submitted_at, self.fallback_started_at
            ),
            "fallback_flush_ms": self._delta_ms(
                self.fallback_started_at, self.fallback_completed_at
            ),
            "fallback_forced": float(self.fallback_forced),
        }
        return {key: value for key, value in values.items() if value is not None}


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
        self._native_final_count = 0
        self._speech_start_final_count: int | None = None
        self._drain_lock = asyncio.Lock()
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._events_ready: asyncio.Event | None = None
        self._event_delivery_allowed: asyncio.Event | None = None
        self._event_pump_task: asyncio.Task | None = None
        self._last_interim: dict[int, str] = {}
        self._completed: set[int] = set()
        self._completed_order: deque[int] = deque()
        self._closed = False
        self._executor_closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="moonshine-stream",
        )
        self._finalization_timing: _FinalizationTiming | None = None
        self._finalization_event: asyncio.Event | None = None
        self._finalization_task: asyncio.Task | None = None
        self._endpoint_generation = 0

    def can_generate_metrics(self) -> bool:
        return True

    def _on_transcript_event(self, event: Any) -> None:
        received_at = time.perf_counter()
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
                received_at=received_at,
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
                received_at=received_at,
            )
        with self._events_lock:
            self._events.append(item)
            if item.kind == "final":
                self._native_final_count += 1
        loop = self._event_loop
        events_ready = self._events_ready
        if loop is not None and events_ready is not None:
            try:
                loop.call_soon_threadsafe(events_ready.set)
            except RuntimeError:
                # Shutdown may close the loop after the native listener has
                # captured it. The event remains queued for the final drain.
                pass

    def _ensure_stream_sync(self) -> None:
        if self._closed:
            raise RuntimeError("Moonshine STT service is closed")
        if self._stream is None:
            # Native line ids restart with each stream. Previous ids must not
            # suppress transcripts from the next independently finalized turn.
            self._last_interim.clear()
            self._completed.clear()
            self._completed_order.clear()
            self._stream = self._runtime.create_stream()
            self._stream.add_listener(self._on_transcript_event)
            self._stream.start()

    def _add_audio_sync(self, samples: list[float]) -> None:
        self._ensure_stream_sync()
        self._stream.add_audio(samples, MOONSHINE_SAMPLE_RATE)

    def _timed_fallback_flush_sync(self, timing: _FinalizationTiming) -> None:
        """Flush only after native line completion exceeded its safety budget.

        Normal VAD events never rotate the stream. By the time this fallback
        runs, Moonshine has received the configured grace period of real
        trailing audio, so a stalled stream can be recovered without making an
        early external VAD decision the destructive recognition boundary.
        """
        timing.fallback_started_at = time.perf_counter()
        try:
            self._finish_stream_sync()
        finally:
            timing.fallback_completed_at = time.perf_counter()

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

    async def _cancel_finalization_watchdog(self, *, clear_timing: bool) -> None:
        task = self._finalization_task
        self._finalization_task = None
        self._finalization_event = None
        self._endpoint_generation += 1
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if clear_timing:
            self._finalization_timing = None

    async def _await_native_finalization(
        self,
        generation: int,
        completion: asyncio.Event,
        timing: _FinalizationTiming,
    ) -> None:
        """Wait for the fast native final and force a flush only on a stall."""
        try:
            try:
                await asyncio.wait_for(
                    completion.wait(),
                    timeout=self._config.finalize_grace_seconds,
                )
                return
            except TimeoutError:
                if generation != self._endpoint_generation or self._closed:
                    return

            timing.fallback_forced = True
            timing.fallback_submitted_at = time.perf_counter()
            logger.warning(
                "local_stt provider=moonshine model={} "
                "stage=native_final_timeout grace_ms={}",
                self._config.model,
                round(self._config.finalize_grace_seconds * 1000, 1),
            )
            await self._run_native(self._timed_fallback_flush_sync, timing)
            await self._push_pending_frames()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "local_stt provider=moonshine model={} "
                "stage=fallback_flush status=failed",
                self._config.model,
            )
        finally:
            if generation == self._endpoint_generation:
                self._finalization_task = None
                self._finalization_event = None

    def _take_events(self) -> list[_TranscriptEvent]:
        with self._events_lock:
            events = list(self._events)
            self._events.clear()
        return events

    def _final_count(self) -> int:
        with self._events_lock:
            return self._native_final_count

    async def _pending_frames_unlocked(self) -> list[Frame]:
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
                self._completed_order.append(key)
                if len(self._completed_order) > 256:
                    self._completed.discard(self._completed_order.popleft())
                interim_text = self._last_interim.pop(key, "")
                timing = self._finalization_timing
                if timing is not None:
                    timing.native_completed_at = event.received_at
                completion = self._finalization_event
                if completion is not None:
                    completion.set()
                await self.stop_processing_metrics()
                frame_built_at = time.perf_counter()
                finalization_ms = (
                    timing.payload(
                        callback_received_at=event.received_at,
                        frame_built_at=frame_built_at,
                    )
                    if timing is not None
                    else None
                )
                if finalization_ms is not None:
                    finalization_ms["native_final_ms"] = float(event.latency_ms)
                    interim_words = len(interim_text.split())
                    final_words = len(event.text.split())
                    finalization_ms.update(
                        {
                            "latest_interim_chars": float(len(interim_text)),
                            "final_chars": float(len(event.text)),
                            "latest_interim_words": float(interim_words),
                            "final_words": float(final_words),
                            "final_shorter_than_interim": float(
                                bool(interim_text)
                                and (
                                    len(event.text) < len(interim_text)
                                    or final_words < interim_words
                                )
                            ),
                        }
                    )
                diagnostics_ms = finalization_ms or {
                    "callback_delivery_ms": round(
                        (frame_built_at - event.received_at) * 1000, 1
                    )
                }
                logger.info(
                    "local_stt provider=moonshine model={} final_latency_ms={} "
                    "finalization_ms={}",
                    self._config.model,
                    event.latency_ms,
                    diagnostics_ms,
                )
                result = {
                    "line_id": event.line_id,
                    "latency_ms": event.latency_ms,
                }
                if finalization_ms is not None:
                    result["finalization_ms"] = finalization_ms
                    self._finalization_timing = None
                frames.append(
                    TranscriptionFrame(
                        event.text,
                        self._user_id,
                        time_now_iso8601(),
                        self._settings.language,
                        result=result,
                        finalized=True,
                    )
                )
        return frames

    async def _pending_frames(self) -> list[Frame]:
        async with self._drain_lock:
            return await self._pending_frames_unlocked()

    async def _push_pending_frames(self) -> None:
        """Push native transcript events toward the user aggregator."""
        async with self._drain_lock:
            for frame in await self._pending_frames_unlocked():
                await self.push_frame(frame, FrameDirection.DOWNSTREAM)

    async def _run_event_pump(self) -> None:
        """Forward native callbacks without waiting for another audio frame."""
        events_ready = self._events_ready
        if events_ready is None:
            return
        while True:
            try:
                await events_ready.wait()
                events_ready.clear()
                delivery_allowed = self._event_delivery_allowed
                if delivery_allowed is not None:
                    await delivery_allowed.wait()
                await self._push_pending_frames()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "local_stt provider=moonshine model={} "
                    "status=event_pump_failed",
                    self._config.model,
                )

    async def _stop_event_pump(self) -> None:
        task = self._event_pump_task
        self._event_pump_task = None
        self._event_loop = None
        self._events_ready = None
        self._event_delivery_allowed = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._event_loop = asyncio.get_running_loop()
        self._events_ready = asyncio.Event()
        self._event_delivery_allowed = asyncio.Event()
        self._event_delivery_allowed.set()
        self._event_pump_task = asyncio.create_task(
            self._run_event_pump(),
            name=f"{self.name}-event-pump",
        )
        try:
            await self._run_native(self._ensure_stream_sync)
        except BaseException:
            await self._stop_event_pump()
            raise

    async def stop(self, frame: EndFrame):
        try:
            await self._cancel_finalization_watchdog(clear_timing=False)
            await self._run_native(self._finish_stream_sync)
            await self._push_pending_frames()
        finally:
            await self._stop_event_pump()
            await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        self._closed = True
        try:
            await self._cancel_finalization_watchdog(clear_timing=True)
            await self._run_native(self._close_stream_sync)
        finally:
            await self._stop_event_pump()
            await super().cancel(frame)

    async def cleanup(self):
        self._closed = True
        try:
            await self._cancel_finalization_watchdog(clear_timing=True)
            if not self._executor_closed:
                await self._run_native(self._close_stream_sync)
        finally:
            await self._stop_event_pump()
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
            if self._event_pump_task is not None:
                await self._push_pending_frames()
            else:
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
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._cancel_finalization_watchdog(clear_timing=True)
            final_count = self._final_count()
            if (
                self._speech_start_final_count is not None
                and final_count > self._speech_start_final_count
            ):
                # Moonshine may complete before an intentionally conservative
                # external VAD. Do not create a watchdog for a line that is
                # already final.
                self._speech_start_final_count = None
                await super().process_frame(frame, direction)
                logger.info(
                    "local_stt provider=moonshine model={} "
                    "stage=utterance_finalization status=already_complete",
                    self._config.model,
                )
                return
            received_at = time.perf_counter()
            timing = _FinalizationTiming(
                vad_received_at=received_at,
                vad_frame_age_ms=(
                    max(0.0, time.time() - frame.timestamp) * 1000
                    if frame.timestamp is not None
                    else 0.0
                ),
                vad_silence_ms=max(0.0, frame.stop_secs) * 1000,
            )
            self._finalization_timing = timing
            completion = asyncio.Event()
            self._finalization_event = completion
            generation = self._endpoint_generation
            delivery_allowed = self._event_delivery_allowed
            if delivery_allowed is not None:
                delivery_allowed.clear()
            self._finalization_task = asyncio.create_task(
                self._await_native_finalization(
                    generation,
                    completion,
                    timing,
                ),
                name=f"{self.name}-native-final-watchdog",
            )
            try:
                # VAD stop is a speech-state hint, not Moonshine's recognition
                # boundary. Keep the native stream alive so it receives the
                # remainder of its own VAD window and completes the line from
                # real trailing audio. The gate only preserves frame ordering.
                await super().process_frame(frame, direction)
                timing.vad_forwarded_at = time.perf_counter()
            except BaseException:
                await self._cancel_finalization_watchdog(clear_timing=True)
                raise
            finally:
                timing.delivery_released_at = time.perf_counter()
                if delivery_allowed is not None:
                    delivery_allowed.set()

            logger.info(
                "local_stt provider=moonshine model={} stage=utterance_finalization "
                "timing_ms={}",
                self._config.model,
                timing.payload(),
            )
            await self._push_pending_frames()
            return

        if isinstance(frame, VADUserStartedSpeakingFrame):
            # A new speech start before native completion means the external
            # VAD observed a brief pause. Cancel the pending safety flush and
            # let Moonshine keep the resumed speech in the same native line.
            await self._cancel_finalization_watchdog(clear_timing=True)
            self._speech_start_final_count = self._final_count()

        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self.start_processing_metrics()


def get_moonshine_stt() -> MoonshineSTTService:
    config = load_moonshine_config()
    return MoonshineSTTService(
        runtime=get_moonshine_runtime(config),
        config=config,
        audio_passthrough=True,
    )
