"""Bounded diagnostics for processor residence time and event-loop stalls."""

import asyncio
from collections import OrderedDict
import os
import time

from loguru import logger
from pipecat.frames.frames import (
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.observers.base_observer import BaseObserver, FrameProcessed, FramePushed


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}")
    return value


class PipelineLatencyObserver(BaseObserver):
    """Log frames whose time inside a processor exceeds a configured budget."""

    _IGNORED_FRAMES = (
        InputAudioRawFrame,
        TTSAudioRawFrame,
        LLMFullResponseEndFrame,
        TTSStoppedFrame,
    )

    def __init__(
        self,
        *,
        slow_frame_ms: float | None = None,
        max_inflight: int = 1024,
        warning_interval_seconds: float | None = None,
    ):
        super().__init__()
        self._slow_frame_ms = slow_frame_ms or _positive_float("VOICE_SLOW_FRAME_MS", 75.0)
        self._max_inflight = max_inflight
        self._warning_interval = warning_interval_seconds or _positive_float(
            "VOICE_SLOW_FRAME_WARNING_INTERVAL_SECONDS", 1.0
        )
        self._entries: OrderedDict[tuple[int, int], float] = OrderedDict()
        self._last_warning: dict[tuple[str, str], float] = {}

    async def on_process_frame(self, data: FrameProcessed):
        if isinstance(data.frame, self._IGNORED_FRAMES):
            return
        key = (id(data.processor), data.frame.id)
        self._entries[key] = time.perf_counter()
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_inflight:
            self._entries.popitem(last=False)

    async def on_push_frame(self, data: FramePushed):
        started = self._entries.pop((id(data.source), data.frame.id), None)
        if started is None:
            return
        duration_ms = (time.perf_counter() - started) * 1000
        if duration_ms >= self._slow_frame_ms:
            processor_name = getattr(data.source, "name", type(data.source).__name__)
            frame_name = type(data.frame).__name__
            warning_key = (processor_name, frame_name)
            now = time.monotonic()
            if now - self._last_warning.get(warning_key, 0.0) < self._warning_interval:
                return
            self._last_warning[warning_key] = now
            logger.warning(
                "voice_pipeline slow_processor={} frame={} residence_ms={:.1f}",
                processor_name,
                frame_name,
                duration_ms,
            )


class EventLoopLagMonitor:
    """Periodically report scheduler lag without blocking the voice loop."""

    def __init__(self, *, interval_seconds: float | None = None, warning_ms: float | None = None):
        self._interval = interval_seconds or _positive_float("VOICE_LOOP_LAG_INTERVAL_SECONDS", 0.1)
        self._warning_ms = warning_ms or _positive_float("VOICE_LOOP_LAG_WARNING_MS", 20.0)
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="voice-event-loop-lag")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    async def _run(self) -> None:
        deadline = asyncio.get_running_loop().time() + self._interval
        while True:
            await asyncio.sleep(max(0, deadline - asyncio.get_running_loop().time()))
            now = asyncio.get_running_loop().time()
            lag_ms = max(0.0, (now - deadline) * 1000)
            if lag_ms >= self._warning_ms:
                tasks = self._task_snapshot()
                logger.warning(
                    "voice_pipeline event_loop_lag_ms={:.1f} active_tasks={}",
                    lag_ms,
                    tasks,
                )
            deadline = now + self._interval

    def _task_snapshot(self, limit: int = 8) -> list[str]:
        """Return a cheap post-stall snapshot without formatting full tracebacks."""
        current = asyncio.current_task()
        details = []
        for task in asyncio.all_tasks():
            if task is current or task.done():
                continue
            frames = task.get_stack(limit=1)
            location = "waiting"
            if frames:
                frame = frames[-1]
                location = f"{frame.f_code.co_filename}:{frame.f_lineno}"
            details.append(f"{task.get_name()}@{location}")
            if len(details) >= limit:
                break
        return details


event_loop_lag_monitor = EventLoopLagMonitor()
