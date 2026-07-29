from types import SimpleNamespace

import pytest
from pipecat.frames.frames import InputAudioRawFrame, LLMFullResponseEndFrame, TTSAudioRawFrame, TTSStoppedFrame

from core.latency_observer import EventLoopLagMonitor, PipelineLatencyObserver


@pytest.mark.anyio
async def test_pipeline_latency_observer_is_bounded_and_clears_completed_frame(monkeypatch):
    observer = PipelineLatencyObserver(slow_frame_ms=1, max_inflight=2)
    processor = SimpleNamespace(name="processor")
    frames = [SimpleNamespace(id=value) for value in range(3)]

    for frame in frames:
        await observer.on_process_frame(SimpleNamespace(processor=processor, frame=frame))

    assert len(observer._entries) == 2
    await observer.on_push_frame(SimpleNamespace(source=processor, frame=frames[-1]))
    assert len(observer._entries) == 1


@pytest.mark.anyio
async def test_event_loop_lag_monitor_starts_once_and_stops():
    monitor = EventLoopLagMonitor(interval_seconds=0.001, warning_ms=1000)
    monitor.start()
    first_task = monitor._task
    monitor.start()

    assert monitor._task is first_task
    await monitor.stop()
    assert monitor._task is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "frame",
    [
        InputAudioRawFrame(b"\x00\x00", 16000, 1),
        TTSAudioRawFrame(b"\x00\x00", 24000, 1),
        LLMFullResponseEndFrame(),
        TTSStoppedFrame(),
    ],
)
async def test_pipeline_latency_observer_ignores_high_volume_and_drain_frames(
    monkeypatch, frame
):
    observer = PipelineLatencyObserver(slow_frame_ms=1)
    processor = SimpleNamespace(name="audio processor")
    warnings = []
    timestamps = iter([1.0, 2.0])
    monkeypatch.setattr("core.latency_observer.time.perf_counter", lambda: next(timestamps))
    monkeypatch.setattr("core.latency_observer.logger.warning", warnings.append)

    await observer.on_process_frame(SimpleNamespace(processor=processor, frame=frame))
    await observer.on_push_frame(SimpleNamespace(source=processor, frame=frame))

    assert warnings == []


@pytest.mark.anyio
async def test_pipeline_latency_observer_rate_limits_repeated_warning(monkeypatch):
    observer = PipelineLatencyObserver(
        slow_frame_ms=1,
        warning_interval_seconds=10,
    )
    processor = SimpleNamespace(name="processor")
    frames = [SimpleNamespace(id=value) for value in range(2)]
    warnings = []
    timestamps = iter([1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr("core.latency_observer.time.perf_counter", lambda: next(timestamps))
    monkeypatch.setattr("core.latency_observer.logger.warning", lambda *args: warnings.append(args))

    for frame in frames:
        await observer.on_process_frame(SimpleNamespace(processor=processor, frame=frame))
        await observer.on_push_frame(SimpleNamespace(source=processor, frame=frame))

    assert len(warnings) == 1
