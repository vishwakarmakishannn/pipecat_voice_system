import asyncio

import pytest
from pipecat.frames.frames import FunctionCallInProgressFrame, FunctionCallResultFrame, OutputTransportMessageFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection

from core.processors import ToolFillerProcessor, TurnLatencyState


def test_tool_filler_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv("VOICE_TOOL_FILLER_ENABLED", raising=False)

    assert ToolFillerProcessor()._enabled is True


@pytest.mark.anyio
async def test_fast_tool_cancels_filler_before_it_speaks(monkeypatch):
    frames = []
    processor = ToolFillerProcessor(delay_seconds=0.02, enabled=True)

    async def capture(frame, _direction):
        frames.append(frame)

    monkeypatch.setattr(processor, "push_frame", capture)
    started = FunctionCallInProgressFrame("search", "1", {})
    result = FunctionCallResultFrame("search", "1", {}, {"ok": True})
    await processor.process_frame(started, FrameDirection.DOWNSTREAM)
    await processor.process_frame(result, FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.03)

    assert not any(isinstance(frame, TTSSpeakFrame) for frame in frames)


@pytest.mark.anyio
async def test_slow_tool_gets_one_delayed_filler(monkeypatch):
    frames = []
    processor = ToolFillerProcessor(delay_seconds=0.01, enabled=True)

    async def capture(frame, _direction):
        frames.append(frame)

    monkeypatch.setattr(processor, "push_frame", capture)
    await processor.process_frame(
        FunctionCallInProgressFrame("search", "1", {}),
        FrameDirection.DOWNSTREAM,
    )
    await asyncio.sleep(0.02)

    assert len([frame for frame in frames if isinstance(frame, TTSSpeakFrame)]) == 1
    transcript_frames = [frame for frame in frames if isinstance(frame, OutputTransportMessageFrame)]
    assert any(
        frame.message["data"]["type"] == "assistant_transcript"
        and frame.message["data"]["payload"]["text"] == "Let me check that."
        for frame in transcript_frames
    )


@pytest.mark.anyio
async def test_immediate_filler_precedes_tool_transcription(monkeypatch):
    frames = []
    processor = ToolFillerProcessor(
        latency_state=TurnLatencyState(session_id="test"),
        delay_seconds=0,
        enabled=True,
    )

    async def capture(frame, _direction):
        frames.append(frame)

    monkeypatch.setattr(processor, "push_frame", capture)
    await processor.process_frame(
        FunctionCallInProgressFrame("search", "ordered-call", {}),
        FrameDirection.DOWNSTREAM,
    )

    assert isinstance(frames[0], OutputTransportMessageFrame)
    assert frames[0].message["data"]["type"] == "assistant_transcript"
    assert isinstance(frames[1], TTSSpeakFrame)
    assert frames[2].message["data"]["type"] == "tool_call"


@pytest.mark.anyio
async def test_turn_scoped_filler_guard_prevents_provider_fallback_duplicate(monkeypatch):
    frames = []
    state = TurnLatencyState(session_id="test", tool_filler_spoken=True)
    processor = ToolFillerProcessor(
        latency_state=state,
        delay_seconds=0.01,
        enabled=True,
    )

    async def capture(frame, _direction):
        frames.append(frame)

    monkeypatch.setattr(processor, "push_frame", capture)
    await processor.process_frame(
        FunctionCallInProgressFrame("search", "provider-fallback", {}),
        FrameDirection.DOWNSTREAM,
    )
    await asyncio.sleep(0.02)

    assert not any(isinstance(frame, TTSSpeakFrame) for frame in frames)


@pytest.mark.anyio
async def test_tool_lifecycle_is_sent_to_ui_with_result(monkeypatch):
    frames = []
    processor = ToolFillerProcessor(delay_seconds=1, enabled=False)

    async def capture(frame, _direction):
        frames.append(frame)

    monkeypatch.setattr(processor, "push_frame", capture)
    started = FunctionCallInProgressFrame("search", "call-1", {"query": "news"})
    result = FunctionCallResultFrame("search", "call-1", {"query": "news"}, {"answer": "done"})
    await processor.process_frame(started, FrameDirection.DOWNSTREAM)
    await processor.process_frame(result, FrameDirection.DOWNSTREAM)

    messages = [
        frame.message["data"]
        for frame in frames
        if isinstance(frame, OutputTransportMessageFrame)
        and frame.message["data"]["type"] == "tool_call"
    ]
    assert [message["payload"]["status"] for message in messages] == ["in_progress", "completed"]
    assert messages[-1]["payload"]["result"] == {"answer": "done"}


@pytest.mark.anyio
async def test_default_configuration_never_queues_filler_ahead_of_answer(monkeypatch):
    frames = []
    state = TurnLatencyState(session_id="test")
    state.tool_used = True
    processor = ToolFillerProcessor(latency_state=state, delay_seconds=0.01, enabled=False)

    async def capture(frame, _direction):
        frames.append(frame)

    monkeypatch.setattr(processor, "push_frame", capture)
    await processor.process_frame(
        FunctionCallInProgressFrame("search", "1", {}),
        FrameDirection.DOWNSTREAM,
    )
    await asyncio.sleep(0.02)

    assert not any(isinstance(frame, TTSSpeakFrame) for frame in frames)
