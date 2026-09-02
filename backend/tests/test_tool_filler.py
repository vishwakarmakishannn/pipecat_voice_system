import asyncio
import json

import pytest
from pipecat.frames.frames import FunctionCallInProgressFrame, FunctionCallResultFrame, OutputTransportMessageUrgentFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection

from core.processors import ToolFillerProcessor, TurnLatencyState


def test_tool_filler_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VOICE_TOOL_FILLER_ENABLED", raising=False)

    assert ToolFillerProcessor()._enabled is False


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
    transcript_frames = [frame for frame in frames if isinstance(frame, OutputTransportMessageUrgentFrame)]
    assert any(
        frame.message["data"]["type"] == "assistant_transcript"
        and frame.message["data"]["payload"]["text"] == "Let me look that up for you."
        for frame in transcript_frames
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "function_name",
    ["search_mswipe_knowledge", "manage_issue_draft", "get_current_datetime"],
)
async def test_immediate_filler_precedes_tool_transcription(
    monkeypatch,
    function_name,
):
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
        FunctionCallInProgressFrame(function_name, "ordered-call", {}),
        FrameDirection.DOWNSTREAM,
    )

    assert isinstance(frames[0], OutputTransportMessageUrgentFrame)
    assert frames[0].message["data"]["type"] == "assistant_transcript"
    assert frames[0].message["data"]["payload"]["text"] == "Let me look that up for you."
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
        if isinstance(frame, OutputTransportMessageUrgentFrame)
        and frame.message["data"]["type"] == "tool_call"
    ]
    assert [message["payload"]["status"] for message in messages] == ["in_progress", "completed"]
    assert messages[-1]["payload"]["result"] == {"answer": "done"}


@pytest.mark.anyio
async def test_timeout_lifecycle_is_sent_to_ui_as_completed_with_result(monkeypatch):
    frames = []
    processor = ToolFillerProcessor(delay_seconds=1, enabled=False)

    async def capture(frame, _direction):
        frames.append(frame)

    monkeypatch.setattr(processor, "push_frame", capture)
    arguments = {"query": "current news"}
    timeout_result = {
        "status": "timeout",
        "message": "Web search timed out. Continue without live results.",
    }

    await processor.process_frame(
        FunctionCallInProgressFrame("tavily_search", "timeout-call", arguments),
        FrameDirection.DOWNSTREAM,
    )
    await processor.process_frame(
        FunctionCallResultFrame(
            "tavily_search",
            "timeout-call",
            arguments,
            timeout_result,
        ),
        FrameDirection.DOWNSTREAM,
    )

    messages = [
        frame.message["data"]["payload"]
        for frame in frames
        if isinstance(frame, OutputTransportMessageUrgentFrame)
        and frame.message["data"]["type"] == "tool_call"
    ]
    assert [message["status"] for message in messages] == ["in_progress", "completed"]
    assert messages[-1]["result"] == timeout_result


@pytest.mark.anyio
async def test_issue_tool_lifecycle_redacts_pii_before_browser_event(monkeypatch):
    frames = []
    processor = ToolFillerProcessor(delay_seconds=1, enabled=False)

    async def capture(frame, _direction):
        frames.append(frame)

    monkeypatch.setattr(processor, "push_frame", capture)
    arguments = {
        "operation": "update",
        "cust_id": "C123456",
        "email": "rohan22@example.com",
        "mobile": "9876543210",
        "device_id": "MSW12345678",
        "description": "Payments are not being announced",
    }
    result = {"status": "collecting_fields", "draft": arguments}
    await processor.process_frame(
        FunctionCallInProgressFrame("manage_issue_draft", "private-call", arguments),
        FrameDirection.DOWNSTREAM,
    )
    await processor.process_frame(
        FunctionCallResultFrame(
            "manage_issue_draft",
            "private-call",
            arguments,
            result,
        ),
        FrameDirection.DOWNSTREAM,
    )

    browser_payload = json.dumps([
        frame.message
        for frame in frames
        if isinstance(frame, OutputTransportMessageUrgentFrame)
    ])
    for raw_value in arguments.values():
        if raw_value != "update":
            assert raw_value not in browser_payload
    assert "***3456" in browser_payload
    assert "r***@example.com" in browser_payload


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
