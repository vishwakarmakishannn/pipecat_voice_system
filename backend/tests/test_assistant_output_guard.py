from types import SimpleNamespace

import pytest
from pipecat.frames.frames import (
    FunctionCallInProgressFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from core.processors import AssistantOutputGuardProcessor


class DiagnosticRecorder:
    def __init__(self):
        self.events = []

    def record(self, **payload):
        self.events.append(payload)


@pytest.mark.anyio
async def test_normal_assistant_text_streams_to_the_normalization_stage(monkeypatch):
    frames = []
    processor = AssistantOutputGuardProcessor()

    async def capture(frame, _direction):
        frames.append(frame)

    monkeypatch.setattr(processor, "push_frame", capture)
    await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMTextFrame("Hello "), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMTextFrame("there."), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    text = "".join(frame.text for frame in frames if isinstance(frame, LLMTextFrame))
    assert text == "Hello there."


@pytest.mark.anyio
async def test_split_simulated_tool_markup_never_reaches_downstream(monkeypatch):
    frames = []
    recorder = DiagnosticRecorder()
    processor = AssistantOutputGuardProcessor(
        recovery_text="Please try again.",
        diagnostic_recorder=recorder,
    )

    async def capture(frame, _direction):
        frames.append(frame)

    monkeypatch.setattr(processor, "push_frame", capture)
    await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMTextFrame("Let me search. <fun"), FrameDirection.DOWNSTREAM)
    await processor.process_frame(
        LLMTextFrame('ction=tavily_search>{"query":"x"}</function>'),
        FrameDirection.DOWNSTREAM,
    )
    await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    text_frames = [frame for frame in frames if isinstance(frame, LLMTextFrame)]
    rendered = "".join(frame.text for frame in text_frames)
    assert rendered == "Let me search. Please try again."
    assert "function" not in rendered.lower()
    assert text_frames[-1].append_to_context is False
    assert text_frames[-1].invalid_output_recovery is True
    assert len(recorder.events) == 1
    assert recorder.events[0]["code"] == "llm.invalid_tool_markup"
    assert recorder.events[0]["details"]["rejected_characters"] > 0
    assert len(recorder.events[0]["details"]["rejected_sha256"]) == 64
    assert "query" not in str(recorder.events[0]["details"])


@pytest.mark.anyio
async def test_native_function_call_frames_pass_unchanged(monkeypatch):
    frames = []
    processor = AssistantOutputGuardProcessor()
    native = FunctionCallInProgressFrame("tavily_search", "tool-1", {"query": "x"})

    async def capture(frame, _direction):
        frames.append(frame)

    monkeypatch.setattr(processor, "push_frame", capture)
    await processor.process_frame(native, FrameDirection.DOWNSTREAM)

    assert frames == [native]
