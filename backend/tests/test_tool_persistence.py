import json

import pytest
from pipecat.frames.frames import FunctionCallResultFrame
from pipecat.processors.frame_processor import FrameDirection

from core.processors import ConversationMemoryProcessor
from core.task_queue import task_queue


@pytest.mark.anyio
async def test_server_queues_completed_tool_call_persistence(monkeypatch):
    queued = []
    processor = ConversationMemoryProcessor(7, capture="assistant")

    async def capture(_frame, _direction):
        return None

    monkeypatch.setattr(processor, "push_frame", capture)
    monkeypatch.setattr(task_queue, "enqueue", lambda *args, **kwargs: queued.append((args, kwargs)))
    frame = FunctionCallResultFrame("search", "call-1", {"q": "x"}, {"answer": "y"})

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert len(queued) == 1
    args, kwargs = queued[0]
    assert args[1:3] == (7, "ToolCall")
    assert json.loads(args[3])["tool_call_id"] == "call-1"
    assert kwargs["key"] == 7
