import pytest
from pipecat.frames.frames import FunctionCallInProgressFrame, FunctionCallResultFrame
from pipecat.processors.frame_processor import FrameDirection

from core.processors import ToolFillerProcessor
from core.task_queue import task_queue


@pytest.mark.anyio
async def test_server_queues_completed_tool_call_persistence(monkeypatch):
    queued = []
    processor = ToolFillerProcessor(call_id=7, enabled=False)

    async def capture(_frame, _direction):
        return None

    monkeypatch.setattr(processor, "push_frame", capture)
    monkeypatch.setattr(task_queue, "enqueue", lambda *args, **kwargs: queued.append((args, kwargs)))
    start = FunctionCallInProgressFrame("search", "call-1", {"q": "x"})
    frame = FunctionCallResultFrame("search", "call-1", {"q": "x"}, {"answer": "y"})

    await processor.process_frame(start, FrameDirection.DOWNSTREAM)
    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert len(queued) == 1
    args, kwargs = queued[0]
    assert args[1] == 7
    assert kwargs["operation_type"] == "tool"
    assert kwargs["name"] == "search"
    assert kwargs["request_id"] == "call-1"
    assert kwargs["result"] == {"answer": "y"}
    assert kwargs["key"] == "7"


@pytest.mark.anyio
async def test_failed_tool_is_counted_through_one_linked_diagnostic(monkeypatch):
    queued = []
    diagnostics = []

    class Recorder:
        def record(self, **payload):
            diagnostics.append(payload)

    processor = ToolFillerProcessor(
        call_id=7,
        enabled=False,
        diagnostic_recorder=Recorder(),
    )

    async def capture(_frame, _direction):
        return None

    monkeypatch.setattr(processor, "push_frame", capture)
    monkeypatch.setattr(
        task_queue,
        "enqueue",
        lambda *args, **kwargs: queued.append((args, kwargs)) or True,
    )
    arguments = {"query": "current news"}
    await processor.process_frame(
        FunctionCallInProgressFrame("tavily_search", "call-timeout", arguments),
        FrameDirection.DOWNSTREAM,
    )
    await processor.process_frame(
        FunctionCallResultFrame(
            "tavily_search",
            "call-timeout",
            arguments,
            {"status": "timeout"},
        ),
        FrameDirection.DOWNSTREAM,
    )

    assert queued[0][1]["error_code"] == "tool.execution_timeout"
    assert len(diagnostics) == 1
    assert diagnostics[0]["code"] == "tool.execution_timeout"
    assert diagnostics[0]["request_id"] == "call-timeout"


@pytest.mark.anyio
async def test_issue_operation_persistence_is_redacted_by_default(monkeypatch):
    queued = []
    processor = ToolFillerProcessor(call_id=7, enabled=False)

    async def capture(_frame, _direction):
        return None

    monkeypatch.setattr(processor, "push_frame", capture)
    monkeypatch.setattr(
        task_queue,
        "enqueue",
        lambda *args, **kwargs: queued.append((args, kwargs)) or True,
    )
    arguments = {
        "operation": "update",
        "email": "rohan22@example.com",
        "mobile": "9876543210",
    }
    result = {"status": "collecting_fields", "draft": dict(arguments)}
    await processor.process_frame(
        FunctionCallInProgressFrame(
            "manage_issue_draft",
            "private-call",
            arguments,
        ),
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

    persisted = queued[0][1]
    assert persisted["arguments"]["email"] == "r***@example.com"
    assert persisted["arguments"]["mobile"] == "***3210"
    assert persisted["result"]["draft"]["mobile"] == "***3210"
