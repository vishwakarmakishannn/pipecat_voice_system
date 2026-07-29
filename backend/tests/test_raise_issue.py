import importlib

import pytest
from pipecat.frames.frames import OutputTransportMessageFrame, TTSSpeakFrame

from core.processors import TurnLatencyState


raise_issue_module = importlib.import_module("tools.raise_issue")


@pytest.mark.anyio
async def test_raise_issue_uses_flush_without_post_commit_refresh(monkeypatch):
    events = []
    results = []
    frames = []

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def add(self, issue):
            self.issue = issue
            events.append("add")

        async def flush(self):
            self.issue.id = 42
            events.append("flush")

        async def commit(self):
            events.append("commit")

        async def refresh(self, _issue):
            raise AssertionError("post-commit refresh must not be used")

    class Worker:
        @staticmethod
        async def queue_frame(frame):
            frames.append(frame)

        @staticmethod
        async def queue_frames(items):
            frames.extend(items)

    class Params:
        function_name = "raise_issue"
        tool_call_id = "issue-call-1"
        arguments = {
            "cust_id": "C123456",
            "email": "person@example.com",
            "mobile": "9876543210",
            "device_id": "MSW12345678",
            "description": "Intermittent connection",
        }
        pipeline_worker = Worker()
        app_resources = {"latency_state": TurnLatencyState(session_id="test")}

        @staticmethod
        async def result_callback(result):
            results.append(result)

    monkeypatch.setattr(raise_issue_module, "VoiceSessionLocal", FakeSession)

    await raise_issue_module.raise_issue(
        Params(),
        cust_id="C123456",
        email="person@example.com",
        mobile="9876543210",
        device_id="MSW12345678",
        description="Intermittent connection",
    )

    assert events == ["add", "flush", "commit"]
    assert results == [{"status": "success", "message": "Issue #42 has been successfully raised."}]
    assert isinstance(frames[0], OutputTransportMessageFrame)
    assert frames[0].message["data"]["type"] == "assistant_transcript"
    assert isinstance(frames[1], TTSSpeakFrame)
    messages = [
        frame.message["data"]
        for frame in frames
        if isinstance(frame, OutputTransportMessageFrame)
        and frame.message["data"]["type"] == "tool_call"
    ]
    assert [message["payload"]["status"] for message in messages] == [
        "in_progress",
        "completed",
    ]
    assert messages[0]["payload"]["tool_call_id"] == "issue-call-1"
    assert messages[-1]["payload"]["result"] == results[-1]
