import asyncio

import pytest
from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

import core.processors as processors
from core.processors import MemoryRetrievalProcessor


@pytest.mark.anyio
async def test_memory_processor_bypasses_non_recall_turn(monkeypatch):
    context = LLMContext(messages=[{"role": "user", "content": "Tell me a joke"}])
    processor = MemoryRetrievalProcessor(7, "call-1", context)
    delivered = []

    async def push(frame, direction):
        delivered.append((frame, direction))

    async def unexpected(*_args, **_kwargs):
        raise AssertionError("ordinary turns must not query long-term memory")

    monkeypatch.setattr(processor, "push_frame", push)
    monkeypatch.setattr(processors, "build_turn_memory_context", unexpected)

    frame = LLMContextFrame(context)
    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert delivered == [(frame, FrameDirection.DOWNSTREAM)]


@pytest.mark.anyio
async def test_memory_processor_injects_recall_context(monkeypatch):
    context = LLMContext(
        messages=[{"role": "user", "content": "What did I say in our previous call?"}]
    )
    processor = MemoryRetrievalProcessor(7, "call-1", context)
    delivered = []

    async def push(frame, direction):
        delivered.append((frame, direction))

    async def retrieve(user_id, query, *, current_call_id):
        assert (user_id, current_call_id) == (7, "call-1")
        assert "previous call" in query
        return "RELEVANT_MEMORY: The user prefers concise answers."

    monkeypatch.setattr(processor, "push_frame", push)
    monkeypatch.setattr(processors, "build_turn_memory_context", retrieve)

    frame = LLMContextFrame(context)
    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
    await asyncio.wait_for(processor._active_task, timeout=0.2)

    assert delivered == [(frame, FrameDirection.DOWNSTREAM)]
    assert any(
        "RELEVANT_MEMORY" in message.get("content", "")
        for message in context.messages
        if isinstance(message, dict)
    )
