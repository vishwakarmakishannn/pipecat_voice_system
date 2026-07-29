import asyncio

import pytest

from core.task_queue import BackgroundTaskQueue


def test_background_queue_rejects_overload_without_growing():
    queue = BackgroundTaskQueue(maxsize=1)

    async def work():
        return None

    assert queue.enqueue(work) is True
    assert queue.enqueue(work) is False
    assert queue.depth == 1
    assert queue.capacity == 1


@pytest.mark.anyio
async def test_background_queue_executes_bounded_work():
    queue = BackgroundTaskQueue(maxsize=2)
    completed = asyncio.Event()

    async def work():
        completed.set()

    queue.start(num_workers=1)
    assert queue.enqueue(work) is True
    await asyncio.wait_for(completed.wait(), timeout=0.2)
    await queue.stop()
    assert queue.depth == 0


@pytest.mark.anyio
async def test_enrichment_waits_for_voice_idle(monkeypatch):
    import core.realtime_gate as gate_module

    realtime_turn_gate = gate_module.RealtimeTurnGate()
    monkeypatch.setattr(gate_module, "realtime_turn_gate", realtime_turn_gate)

    queue = BackgroundTaskQueue(maxsize=2)
    completed = asyncio.Event()

    async def work():
        completed.set()

    realtime_turn_gate.begin("test-turn")
    queue.start(num_workers=1)
    assert queue.enqueue(work, enrichment=True) is True
    await asyncio.sleep(0.02)
    assert not completed.is_set()
    realtime_turn_gate.end("test-turn")
    await asyncio.wait_for(completed.wait(), timeout=0.2)
    await queue.stop()
