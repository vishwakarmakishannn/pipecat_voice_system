import asyncio
import json
import uuid

import pytest
from pipecat.frames.frames import LLMFullResponseEndFrame
from pipecat.processors.frame_processor import FrameDirection

from core.processors import CallTimelineProcessor
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


@pytest.mark.anyio
async def test_new_voice_turn_preempts_and_retries_running_enrichment(monkeypatch):
    import core.realtime_gate as gate_module

    realtime_turn_gate = gate_module.RealtimeTurnGate()
    monkeypatch.setattr(gate_module, "realtime_turn_gate", realtime_turn_gate)
    queue = BackgroundTaskQueue(maxsize=2)
    first_started = asyncio.Event()
    first_cancelled = asyncio.Event()
    second_started = asyncio.Event()
    allow_completion = asyncio.Event()
    attempts = 0

    async def work():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            first_started.set()
        else:
            second_started.set()
        try:
            await allow_completion.wait()
        except asyncio.CancelledError:
            first_cancelled.set()
            raise

    queue.start(num_workers=1)
    assert queue.enqueue(work, enrichment=True)
    await asyncio.wait_for(first_started.wait(), timeout=0.2)

    realtime_turn_gate.begin("live-turn")
    await asyncio.wait_for(first_cancelled.wait(), timeout=0.2)
    assert not second_started.is_set()

    realtime_turn_gate.end("live-turn")
    await asyncio.wait_for(second_started.wait(), timeout=0.2)
    allow_completion.set()
    await queue.stop()

    assert attempts == 2


@pytest.mark.anyio
async def test_wait_for_key_drains_only_matching_persistence():
    queue = BackgroundTaskQueue(maxsize=4)
    release_other = asyncio.Event()
    completed = []

    async def work(name, release=None):
        if release:
            await release.wait()
        completed.append(name)

    queue.start(num_workers=2)
    assert queue.enqueue(work, "call", key="call-1")
    assert queue.enqueue(work, "other", release_other, key="call-2")
    assert await queue.wait_for_key("call-1", timeout=0.2)
    assert "call" in completed
    assert "other" not in completed
    release_other.set()
    await queue.stop()


@pytest.mark.anyio
async def test_enrichment_lock_does_not_block_same_call_persistence(monkeypatch):
    import core.realtime_gate as gate_module

    monkeypatch.setattr(gate_module, "realtime_turn_gate", gate_module.RealtimeTurnGate())
    queue = BackgroundTaskQueue(maxsize=4)
    enrichment_started = asyncio.Event()
    release_enrichment = asyncio.Event()
    persistence_finished = asyncio.Event()

    async def enrichment():
        enrichment_started.set()
        await release_enrichment.wait()

    async def persistence():
        persistence_finished.set()

    queue.start(num_workers=1)
    assert queue.enqueue(enrichment, key="call-1", enrichment=True)
    await asyncio.wait_for(enrichment_started.wait(), timeout=0.2)
    assert queue.enqueue(persistence, key="call-1")
    await asyncio.wait_for(persistence_finished.wait(), timeout=0.2)
    release_enrichment.set()
    await queue.stop()


def test_queue_rejection_notifies_call_handler():
    queue = BackgroundTaskQueue(maxsize=1)
    rejected = []

    async def work():
        return None

    queue.register_rejection_handler(
        "call-1", lambda task_name, lane: rejected.append((task_name, lane))
    )
    assert queue.enqueue(work, key="call-1")
    assert not queue.enqueue(work, key="call-1")
    assert rejected == [("work", "persistence")]


@pytest.mark.anyio
async def test_invalid_invocation_is_rejected_before_it_can_block_call_drain():
    queue = BackgroundTaskQueue(maxsize=1)

    async def work(required_value):
        return required_value

    assert queue.enqueue(work, key="call-1") is False
    assert queue.depth == 0
    assert await queue.wait_for_key("call-1", timeout=0.01) is True


@pytest.mark.anyio
async def test_non_uuid_call_journal_is_rejected_before_write(monkeypatch, tmp_path):
    async def fake_operation(call_id, **_kwargs):
        return call_id

    fake_operation.__module__ = "services.calls"
    fake_operation.__name__ = "save_call_operation"
    fake_operation.__qualname__ = "save_call_operation"
    monkeypatch.setenv("PERSISTENCE_SPOOL_DIR", str(tmp_path))
    queue = BackgroundTaskQueue(maxsize=1)

    assert queue.enqueue(fake_operation, 1, key="1") is False
    assert queue.depth == 0
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.anyio
async def test_non_uuid_replayed_journal_is_quarantined(monkeypatch, tmp_path):
    import services.calls as calls

    called = False

    async def fake_operation(call_id, **_kwargs):
        nonlocal called
        called = True
        return call_id

    fake_operation.__module__ = "services.calls"
    fake_operation.__name__ = "save_call_operation"
    fake_operation.__qualname__ = "save_call_operation"
    monkeypatch.setattr(calls, "save_call_operation", fake_operation)
    monkeypatch.setenv("PERSISTENCE_SPOOL_DIR", str(tmp_path))
    journal_path = tmp_path / "test-invalid-call-id.json"
    journal_path.write_text(
        json.dumps({
            "version": 1,
            "id": "test-invalid-call-id",
            "module": "services.calls",
            "qualname": "save_call_operation",
            "args": [1],
            "kwargs": {},
            "key": "1",
        }),
        encoding="utf-8",
    )

    queue = BackgroundTaskQueue(maxsize=1)
    queue.start(num_workers=1)
    rejected_path = journal_path.with_suffix(".rejected")
    for _ in range(50):
        if rejected_path.exists():
            break
        await asyncio.sleep(0.01)
    await queue.stop()

    assert called is False
    assert journal_path.exists() is False
    assert rejected_path.exists() is True


@pytest.mark.anyio
async def test_durable_call_task_is_journaled_and_removed_after_success(
    monkeypatch, tmp_path
):
    import services.calls as calls

    completed = asyncio.Event()

    async def fake_summary(_call_id, _summary):
        completed.set()
        return True

    fake_summary.__module__ = "services.calls"
    fake_summary.__qualname__ = "save_call_summary"
    monkeypatch.setattr(calls, "save_call_summary", fake_summary)
    monkeypatch.setenv("PERSISTENCE_SPOOL_DIR", str(tmp_path))
    queue = BackgroundTaskQueue(maxsize=1)

    assert queue.enqueue(fake_summary, "call-id", "summary", key="call-id")
    assert len(list(tmp_path.glob("*.json"))) == 1
    queue.start(num_workers=1)
    await asyncio.wait_for(completed.wait(), timeout=0.3)
    assert await queue.wait_for_key("call-id", timeout=0.3)
    await queue.stop()

    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.anyio
async def test_durable_no_write_is_discarded_without_retrying(monkeypatch, tmp_path):
    import services.calls as calls

    attempts = 0

    async def declined_write(_call_id, _summary):
        nonlocal attempts
        attempts += 1
        return None

    declined_write.__module__ = "services.calls"
    declined_write.__qualname__ = "save_call_summary"
    monkeypatch.setattr(calls, "save_call_summary", declined_write)
    monkeypatch.setenv("PERSISTENCE_SPOOL_DIR", str(tmp_path))
    queue = BackgroundTaskQueue(maxsize=1)

    assert queue.enqueue(declined_write, "call-id", "summary", key="call-id")
    queue.start(num_workers=1)
    assert await queue.wait_for_key("call-id", timeout=0.3)
    await queue.stop()

    assert attempts == 1
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.anyio
async def test_empty_assistant_response_is_not_queued(monkeypatch):
    queued = []
    processor = CallTimelineProcessor("call-id", capture="assistant")

    async def capture(frame, direction):
        return None

    monkeypatch.setattr("core.task_queue.task_queue.enqueue", queued.append)
    monkeypatch.setattr(processor, "push_frame", capture)

    await processor.process_frame(
        LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM
    )

    assert queued == []


@pytest.mark.anyio
async def test_new_process_replays_an_unfinished_persistence_journal(
    monkeypatch, tmp_path
):
    import services.calls as calls

    completed = asyncio.Event()

    async def fake_summary(_call_id, _summary):
        completed.set()
        return True

    fake_summary.__module__ = "services.calls"
    fake_summary.__qualname__ = "save_call_summary"
    monkeypatch.setattr(calls, "save_call_summary", fake_summary)
    monkeypatch.setenv("PERSISTENCE_SPOOL_DIR", str(tmp_path))

    interrupted_process = BackgroundTaskQueue(maxsize=1)
    assert interrupted_process.enqueue(
        fake_summary, "call-id", "summary", key="call-id"
    )
    assert len(list(tmp_path.glob("*.json"))) == 1

    restarted_process = BackgroundTaskQueue(maxsize=1)
    restarted_process.start(num_workers=1)
    await asyncio.wait_for(completed.wait(), timeout=1)
    assert await restarted_process.wait_for_key("call-id", timeout=0.3)
    await restarted_process.stop()

    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.anyio
async def test_replay_repairs_legacy_rag_operation_missing_call_id(
    monkeypatch, tmp_path
):
    import services.calls as calls

    completed = asyncio.Event()
    observed = []
    call_id = uuid.uuid4()

    async def repaired_operation(
        received_call_id,
        *,
        operation_type,
        name,
        arguments,
        persistence_id=None,
    ):
        observed.append((received_call_id, operation_type, name, arguments, persistence_id))
        completed.set()
        return True

    repaired_operation.__module__ = "services.calls"
    repaired_operation.__name__ = "save_call_operation"
    repaired_operation.__qualname__ = "save_call_operation"
    monkeypatch.setattr(calls, "save_call_operation", repaired_operation)
    monkeypatch.setenv("PERSISTENCE_SPOOL_DIR", str(tmp_path))
    journal_id = "legacy-rag-operation"
    (tmp_path / f"{journal_id}.json").write_text(
        json.dumps({
            "version": 1,
            "id": journal_id,
            "module": "services.calls",
            "qualname": "save_call_operation",
            "args": [],
            "kwargs": {
                "operation_type": "rag",
                "name": "knowledge_retrieval",
                "arguments": {"query": "saved document"},
                "persistence_id": journal_id,
            },
            "key": str(call_id),
        }),
        encoding="utf-8",
    )

    queue = BackgroundTaskQueue(maxsize=1)
    queue.start(num_workers=1)
    await asyncio.wait_for(completed.wait(), timeout=1)
    assert await queue.wait_for_key(str(call_id), timeout=0.3)
    await queue.stop()

    assert observed == [(
        call_id,
        "rag",
        "knowledge_retrieval",
        {"query": "saved document"},
        journal_id,
    )]
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.anyio
async def test_durable_call_tasks_spill_instead_of_dropping_on_queue_full(
    monkeypatch, tmp_path
):
    import services.calls as calls

    completed = []

    async def fake_summary(_call_id, summary):
        completed.append(summary)
        return True

    fake_summary.__module__ = "services.calls"
    fake_summary.__qualname__ = "save_call_summary"
    monkeypatch.setattr(calls, "save_call_summary", fake_summary)
    monkeypatch.setenv("PERSISTENCE_SPOOL_DIR", str(tmp_path))
    queue = BackgroundTaskQueue(maxsize=1)

    assert queue.enqueue(fake_summary, "call-id", "first", key="call-id")
    assert queue.enqueue(fake_summary, "call-id", "second", key="call-id")
    queue.start(num_workers=1)
    assert await queue.wait_for_key("call-id", timeout=1)
    await queue.stop()

    assert completed == ["first", "second"]
    assert list(tmp_path.glob("*.json")) == []
