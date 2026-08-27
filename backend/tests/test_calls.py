import uuid
import asyncio
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError

from core.database import AsyncSessionLocal, engine
from core.models import Call, CallEvent, CallOperation, CallRecording, CallTurn, TranscriptEntry, User, UserMemory
from core.task_queue import task_queue
from services.calls import (
    finalize_call,
    fail_nonterminal_runner_call,
    abandon_stale_calls,
    mark_call_active,
    restore_call,
    save_call_event,
    save_call_operation,
    save_call_turn,
    save_transcript_entry,
    snapshot_call_configuration,
    soft_delete_call,
    utcnow,
)
from services.recordings import (
    CallAudioBufferProcessor,
    CallRecordingWriter,
    _encode_pcm_to_mp3,
    recover_unfinished_recordings,
)
import services.recordings as recording_service
from services.call_maintenance import purge_expired_calls
from services.memory import _load_active_facts


pytestmark = pytest.mark.database


@pytest.fixture(autouse=True)
async def fresh_engine_pool():
    await engine.dispose()
    yield
    await engine.dispose()


async def _create_call() -> tuple[int, uuid.UUID]:
    async with AsyncSessionLocal() as db:
        user = User(username=f"voice2-{uuid.uuid4().hex}", password_hash="test")
        db.add(user)
        await db.flush()
        call = Call(user_id=user.id, title="Test call")
        db.add(call)
        await db.commit()
        return user.id, call.id


async def _delete_user(user_id: int) -> None:
    async with AsyncSessionLocal() as db:
        user = await db.get(User, user_id)
        if user:
            await db.execute(text("SET LOCAL aura.allow_call_purge = 'on'"))
            await db.delete(user)
            await db.commit()


@pytest.mark.anyio
async def test_call_lifecycle_finalizes_once_and_rejects_terminal_writes(monkeypatch):
    user_id, call_id = await _create_call()
    monkeypatch.setattr(task_queue, "enqueue", lambda *_args, **_kwargs: True)
    try:
        assert await mark_call_active(call_id) is True
        assert await save_transcript_entry(call_id, "You", "Hello", source="stt_final", turn_id=1)
        assert await save_call_operation(
            call_id,
            operation_type="tool",
            name="test_tool",
            arguments={"value": 1},
            result={"status": "ok"},
            turn_id=1,
        )
        assert await save_call_event(
            call_id,
            component="llm",
            code="llm.test_warning",
            severity="warning",
            outcome="recovered",
            safe_message="Recovered test event",
            turn_id=1,
            recovered=True,
        )
        assert await save_call_turn(
            call_id,
            {"turn_id": 1, "latency_complete": True, "answer_audio_ms": 420.0},
        )
        assert await finalize_call(call_id, status="completed") is True
        assert await finalize_call(call_id, status="failed") is False
        assert await save_transcript_entry(call_id, "You", "Too late", source="typed_user") is None
        assert await save_call_operation(
            call_id, operation_type="tool", name="late", arguments={}
        ) is None
        assert await save_call_turn(call_id, {"turn_id": 2}) is None
        assert await save_call_event(
            call_id,
            component="llm",
            code="llm.late",
            severity="error",
            outcome="failed",
            safe_message="Late event",
        ) is None

        async with AsyncSessionLocal() as db:
            finalized = await db.get(Call, call_id)
            assert finalized.status == "completed"
            assert finalized.summary == "User: Hello"
            assert finalized.avg_latency_ms == 420.0
            assert finalized.p50_latency_ms == 420.0
            assert finalized.p90_latency_ms == 420.0

        async with AsyncSessionLocal() as db:
            db.add(TranscriptEntry(
                call_id=call_id,
                sequence=999,
                speaker="You",
                source="typed_user",
                text="database trigger must reject this",
            ))
            with pytest.raises(DBAPIError):
                await db.commit()
            await db.rollback()

        terminal_children = (
            CallTurn(call_id=call_id, sequence=999),
            CallOperation(
                call_id=call_id,
                sequence=999,
                operation_type="tool",
                name="late",
            ),
            CallEvent(
                call_id=call_id,
                sequence=999,
                component="pipeline",
                code="pipeline.late",
                severity="error",
                outcome="failed",
                safe_message="late",
                fingerprint=uuid.uuid4().hex,
            ),
        )
        for child in terminal_children:
            async with AsyncSessionLocal() as db:
                db.add(child)
                with pytest.raises(DBAPIError):
                    await db.commit()
                await db.rollback()

        async with AsyncSessionLocal() as db:
            call = await db.get(Call, call_id)
            call.status = "active"
            with pytest.raises(DBAPIError):
                await db.commit()
            await db.rollback()

        async with AsyncSessionLocal() as db:
            call = await db.get(Call, call_id)
            call.title = "mutated terminal title"
            with pytest.raises(DBAPIError):
                await db.commit()
            await db.rollback()
    finally:
        await _delete_user(user_id)


@pytest.mark.anyio
async def test_event_deduplication_and_recursive_secret_redaction():
    user_id, call_id = await _create_call()
    try:
        first = await save_call_event(
            call_id,
            component="llm",
            code="llm.first_output_timeout",
            severity="error",
            outcome="recovered",
            safe_message="Provider timed out token=top-secret",
            details={"headers": {"Authorization": "Bearer secret"}, "llm_input_tokens": 42},
            request_id="req-1",
            recovered=True,
        )
        duplicate = await save_call_event(
            call_id,
            component="llm",
            code="llm.first_output_timeout",
            severity="error",
            outcome="recovered",
            safe_message="A differently worded duplicate for the same request",
            details={"api_key": "secret"},
            request_id="req-1",
            recovered=True,
        )
        operation = await save_call_operation(
            call_id,
            operation_type="tool",
            name="search",
            arguments={"query": "safe", "headers": {"Authorization": "Bearer secret"}},
            result={"answer": "safe"},
        )
        assert first.id == duplicate.id
        assert "top-secret" not in first.safe_message
        assert first.details["headers"] == "[REDACTED]"
        assert first.details["llm_input_tokens"] == 42
        assert operation.arguments["headers"] == "[REDACTED]"
        credential_event = await save_call_event(
            call_id,
            component="llm",
            code="llm.credential_test",
            severity="warning",
            outcome="degraded",
            safe_message="Bearer secret-value-that-must-not-leak",
        )
        assert "secret-value" not in credential_event.safe_message
        async with AsyncSessionLocal() as db:
            call = await db.get(Call, call_id)
            count = len((await db.execute(select(CallEvent).where(CallEvent.call_id == call_id))).scalars().all())
            assert count == 2
            assert call.error_count == 1
            assert call.warning_count == 1
    finally:
        await _delete_user(user_id)


@pytest.mark.anyio
async def test_provider_configuration_is_a_sanitized_immutable_snapshot(monkeypatch):
    user_id, call_id = await _create_call()
    try:
        assert await snapshot_call_configuration(
            call_id,
            stt_provider="deepgram",
            stt_model="nova-3",
            stt_language="en",
            llm_provider="google",
            llm_model="gemini-test",
            tts_provider="cartesia",
            tts_model="sonic-3",
            tts_voice="voice-1",
            provider_config={
                "llm": {"model": "gemini-test", "api_key": "must-not-store"}
            },
        )
        monkeypatch.setenv("GOOGLE_MODEL", "changed-after-call")
        async with AsyncSessionLocal() as db:
            call = await db.get(Call, call_id)
            assert call.stt_model == "nova-3"
            assert call.llm_model == "gemini-test"
            assert call.tts_model == "sonic-3"
            assert call.tts_voice == "voice-1"
            assert call.provider_config["llm"]["api_key"] == "[REDACTED]"
    finally:
        await _delete_user(user_id)


@pytest.mark.anyio
async def test_pcm_encoder_produces_bounded_mono_mp3(tmp_path: Path):
    pcm = tmp_path / "call.pcm"
    mp3 = tmp_path / "call.mp3"
    pcm.write_bytes(b"\x00\x00" * 16_000)
    size, duration_ms, checksum = _encode_pcm_to_mp3(pcm, mp3)
    assert mp3.read_bytes()[:3] == b"ID3"
    assert 8_000 < size < 12_000
    assert duration_ms == 1000.0
    assert len(checksum) == 64


@pytest.mark.anyio
async def test_call_audio_buffer_offset_includes_emitted_and_pending_audio():
    processor = CallAudioBufferProcessor(sample_rate=16_000, num_channels=1)
    processor._sample_rate = 16_000
    processor._emitted_bytes = 32_000
    processor._user_audio_buffer.extend(b"\x00" * 16_000)
    assert processor.elapsed_ms == 1500.0


@pytest.mark.anyio
async def test_call_audio_buffer_preserves_start_and_teardown_silence():
    processor = CallAudioBufferProcessor(sample_rate=16_000, num_channels=1)
    processor._sample_rate = 16_000
    captured = []

    @processor.event_handler("on_audio_data")
    async def capture(_processor, audio, sample_rate, channels):
        captured.append((audio, sample_rate, channels))

    await processor.start_recording()
    processor._last_user_buffer_update_time -= 0.5
    processor._last_bot_buffer_update_time -= 0.5
    await processor.stop_recording()
    await asyncio.sleep(0)

    assert len(captured) == 1
    assert 15_900 <= len(captured[0][0]) <= 16_100
    assert captured[0][1:] == (16_000, 1)


@pytest.mark.anyio
async def test_failed_recording_writer_cannot_deadlock_finalize(monkeypatch, tmp_path: Path):
    states = []

    async def set_state(_call_id, **values):
        states.append(values)

    monkeypatch.setattr(recording_service, "_set_recording_state", set_state)
    writer = CallRecordingWriter(uuid.uuid4(), 1)
    writer.spool_path = tmp_path / "call.pcm"
    writer._failed = True
    writer._queue.put_nowait(b"orphaned")
    writer._writer_task = asyncio.create_task(asyncio.sleep(0))
    await writer._writer_task

    assert await asyncio.wait_for(writer.finalize(), timeout=0.1) is False
    assert states[-1]["status"] == "failed"


@pytest.mark.anyio
async def test_restart_recovery_and_purge_manage_the_private_mp3(monkeypatch, tmp_path: Path):
    spool_root = tmp_path / "spool"
    storage_root = tmp_path / "objects"
    monkeypatch.setenv("RECORDING_SPOOL_DIR", str(spool_root))
    monkeypatch.setenv("RECORDING_STORAGE_DIR", str(storage_root))
    user_id, call_id = await _create_call()
    spool = spool_root / f"{call_id}.pcm"
    spool.parent.mkdir(parents=True)
    spool.write_bytes(b"\x00\x00" * 16_000)
    try:
        async with AsyncSessionLocal() as db:
            db.add(CallRecording(
                call_id=call_id,
                status="processing",
                spool_path=str(spool),
                updated_at=utcnow() - timedelta(minutes=1),
            ))
            await db.commit()
        await finalize_call(call_id)

        assert await recover_unfinished_recordings() == 1
        object_path = storage_root / f"calls/{user_id}/{call_id}.mp3"
        assert object_path.read_bytes()[:3] == b"ID3"
        assert not spool.exists()
        async with AsyncSessionLocal() as db:
            recording = (
                await db.execute(
                    select(CallRecording).where(CallRecording.call_id == call_id)
                )
            ).scalars().one()
            assert recording.status == "available"
            assert recording.duration_ms == 1000.0

        assert await soft_delete_call(call_id, user_id)
        async with AsyncSessionLocal() as db:
            call = await db.get(Call, call_id)
            call.purge_after = utcnow() - timedelta(seconds=1)
            await db.commit()
        assert await purge_expired_calls() == 1
        assert not object_path.exists()
        async with AsyncSessionLocal() as db:
            assert await db.get(Call, call_id) is None
    finally:
        await _delete_user(user_id)


@pytest.mark.anyio
async def test_soft_delete_hides_recording_and_restore_recovers_it():
    user_id, call_id = await _create_call()
    try:
        ended_at = utcnow()
        async with AsyncSessionLocal() as db:
            call = await db.get(Call, call_id)
            call.started_at = ended_at - timedelta(seconds=2)
            db.add(CallRecording(
                call_id=call_id,
                status="available",
                object_key=f"calls/{user_id}/{call_id}.mp3",
            ))
            await db.commit()
        assert await finalize_call(call_id, ended_at=ended_at)
        assert await soft_delete_call(call_id, user_id)
        async with AsyncSessionLocal() as db:
            call = await db.get(Call, call_id)
            recording = (await db.execute(
                select(CallRecording).where(CallRecording.call_id == call_id)
            )).scalars().one()
            assert 1900 <= call.duration_ms <= 2100
            assert recording.status == "deleted"
        assert await restore_call(call_id, user_id)
        async with AsyncSessionLocal() as db:
            recording = (await db.execute(
                select(CallRecording).where(CallRecording.call_id == call_id)
            )).scalars().one()
            assert recording.status == "available"
    finally:
        await _delete_user(user_id)


@pytest.mark.anyio
async def test_concurrent_finalization_is_exactly_once():
    user_id, call_id = await _create_call()
    try:
        results = await asyncio.gather(
            finalize_call(call_id, end_reason="normal"),
            finalize_call(call_id, end_reason="duplicate"),
        )
        assert sorted(results) == [False, True]
        async with AsyncSessionLocal() as db:
            call = await db.get(Call, call_id)
            assert call.status == "completed"
            assert call.end_reason in {"normal", "duplicate"}
    finally:
        await _delete_user(user_id)


@pytest.mark.anyio
async def test_watchdog_marks_only_stale_nonterminal_calls_abandoned():
    user_id, call_id = await _create_call()
    try:
        async with AsyncSessionLocal() as db:
            call = await db.get(Call, call_id)
            call.updated_at = utcnow() - timedelta(minutes=2)
            await db.commit()
        assert await abandon_stale_calls(30) == 1
        async with AsyncSessionLocal() as db:
            call = await db.get(Call, call_id)
            assert call.status == "abandoned"
            assert call.end_reason == "worker_lost"
            assert call.ended_by == "watchdog"
    finally:
        await _delete_user(user_id)


@pytest.mark.anyio
async def test_pipeline_assembly_failure_is_durable_before_worker_start():
    user_id, call_id = await _create_call()
    try:
        async with AsyncSessionLocal() as db:
            call = await db.get(Call, call_id)
            call.runner_session_id = "runner-startup-failure"
            await db.commit()

        assert await fail_nonterminal_runner_call(
            "runner-startup-failure",
            RuntimeError("provider secret=must-not-leak"),
        )

        async with AsyncSessionLocal() as db:
            call = await db.get(Call, call_id)
            event = (
                await db.execute(
                    select(CallEvent).where(CallEvent.call_id == call_id)
                )
            ).scalars().one()
            assert call.status == "failed"
            assert call.end_reason == "pipeline_startup_failed"
            assert event.code == "pipeline.startup_failed"
            assert "must-not-leak" not in (event.operator_detail or "")
            assert "[REDACTED]" in (event.operator_detail or "")
    finally:
        await _delete_user(user_id)


@pytest.mark.anyio
async def test_deleted_and_purged_calls_are_excluded_from_cross_call_facts(monkeypatch):
    user_id, call_id = await _create_call()
    monkeypatch.setattr(task_queue, "enqueue", lambda *_args, **_kwargs: True)
    try:
        transcript = await save_transcript_entry(
            call_id,
            "You",
            "Call me Mira",
            source="stt_final",
            turn_id=1,
        )
        async with AsyncSessionLocal() as db:
            db.add(UserMemory(
                user_id=user_id,
                fact_type="profile",
                key="preferred_name",
                value="Mira",
                source_transcript_id=transcript.id,
            ))
            await db.commit()
        assert await _load_active_facts(user_id) == []
        await finalize_call(call_id)
        assert [fact.value for fact in await _load_active_facts(user_id)] == ["Mira"]
        await soft_delete_call(call_id, user_id)
        assert await _load_active_facts(user_id) == []
        await restore_call(call_id, user_id)
        assert [fact.value for fact in await _load_active_facts(user_id)] == ["Mira"]
        await soft_delete_call(call_id, user_id)
        async with AsyncSessionLocal() as db:
            call = await db.get(Call, call_id)
            call.purge_after = utcnow() - timedelta(seconds=1)
            await db.commit()
        assert await purge_expired_calls() == 1
        async with AsyncSessionLocal() as db:
            remaining = await db.execute(
                select(UserMemory).where(UserMemory.user_id == user_id)
            )
            assert remaining.scalars().first() is None
    finally:
        await _delete_user(user_id)
