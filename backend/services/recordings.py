"""Nonblocking PCM spooling, MP3 finalization, and restart recovery."""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from datetime import timedelta
from pathlib import Path

import aiofiles
import av
from loguru import logger
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from sqlalchemy import or_, select

from core.database import AsyncSessionLocal
from core.models import Call, CallRecording, TERMINAL_CALL_STATUSES
from core.recording_config import (
    RECORDING_BIT_RATE,
    RECORDING_CHANNELS,
    RECORDING_SAMPLE_RATE,
    local_recording_dir,
    recording_queue_chunks,
    recording_spool_dir,
)
from core.storage import storage_client
from core.task_queue import task_queue
from services.calls import save_call_event, utcnow


_STOP = object()


class CallAudioBufferProcessor(AudioBufferProcessor):
    """Expose the current mixed-stream position, including the pending chunk."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._emitted_bytes = 0

    @property
    def elapsed_ms(self) -> float:
        if not self.sample_rate:
            return 0.0
        pending = max(len(self._user_audio_buffer), len(self._bot_audio_buffer))
        return round(
            (self._emitted_bytes + pending)
            / (self.sample_rate * self.num_channels * 2)
            * 1000,
            1,
        )

    async def start_recording(self):
        starting = not self._recording
        if starting:
            self._emitted_bytes = 0
        await super().start_recording()
        if starting and self._recording:
            now = time.monotonic()
            # Pipecat normally starts with no time origin, which omits the
            # silence between call start and the first audio frame.
            self._last_user_buffer_update_time = now
            self._last_bot_buffer_update_time = now

    async def stop_recording(self):
        if self._recording and self.sample_rate:
            timestamps = [
                value
                for value in (
                    self._last_user_buffer_update_time,
                    self._last_bot_buffer_update_time,
                )
                if value is not None
            ]
            if timestamps:
                # Input frames normally keep the timeline current even in
                # silence. Pad the small teardown tail so the MP3 and call
                # duration stay aligned without allowing an unbounded final
                # allocation after a suspended browser.
                trailing_seconds = min(1.0, max(0.0, time.monotonic() - max(timestamps)))
                silence_bytes = int(trailing_seconds * self.sample_rate * 2)
                silence_bytes -= silence_bytes % 2
                if silence_bytes:
                    silence = b"\x00" * silence_bytes
                    self._user_audio_buffer.extend(silence)
                    self._bot_audio_buffer.extend(silence)
        await super().stop_recording()

    async def _call_on_audio_data_handler(self):
        pending = max(len(self._user_audio_buffer), len(self._bot_audio_buffer))
        await super()._call_on_audio_data_handler()
        if pending:
            self._emitted_bytes += pending


def _safe_spool_path(call_id: uuid.UUID) -> Path:
    root = recording_spool_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{call_id}.pcm"


def _object_key(user_id: int, call_id: uuid.UUID) -> str:
    return f"calls/{user_id}/{call_id}.mp3"


def _encode_pcm_to_mp3(source: Path, destination: Path) -> tuple[int, float, str]:
    """Encode raw mono s16le PCM without loading the call into memory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    input_container = av.open(
        str(source),
        mode="r",
        format="s16le",
        options={"sample_rate": str(RECORDING_SAMPLE_RATE), "channels": "1"},
    )
    output_container = av.open(str(destination), mode="w", format="mp3")
    stream = output_container.add_stream("libmp3lame", rate=RECORDING_SAMPLE_RATE)
    stream.bit_rate = RECORDING_BIT_RATE
    stream.layout = "mono"
    try:
        for frame in input_container.decode(audio=0):
            frame.sample_rate = RECORDING_SAMPLE_RATE
            for packet in stream.encode(frame):
                output_container.mux(packet)
        for packet in stream.encode(None):
            output_container.mux(packet)
    finally:
        input_container.close()
        output_container.close()

    digest = hashlib.sha256()
    size = 0
    with destination.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    pcm_size = source.stat().st_size
    duration_ms = pcm_size / (RECORDING_SAMPLE_RATE * RECORDING_CHANNELS * 2) * 1000
    return size, round(duration_ms, 1), digest.hexdigest()


async def _set_recording_state(call_id: uuid.UUID, **values) -> bool:
    async with AsyncSessionLocal() as db:
        call_result = await db.execute(
            select(Call).where(Call.id == call_id).with_for_update()
        )
        call = call_result.scalars().first()
        if call is None:
            return False
        result = await db.execute(
            select(CallRecording).where(CallRecording.call_id == call_id).with_for_update()
        )
        recording = result.scalars().first()
        requested_status = values.get("status")
        if call.deleted_at is not None and requested_status != "deleted":
            # Soft deletion wins over any in-flight encoder/upload transition.
            if recording is not None and recording.status != "deleted":
                recording.status = "deleted"
                recording.updated_at = utcnow()
                await db.commit()
            return False
        if recording is None:
            recording = CallRecording(call_id=call_id)
            db.add(recording)
        for key, value in values.items():
            if hasattr(recording, key):
                setattr(recording, key, value)
        recording.updated_at = utcnow()
        await db.commit()
        return True


async def mark_recording_failed(
    call_id: uuid.UUID,
    code: str,
    message: str,
) -> None:
    await _set_recording_state(
        call_id,
        status="failed",
        failure_code=code[:128],
        failure_message=message[:1000],
    )


async def finalize_recording(call_id: uuid.UUID, user_id: int, spool_path: Path) -> bool:
    """Encode/upload a completed spool and retain only the private MP3."""
    if not await _set_recording_state(
        call_id, status="processing", spool_path=str(spool_path)
    ):
        return False
    destination = local_recording_dir() / f"{call_id}.mp3.tmp"
    object_key = _object_key(user_id, call_id)
    try:
        if not spool_path.exists() or spool_path.stat().st_size == 0:
            raise ValueError("recording spool contained no audio")
        size, duration_ms, checksum = await asyncio.to_thread(
            _encode_pcm_to_mp3, spool_path, destination
        )
        await storage_client.upload_path(destination, object_key, content_type="audio/mpeg")
        made_available = await _set_recording_state(
            call_id,
            status="available",
            object_key=object_key,
            spool_path=None,
            duration_ms=duration_ms,
            size_bytes=size,
            checksum_sha256=checksum,
            failure_code=None,
            failure_message=None,
        )
        if not made_available:
            # Deletion may have won while encoding/uploading. Do not leave a
            # newly-created private object behind or resurrect its metadata.
            await storage_client.delete_file_strict(object_key)
            spool_path.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            return False
        spool_path.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        return True
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("recording_finalize status=failed call_id={}", call_id)
        destination.unlink(missing_ok=True)
        await _set_recording_state(
            call_id,
            status="failed",
            failure_code="recording.encode_failed",
            failure_message="The call recording could not be finalized.",
        )
        await save_call_event(
            call_id,
            component="recording",
            code="recording.encode_failed",
            severity="error",
            outcome="degraded",
            safe_message="Call recording processing failed; the voice call was not interrupted.",
            operator_detail=exc,
            recovered=False,
            fatal=False,
        )
        return False


class CallRecordingWriter:
    """Bounded, session-scoped writer used by Pipecat's AudioBufferProcessor."""

    def __init__(self, call_id: uuid.UUID, user_id: int):
        self.call_id = call_id
        self.user_id = user_id
        self.spool_path = _safe_spool_path(call_id)
        self._queue: asyncio.Queue[bytes | object] = asyncio.Queue(
            maxsize=recording_queue_chunks()
        )
        self._writer_task: asyncio.Task | None = None
        self._failed = False
        self._closed = False
        self._accepted_bytes = 0

    @property
    def elapsed_ms(self) -> float:
        return round(
            self._accepted_bytes / (RECORDING_SAMPLE_RATE * RECORDING_CHANNELS * 2) * 1000,
            1,
        )

    async def start(self) -> None:
        self.spool_path.unlink(missing_ok=True)
        await _set_recording_state(
            self.call_id,
            status="recording",
            spool_path=str(self.spool_path),
            sample_rate=RECORDING_SAMPLE_RATE,
            channels=RECORDING_CHANNELS,
            codec="mp3",
            mime_type="audio/mpeg",
        )
        self._writer_task = asyncio.create_task(
            self._write_loop(), name=f"recording-writer-{self.call_id}"
        )

    async def _write_loop(self) -> None:
        self.spool_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            async with aiofiles.open(self.spool_path, "ab") as handle:
                while True:
                    item = await self._queue.get()
                    try:
                        if item is _STOP:
                            return
                        await handle.write(item)
                    finally:
                        self._queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failed = True
            logger.exception("recording_spool status=failed call_id={}", self.call_id)
            await save_call_event(
                self.call_id,
                component="recording",
                code="recording.spool_failed",
                severity="error",
                outcome="degraded",
                safe_message="Call audio could not be written to the recording spool.",
                operator_detail=exc,
            )
        finally:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                else:
                    self._queue.task_done()

    async def accept_audio(self, audio: bytes, sample_rate: int, channels: int) -> None:
        if self._closed or self._failed or not audio:
            return
        if sample_rate != RECORDING_SAMPLE_RATE or channels != RECORDING_CHANNELS:
            self._failed = True
            task_queue.enqueue(
                save_call_event,
                self.call_id,
                component="recording",
                code="recording.format_mismatch",
                severity="error",
                outcome="degraded",
                safe_message="The recording stream used an unexpected audio format.",
                details={"sample_rate": sample_rate, "channels": channels},
                key=str(self.call_id),
            )
            return
        try:
            self._queue.put_nowait(bytes(audio))
            self._accepted_bytes += len(audio)
        except asyncio.QueueFull:
            self._failed = True
            task_queue.enqueue(
                save_call_event,
                self.call_id,
                component="recording",
                code="persistence.queue_rejected",
                severity="error",
                outcome="degraded",
                safe_message="Recording data was dropped because the spool writer was overloaded.",
                key=str(self.call_id),
            )

    async def finalize(self) -> bool:
        if self._closed:
            return not self._failed
        self._closed = True
        if self._writer_task:
            if not self._writer_task.done():
                await self._queue.put(_STOP)
                await self._queue.join()
            await asyncio.gather(self._writer_task, return_exceptions=True)
        if self._failed:
            await _set_recording_state(
                self.call_id,
                status="failed",
                failure_code="recording.spool_failed",
                failure_message="The recording spool was incomplete.",
            )
            return False
        return await finalize_recording(self.call_id, self.user_id, self.spool_path)


async def recover_unfinished_recordings() -> int:
    """Finish intact spools left by a worker/process restart."""
    stale_cutoff = utcnow() - timedelta(seconds=30)
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(CallRecording, Call.user_id)
            .join(Call, Call.id == CallRecording.call_id)
            .where(
                CallRecording.status.in_(("recording", "processing")),
                CallRecording.updated_at < stale_cutoff,
                or_(
                    Call.status.in_(TERMINAL_CALL_STATUSES),
                    Call.updated_at < stale_cutoff,
                ),
            )
            .with_for_update(skip_locked=True)
        )
        rows = list(result.all())
        # Move the lease timestamp while holding the row locks. Other worker
        # replicas will skip these rows until this attempt itself becomes stale.
        claimed_at = utcnow()
        for recording, _user_id in rows:
            recording.status = "processing"
            recording.updated_at = claimed_at
        if rows:
            await db.commit()
    recovered = 0
    for recording, user_id in rows:
        expected_path = _safe_spool_path(recording.call_id)
        path = (
            Path(recording.spool_path).expanduser().resolve()
            if recording.spool_path
            else expected_path
        )
        if path != expected_path:
            logger.error(
                "recording_recovery status=unsafe_spool_path call_id={}",
                recording.call_id,
            )
            await mark_recording_failed(
                recording.call_id,
                "recording.unsafe_spool_path",
                "The interrupted recording spool path was invalid.",
            )
            continue
        if await finalize_recording(recording.call_id, user_id, path):
            recovered += 1
    return recovered
