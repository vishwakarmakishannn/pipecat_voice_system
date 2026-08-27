"""Watchdog and retention maintenance for immutable calls."""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger
from sqlalchemy import select, text

from core.database import AsyncSessionLocal
from core.models import Call, CallRecording
from core.storage import storage_client
from core.recording_config import recording_spool_dir
from services.calls import abandon_stale_calls, purge_expired_call_row, utcnow


async def purge_expired_calls() -> int:
    """Claim due calls before external deletion so restore cannot race purge."""
    now = utcnow()
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Call, CallRecording)
            .outerjoin(CallRecording, CallRecording.call_id == Call.id)
            .where(
                Call.purge_after.is_not(None),
                Call.purge_after <= now,
                Call.purge_started_at.is_(None),
            )
            # PostgreSQL cannot lock the nullable recording side of a LEFT
            # JOIN. Only the call row is the purge/restore serialization point.
            .with_for_update(skip_locked=True, of=Call)
        )
        rows = list(result.all())
        for call, _recording in rows:
            call.purge_started_at = now
            call.updated_at = now
        if rows:
            await db.commit()
    purged = 0
    for call, recording in rows:
        try:
            if recording and recording.object_key:
                await storage_client.delete_file_strict(recording.object_key)
            if recording and recording.spool_path:
                expected = (recording_spool_dir() / f"{call.id}.pcm").resolve()
                configured = Path(recording.spool_path).expanduser().resolve()
                if configured != expected:
                    raise ValueError("recording spool path did not match its call ID")
                configured.unlink(missing_ok=True)
            async with AsyncSessionLocal() as db:
                await db.execute(text("SET LOCAL aura.allow_call_purge = 'on'"))
                purged += int(await purge_expired_call_row(db, call.id, utcnow()))
                await db.commit()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "call_purge status=retryable_failure call_id={}", call.id
            )
            async with AsyncSessionLocal() as db:
                retry_call = await db.get(Call, call.id, with_for_update=True)
                if retry_call is not None and retry_call.deleted_at is not None:
                    retry_call.purge_started_at = None
                    retry_call.updated_at = utcnow()
                    await db.commit()
    return purged


async def call_maintenance_loop(stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            abandoned = await abandon_stale_calls(30)
            from services.recordings import recover_unfinished_recordings

            recovered = await recover_unfinished_recordings()
            purged = await purge_expired_calls()
            if abandoned or recovered or purged:
                logger.info(
                    "call_maintenance abandoned={} recordings_recovered={} purged={}",
                    abandoned,
                    recovered,
                    purged,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("call_maintenance status=failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=30)
        except TimeoutError:
            pass
