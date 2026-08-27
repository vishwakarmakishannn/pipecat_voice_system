"""Immutable call history, diagnostics, metrics, and private recording access."""

from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

import jwt
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import and_, exists, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import ALGORITHM, SECRET_KEY, get_current_user
from core.database import get_db
from core.models import Call, CallEvent, CallOperation, CallRecording, CallTurn, TranscriptEntry, User
from core.recording_config import recording_access_ttl_seconds
from core.storage import storage_client
from services.calls import (
    CLIENT_EVENT_CODES,
    restore_call,
    save_call_event,
    soft_delete_call,
    summarize_direct_perceived_latency,
    summarize_numeric_metric,
    summarize_perceived_latency,
    summarize_stt_finalization,
)


router = APIRouter(prefix="/api/calls", tags=["calls"])


class ClientEventInput(BaseModel):
    code: str = Field(..., max_length=128)
    message: str = Field(..., min_length=1, max_length=1000)
    severity: Literal["warning", "error"] = "error"
    request_id: str | None = Field(None, max_length=255)
    details: dict[str, Any] = Field(default_factory=dict)


def _encode_cursor(started_at: datetime, call_id: uuid.UUID) -> str:
    raw = json.dumps({"started_at": started_at.isoformat(), "id": str(call_id)}).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(value: str | None) -> tuple[datetime, uuid.UUID] | None:
    if not value:
        return None
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded))
        return datetime.fromisoformat(payload["started_at"]), uuid.UUID(payload["id"])
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid cursor") from exc


def _recording_payload(recording: CallRecording | None) -> dict[str, Any] | None:
    if not recording:
        return None
    return {
        "status": recording.status,
        "mime_type": recording.mime_type,
        "codec": recording.codec,
        "channels": recording.channels,
        "sample_rate": recording.sample_rate,
        "duration_ms": recording.duration_ms,
        "size_bytes": recording.size_bytes,
        "checksum_sha256": recording.checksum_sha256,
        "failure_code": recording.failure_code,
        "failure_message": recording.failure_message,
        "created_at": recording.created_at,
        "updated_at": recording.updated_at,
    }


def _call_payload(call: Call, recording: CallRecording | None = None) -> dict[str, Any]:
    return {
        "id": str(call.id),
        "title": call.title,
        "summary": call.summary,
        "status": call.status,
        "transport": call.transport,
        "direction": call.direction,
        "started_at": call.started_at,
        "connected_at": call.connected_at,
        "ended_at": call.ended_at,
        "duration_ms": call.duration_ms,
        "end_reason": call.end_reason,
        "ended_by": call.ended_by,
        "providers": {
            "stt": {"provider": call.stt_provider, "model": call.stt_model, "language": call.stt_language},
            "llm": {"provider": call.llm_provider, "model": call.llm_model},
            "tts": {
                "provider": call.tts_provider,
                "model": call.tts_model,
                "voice": call.tts_voice,
                "language": call.tts_language,
            },
        },
        "counts": {
            "turns": call.turn_count,
            "tools": call.tool_call_count,
            "errors": call.error_count,
            "warnings": call.warning_count,
            "interruptions": call.interruption_count,
        },
        "latency": {
            "average_ms": call.avg_latency_ms,
            "p50_ms": call.p50_latency_ms,
            "p90_ms": call.p90_latency_ms,
        },
        "recording": _recording_payload(recording),
        "deleted_at": call.deleted_at,
        "purge_after": call.purge_after,
    }


async def _owned_call(
    db: AsyncSession, call_id: uuid.UUID, user_id: int, *, include_deleted: bool = True
) -> Call:
    conditions = [Call.id == call_id, Call.user_id == user_id]
    if not include_deleted:
        conditions.append(Call.deleted_at.is_(None))
    result = await db.execute(select(Call).where(*conditions))
    call = result.scalars().first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    return call


@router.get("")
async def list_calls(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    cursor: str | None = None,
    limit: int = Query(30, ge=1, le=100),
    call_status: str | None = Query(None, alias="status"),
    started_from: datetime | None = None,
    started_to: datetime | None = None,
    provider: str | None = None,
    model: str | None = None,
    recording_status: str | None = None,
    has_errors: bool | None = None,
    deleted: bool = False,
):
    conditions = [Call.user_id == current_user.id]
    conditions.append(Call.deleted_at.is_not(None) if deleted else Call.deleted_at.is_(None))
    if call_status:
        conditions.append(Call.status == call_status)
    if started_from:
        conditions.append(Call.started_at >= started_from)
    if started_to:
        conditions.append(Call.started_at <= started_to)
    if provider:
        conditions.append(or_(Call.stt_provider == provider, Call.llm_provider == provider, Call.tts_provider == provider))
    if model:
        conditions.append(or_(Call.stt_model == model, Call.llm_model == model, Call.tts_model == model))
    if has_errors is not None:
        conditions.append(Call.error_count > 0 if has_errors else Call.error_count == 0)
    decoded = _decode_cursor(cursor)
    if decoded:
        cursor_started, cursor_id = decoded
        conditions.append(
            or_(Call.started_at < cursor_started, and_(Call.started_at == cursor_started, Call.id < cursor_id))
        )
    statement = (
        select(Call, CallRecording)
        .outerjoin(CallRecording, CallRecording.call_id == Call.id)
        .where(*conditions)
        .order_by(Call.started_at.desc(), Call.id.desc())
        .limit(limit + 1)
    )
    if recording_status:
        statement = statement.where(CallRecording.status == recording_status)
    rows = list((await db.execute(statement)).all())
    has_more = len(rows) > limit
    rows = rows[:limit]
    next_cursor = _encode_cursor(rows[-1][0].started_at, rows[-1][0].id) if has_more else None
    return {"items": [_call_payload(call, recording) for call, recording in rows], "next_cursor": next_cursor}


@router.get("/{call_id}")
async def get_call(
    call_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    call = await _owned_call(db, call_id, current_user.id)
    recording = (await db.execute(select(CallRecording).where(CallRecording.call_id == call_id))).scalars().first()
    payload = _call_payload(call, recording)
    turn_metrics = list(
        (
            await db.execute(
                select(CallTurn.metrics).where(CallTurn.call_id == call_id)
            )
        ).scalars()
    )
    metric_records = [metrics or {} for metrics in turn_metrics]
    direct_summary = summarize_direct_perceived_latency(
        metric_records
    )
    payload["latency"].update(
        {
            "basis": "direct_voice_user_stop_to_browser_playback",
            "sample_count": direct_summary["count"],
            "cohorts": {
                category: {
                    **summarize_perceived_latency(
                        metric_records,
                        category=category,
                    ),
                    **summarize_stt_finalization(
                        metric_records,
                        category=category,
                    ),
                }
                for category in ("direct", "rag", "tool")
            },
        }
    )
    turn_aggregates = (
        await db.execute(
            select(
                func.percentile_cont(0.5).within_group(CallTurn.stt_latency_ms),
                func.percentile_cont(0.9).within_group(CallTurn.stt_latency_ms),
                func.percentile_cont(0.5).within_group(CallTurn.llm_latency_ms),
                func.percentile_cont(0.9).within_group(CallTurn.llm_latency_ms),
                func.percentile_cont(0.5).within_group(CallTurn.tts_latency_ms),
                func.percentile_cont(0.9).within_group(CallTurn.tts_latency_ms),
                func.sum(CallTurn.llm_input_tokens),
                func.sum(CallTurn.llm_output_tokens),
                func.sum(CallTurn.stt_audio_ms),
                func.sum(CallTurn.tts_characters),
            ).where(CallTurn.call_id == call_id)
        )
    ).one()
    operation_aggregates = (
        await db.execute(
            select(
                func.sum(CallOperation.duration_ms).filter(
                    CallOperation.operation_type == "tool"
                ),
                func.sum(CallOperation.duration_ms).filter(
                    CallOperation.operation_type == "rag"
                ),
            ).where(CallOperation.call_id == call_id)
        )
    ).one()
    payload["latency"]["components"] = {
        "stt": {"p50_ms": turn_aggregates[0], "p90_ms": turn_aggregates[1]},
        "response_preparation": {
            "p50_ms": turn_aggregates[2],
            "p90_ms": turn_aggregates[3],
        },
        # Compatibility key for older API consumers. Its value is response
        # preparation, not provider-only model inference.
        "llm": {"p50_ms": turn_aggregates[2], "p90_ms": turn_aggregates[3]},
        "model_ttft": summarize_numeric_metric(
            metric_records, "llm_ttft_ms"
        ),
        "turn_release": summarize_numeric_metric(
            metric_records, "turn_release_ms"
        ),
        "endpointing": summarize_numeric_metric(
            metric_records, "endpointing_ms"
        ),
        "tts": {"p50_ms": turn_aggregates[4], "p90_ms": turn_aggregates[5]},
        "tool_total_ms": operation_aggregates[0],
        "rag_total_ms": operation_aggregates[1],
    }
    payload["usage"] = {
        "llm_input_tokens": int(turn_aggregates[6] or 0),
        "llm_output_tokens": int(turn_aggregates[7] or 0),
        "stt_audio_ms": float(turn_aggregates[8] or 0),
        "tts_characters": int(turn_aggregates[9] or 0),
    }
    payload["configuration"] = {
        "audio": {
            "input_sample_rate": call.input_sample_rate,
            "output_sample_rate": call.output_sample_rate,
            "recording_sample_rate": call.recording_sample_rate,
        },
        "provider": call.provider_config,
        "endpointing": call.endpointing_config,
        "pipeline": call.pipeline_config,
        "prompt_version": call.prompt_version,
        "prompt_hash": call.prompt_hash,
        "tool_schema_hash": call.tool_schema_hash,
        "rag_config_version": call.rag_config_version,
        "application_version": call.application_version,
        "recording_policy_version": call.recording_policy_version,
    }
    return payload


def _transcript_item(item: TranscriptEntry) -> dict[str, Any]:
    return {
        "item_type": "transcript",
        "id": str(item.id),
        "sequence": item.sequence,
        "turn_id": item.turn_id,
        "created_at": item.created_at,
        "speaker": item.speaker,
        "source": item.source,
        "text": item.text,
        "audio_offset_ms": item.audio_offset_ms,
        "audio_end_offset_ms": item.audio_end_offset_ms,
        "confidence": item.confidence,
        "is_final": item.is_final,
    }


def _operation_item(item: CallOperation) -> dict[str, Any]:
    return {
        "item_type": "operation",
        "id": str(item.id),
        "sequence": item.sequence,
        "turn_id": item.turn_id,
        "created_at": item.started_at,
        "operation_type": item.operation_type,
        "name": item.name,
        "status": item.status,
        "arguments": item.arguments,
        "result": item.result,
        "request_id": item.request_id,
        "error_code": item.error_code,
        "duration_ms": item.duration_ms,
    }


def _event_item(item: CallEvent) -> dict[str, Any]:
    return {
        "item_type": "event",
        "id": str(item.id),
        "sequence": item.sequence,
        "turn_id": item.turn_id,
        "created_at": item.created_at,
        "component": item.component,
        "code": item.code,
        "severity": item.severity,
        "outcome": item.outcome,
        "message": item.safe_message,
        "technical_detail": item.operator_detail,
        "provider": item.provider,
        "model": item.model,
        "request_id": item.request_id,
        "duration_ms": item.duration_ms,
        "retryable": item.retryable,
        "recovered": item.recovered,
        "fatal": item.fatal,
        "details": item.details,
        "fingerprint": item.fingerprint,
    }


@router.get("/{call_id}/timeline")
async def get_call_timeline(
    call_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    after: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=250),
):
    await _owned_call(db, call_id, current_user.id)
    fetch_limit = limit + 1
    transcripts = list((await db.execute(select(TranscriptEntry).where(TranscriptEntry.call_id == call_id, TranscriptEntry.sequence > after).order_by(TranscriptEntry.sequence).limit(fetch_limit))).scalars())
    operations = list((await db.execute(select(CallOperation).where(CallOperation.call_id == call_id, CallOperation.sequence > after).order_by(CallOperation.sequence).limit(fetch_limit))).scalars())
    represented_tool_request = exists(
        select(1).where(
            CallOperation.call_id == call_id,
            CallOperation.request_id == CallEvent.request_id,
        )
    )
    events = list((await db.execute(
        select(CallEvent)
        .where(
            CallEvent.call_id == call_id,
            CallEvent.sequence > after,
            or_(
                CallEvent.component != "tool",
                CallEvent.request_id.is_(None),
                ~represented_tool_request,
            ),
        )
        .order_by(CallEvent.sequence)
        .limit(fetch_limit)
    )).scalars())
    items = [_transcript_item(x) for x in transcripts] + [_operation_item(x) for x in operations] + [_event_item(x) for x in events]
    items.sort(key=lambda item: item["sequence"])
    has_more = len(items) > limit
    items = items[:limit]
    return {"items": items, "next_cursor": items[-1]["sequence"] if has_more and items else None}


@router.get("/{call_id}/turns")
async def get_call_turns(
    call_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    after: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=250),
):
    await _owned_call(db, call_id, current_user.id)
    rows = list((await db.execute(select(CallTurn).where(CallTurn.call_id == call_id, CallTurn.sequence > after).order_by(CallTurn.sequence).limit(limit + 1))).scalars())
    has_more = len(rows) > limit
    rows = rows[:limit]
    items = [{
        "id": str(turn.id), "sequence": turn.sequence, "input_mode": turn.input_mode,
        "outcome": turn.outcome, "interrupted": turn.interrupted,
        "started_at": turn.started_at, "ended_at": turn.ended_at,
        "stt_latency_ms": turn.stt_latency_ms, "llm_latency_ms": turn.llm_latency_ms,
        "tts_latency_ms": turn.tts_latency_ms, "tool_latency_ms": turn.tool_latency_ms,
        "rag_latency_ms": turn.rag_latency_ms, "first_audio_latency_ms": turn.first_audio_latency_ms,
        "end_to_end_latency_ms": turn.end_to_end_latency_ms,
        "llm_input_tokens": turn.llm_input_tokens, "llm_output_tokens": turn.llm_output_tokens,
        "stt_audio_ms": turn.stt_audio_ms, "tts_characters": turn.tts_characters,
        "metrics": turn.metrics,
    } for turn in rows]
    return {"items": items, "next_cursor": rows[-1].sequence if has_more and rows else None}


@router.post("/{call_id}/client-events", status_code=202)
async def create_client_event(
    call_id: uuid.UUID,
    event: ClientEventInput,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    call = await _owned_call(db, call_id, current_user.id, include_deleted=False)
    if call.status not in {"initializing", "active", "ending"}:
        raise HTTPException(status_code=409, detail="Terminal calls are immutable")
    if event.code not in CLIENT_EVENT_CODES:
        raise HTTPException(status_code=422, detail="Unsupported client event code")
    if len(json.dumps(event.details, default=str)) > 4000:
        raise HTTPException(status_code=413, detail="Client event details are too large")
    saved = await save_call_event(
        call_id,
        component="transport",
        code=event.code,
        severity=event.severity,
        outcome="degraded",
        safe_message=event.message,
        request_id=event.request_id,
        details=event.details,
        retryable=True,
    )
    return {"accepted": saved is not None}


@router.post("/{call_id}/recording-access")
async def create_recording_access(
    call_id: uuid.UUID,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _owned_call(db, call_id, current_user.id, include_deleted=False)
    recording = (await db.execute(select(CallRecording).where(CallRecording.call_id == call_id, CallRecording.status == "available"))).scalars().first()
    if not recording or not recording.object_key:
        raise HTTPException(status_code=409, detail="Recording is not available")
    ttl = recording_access_ttl_seconds()
    if storage_client.use_s3:
        url = await storage_client.create_presigned_get_url(recording.object_key, ttl)
    else:
        token = jwt.encode(
            {"sub": str(current_user.id), "call_id": str(call_id), "aud": "call-recording", "exp": datetime.now(timezone.utc).timestamp() + ttl},
            SECRET_KEY,
            algorithm=ALGORITHM,
        )
        url = str(request.url_for("stream_call_recording", call_id=str(call_id))) + f"?token={token}"
    return {"url": url, "expires_in": ttl}


@router.get("/{call_id}/recording", name="stream_call_recording")
async def stream_call_recording(call_id: uuid.UUID, token: str, db: AsyncSession = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], audience="call-recording")
        if payload.get("call_id") != str(call_id):
            raise ValueError("call mismatch")
        user_id = int(payload["sub"])
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid recording access token") from exc
    await _owned_call(db, call_id, user_id, include_deleted=False)
    recording = (await db.execute(select(CallRecording).where(CallRecording.call_id == call_id, CallRecording.status == "available"))).scalars().first()
    if not recording or not recording.object_key or storage_client.use_s3:
        raise HTTPException(status_code=404, detail="Recording not found")
    path = storage_client.local_object_path(recording.object_key)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Recording object not found")
    return FileResponse(
        path,
        media_type="audio/mpeg",
        filename=f"call-{call_id}.mp3",
        content_disposition_type="inline",
    )


@router.delete("/{call_id}")
async def delete_call(call_id: uuid.UUID, current_user: User = Depends(get_current_user)):
    if not await soft_delete_call(call_id, current_user.id):
        raise HTTPException(status_code=409, detail="Only an existing terminal call can be deleted")
    return {"status": "deleted", "recovery_days": 30}


@router.post("/{call_id}/restore")
async def restore_deleted_call(call_id: uuid.UUID, current_user: User = Depends(get_current_user)):
    if not await restore_call(call_id, current_user.id):
        raise HTTPException(status_code=409, detail="Call is not restorable")
    return {"status": "restored"}
