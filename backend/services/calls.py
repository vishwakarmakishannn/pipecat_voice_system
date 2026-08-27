"""Durable call lifecycle, timeline, and telemetry services."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import AsyncSessionLocal
from core.models import (
    Call,
    CallEvent,
    CallOperation,
    CallRecording,
    CallTurn,
    TERMINAL_CALL_STATUSES,
    TranscriptEntry,
)
from core.task_queue import task_queue


WRITABLE_CALL_STATUSES = {"initializing", "active", "ending"}
EVENT_SEVERITIES = {"info", "warning", "error", "critical"}
EVENT_OUTCOMES = {"observed", "recovered", "degraded", "failed", "cancelled"}
CLIENT_EVENT_CODES = {
    "transport.connection_failed",
    "transport.connection_lost",
    "transport.reconnect_failed",
    "transport.audio_playback_failed",
    "transport.microphone_failed",
}
MAX_EVENT_DETAIL_CHARS = 4000
MAX_OPERATION_JSON_CHARS = 100_000
SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|authorization|bearer|token|secret|password)[\"']?\s*[:=]\s*[\"']?[^\s,;}\"']+"
)
JWT_PATTERN = re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")
CREDENTIAL_VALUE_PATTERN = re.compile(
    r"(?i)\b(?:Bearer\s+[^\s,;}]+|sk-[A-Za-z0-9_-]{12,}|gsk_[A-Za-z0-9_-]{12,}|"
    r"tvly-[A-Za-z0-9_-]{12,}|AIza[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})"
)
SENSITIVE_KEYS = {
    "authorization", "apikey", "token", "accesstoken", "refreshtoken",
    "bearertoken", "secret", "clientsecret", "password", "headers",
    "prompt", "systeminstruction",
    "messages", "requestbody", "responsebody", "rawbody", "providerbody",
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json_safe(value: Any, *, max_chars: int = MAX_OPERATION_JSON_CHARS) -> Any:
    try:
        encoded = json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        encoded = json.dumps(str(value), ensure_ascii=False)
    if len(encoded) > max_chars:
        return {"truncated": True, "preview": encoded[:max_chars]}
    return json.loads(encoded)


def sanitize_detail(value: Any) -> str:
    detail = re.sub(r"\s+", " ", str(value or "")).strip()
    detail = SECRET_PATTERN.sub(r"\1=[REDACTED]", detail)
    detail = JWT_PATTERN.sub("[REDACTED_TOKEN]", detail)
    detail = CREDENTIAL_VALUE_PATTERN.sub("[REDACTED_CREDENTIAL]", detail)
    detail = re.sub(r"(?i)(https?://[^\s?]+)\?[^\s]+", r"\1?[REDACTED_QUERY]", detail)
    return detail[:MAX_EVENT_DETAIL_CHARS]


def sanitize_metadata(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        return "[TRUNCATED_DEPTH]"
    if isinstance(value, dict):
        cleaned = {}
        for raw_key, item in list(value.items())[:100]:
            key = str(raw_key)[:128]
            normalized = re.sub(r"[^a-z0-9]", "", key.lower())
            if normalized in SENSITIVE_KEYS or normalized.endswith("apikey") or normalized.endswith("password") or normalized.endswith("secret"):
                cleaned[key] = "[REDACTED]"
            else:
                cleaned[key] = sanitize_metadata(item, depth=depth + 1)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [sanitize_metadata(item, depth=depth + 1) for item in list(value)[:100]]
    if isinstance(value, str):
        return sanitize_detail(value)
    return value if value is None or isinstance(value, (int, float, bool)) else sanitize_detail(value)


def event_fingerprint(
    *,
    code: str,
    turn_id: int | None,
    request_id: str | None,
    safe_message: str,
) -> str:
    identity = request_id or safe_message.strip()
    raw = f"{code}|{turn_id or 0}|{identity}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def is_terminal(status: str) -> bool:
    return status in TERMINAL_CALL_STATUSES


async def _locked_call(db: AsyncSession, call_id: uuid.UUID) -> Call | None:
    result = await db.execute(select(Call).where(Call.id == call_id).with_for_update())
    return result.scalars().first()


def _next_sequence(call: Call) -> int:
    call.next_timeline_sequence = int(call.next_timeline_sequence or 0) + 1
    return call.next_timeline_sequence


async def mark_call_active(call_id: uuid.UUID) -> bool:
    async with AsyncSessionLocal() as db:
        call = await _locked_call(db, call_id)
        if not call or call.status != "initializing":
            return False
        call.status = "active"
        call.connected_at = utcnow()
        call.updated_at = utcnow()
        await db.commit()
        return True


async def fail_nonterminal_runner_call(
    runner_session_id: str,
    error: BaseException,
) -> bool:
    """Finalize an assembly/startup failure even before a worker exists."""
    if not runner_session_id:
        return False
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Call.id)
            .where(
                Call.runner_session_id == runner_session_id,
                Call.status.in_(tuple(WRITABLE_CALL_STATUSES)),
            )
            .order_by(Call.started_at.desc())
            .limit(1)
        )
        call_id = result.scalar_one_or_none()
    if not call_id:
        return False
    await save_call_event(
        call_id,
        component="pipeline",
        code="pipeline.startup_failed",
        severity="critical",
        outcome="failed",
        safe_message="The call could not start because the voice pipeline failed during setup.",
        operator_detail=error,
        fatal=True,
    )
    return await finalize_call(
        call_id,
        status="failed",
        end_reason="pipeline_startup_failed",
        ended_by="system",
    )


async def touch_active_call(call_id: uuid.UUID) -> bool:
    """Heartbeat one live worker so the stale-call watchdog is accurate."""
    async with AsyncSessionLocal() as db:
        call = await _locked_call(db, call_id)
        if not call or call.status not in {"initializing", "active", "ending"}:
            return False
        call.updated_at = utcnow()
        await db.commit()
        return True


async def snapshot_call_configuration(call_id: uuid.UUID, **values: Any) -> bool:
    allowed = {
        "transport", "direction", "stt_provider", "stt_model", "stt_language",
        "llm_provider", "llm_model", "tts_provider", "tts_model", "tts_voice",
        "tts_language", "input_sample_rate", "output_sample_rate",
        "recording_sample_rate", "provider_config", "endpointing_config",
        "pipeline_config", "prompt_version", "prompt_hash", "tool_schema_hash",
        "rag_config_version", "application_version",
    }
    async with AsyncSessionLocal() as db:
        call = await _locked_call(db, call_id)
        if not call or is_terminal(call.status):
            return False
        for key, value in values.items():
            if key in allowed:
                setattr(
                    call,
                    key,
                    _json_safe(sanitize_metadata(value))
                    if key.endswith("_config")
                    else value,
                )
        call.updated_at = utcnow()
        await db.commit()
        return True


async def save_transcript_entry(
    call_id: uuid.UUID | None,
    speaker: str,
    text: str,
    *,
    source: str,
    turn_id: int | None = None,
    audio_offset_ms: float | None = None,
    audio_end_offset_ms: float | None = None,
    confidence: float | None = None,
    persistence_id: str | None = None,
) -> TranscriptEntry | None:
    text = (text or "").strip()
    if not call_id or not text or speaker not in {"You", "Aura"}:
        return None
    async with AsyncSessionLocal() as db:
        call = await _locked_call(db, call_id)
        if not call or call.status not in WRITABLE_CALL_STATUSES:
            logger.warning("call_timeline rejected_terminal_write call_id={} speaker={}", call_id, speaker)
            return None
        if persistence_id:
            existing = await db.execute(
                select(TranscriptEntry).where(
                    TranscriptEntry.call_id == call_id,
                    TranscriptEntry.persistence_id == persistence_id,
                )
            )
            if duplicate := existing.scalars().first():
                return duplicate
        entry = TranscriptEntry(
            call_id=call_id,
            turn_id=turn_id,
            sequence=_next_sequence(call),
            speaker=speaker,
            source=source,
            text=text,
            audio_offset_ms=audio_offset_ms,
            audio_end_offset_ms=audio_end_offset_ms,
            confidence=confidence,
            is_final=True,
            persistence_id=persistence_id,
        )
        db.add(entry)
        if speaker == "You" and call.title == "New call":
            call.title = " ".join(text.split()[:6])[:255] or call.title
        call.updated_at = utcnow()
        await db.commit()
        await db.refresh(entry)

    from core.task_queue import task_queue
    task_queue.enqueue(
        _process_transcript_enrichment,
        call_id,
        entry.id,
        key=str(call_id),
        enrichment=True,
    )
    return entry


async def _process_transcript_enrichment(call_id: uuid.UUID, transcript_id: int) -> None:
    # Imported lazily to keep call persistence independent from optional memory providers.
    from services.memory import process_saved_transcript

    await process_saved_transcript(call_id, transcript_id)


async def save_call_operation(
    call_id: uuid.UUID | None,
    *,
    operation_type: str,
    name: str,
    arguments: Any,
    result: Any = None,
    status: str = "completed",
    turn_id: int | None = None,
    request_id: str | None = None,
    error_code: str | None = None,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
    duration_ms: float | None = None,
    persistence_id: str | None = None,
) -> CallOperation | None:
    if not call_id or operation_type not in {"tool", "rag"} or not name:
        return None
    async with AsyncSessionLocal() as db:
        call = await _locked_call(db, call_id)
        if not call or call.status not in WRITABLE_CALL_STATUSES:
            return None
        if persistence_id:
            existing = await db.execute(
                select(CallOperation).where(
                    CallOperation.call_id == call_id,
                    CallOperation.persistence_id == persistence_id,
                )
            )
            if duplicate := existing.scalars().first():
                return duplicate
        operation = CallOperation(
            call_id=call_id,
            turn_id=turn_id,
            sequence=_next_sequence(call),
            operation_type=operation_type,
            name=name[:255],
            arguments=_json_safe(sanitize_metadata(arguments)),
            result=_json_safe(sanitize_metadata(result)) if result is not None else None,
            status=status[:24],
            request_id=(request_id or "")[:255] or None,
            error_code=(error_code or "")[:128] or None,
            started_at=started_at or utcnow(),
            ended_at=ended_at or utcnow(),
            duration_ms=duration_ms,
            persistence_id=persistence_id,
        )
        db.add(operation)
        if operation_type == "tool":
            call.tool_call_count = int(call.tool_call_count or 0) + 1
        call.updated_at = utcnow()
        await db.commit()
        await db.refresh(operation)
        logger.info(
            "call_operation call_id={} turn_id={} request_id={} type={} name={} status={} duration_ms={}",
            call_id,
            turn_id,
            request_id,
            operation_type,
            operation.name,
            operation.status,
            duration_ms,
        )
        return operation


async def save_call_event(
    call_id: uuid.UUID | None,
    *,
    component: str,
    code: str,
    severity: str,
    outcome: str,
    safe_message: str,
    operator_detail: Any = None,
    turn_id: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    request_id: str | None = None,
    duration_ms: float | None = None,
    retryable: bool = False,
    recovered: bool = False,
    fatal: bool = False,
    details: Any = None,
    fingerprint: str | None = None,
) -> CallEvent | None:
    safe_message = sanitize_detail(safe_message)
    if not call_id or not code or not safe_message:
        return None
    severity = severity if severity in EVENT_SEVERITIES else "error"
    outcome = outcome if outcome in EVENT_OUTCOMES else "failed"
    fingerprint = fingerprint or event_fingerprint(
        code=code, turn_id=turn_id, request_id=request_id, safe_message=safe_message
    )
    async with AsyncSessionLocal() as db:
        call = await _locked_call(db, call_id)
        if not call or is_terminal(call.status):
            return None
        existing = await db.execute(
            select(CallEvent).where(
                CallEvent.call_id == call_id,
                CallEvent.fingerprint == fingerprint,
            )
        )
        if duplicate := existing.scalars().first():
            return duplicate
        event = CallEvent(
            call_id=call_id,
            turn_id=turn_id,
            sequence=_next_sequence(call),
            component=component[:32],
            code=code[:128],
            severity=severity,
            outcome=outcome,
            safe_message=safe_message,
            operator_detail=sanitize_detail(operator_detail) or None,
            provider=(provider or "")[:64] or None,
            model=(model or "")[:255] or None,
            request_id=(request_id or "")[:255] or None,
            duration_ms=duration_ms,
            retryable=bool(retryable),
            recovered=bool(recovered),
            fatal=bool(fatal),
            details=_json_safe(sanitize_metadata(details or {}), max_chars=MAX_EVENT_DETAIL_CHARS),
            fingerprint=fingerprint,
        )
        db.add(event)
        if severity in {"error", "critical"}:
            call.error_count = int(call.error_count or 0) + 1
        elif severity == "warning":
            call.warning_count = int(call.warning_count or 0) + 1
        call.updated_at = utcnow()
        await db.commit()
        await db.refresh(event)
        logger.info(
            "call_event call_id={} turn_id={} request_id={} component={} code={} severity={} outcome={} recovered={} fatal={}",
            call_id,
            turn_id,
            request_id,
            component,
            code,
            severity,
            outcome,
            recovered,
            fatal,
        )
        return event


async def save_call_turn(call_id: uuid.UUID | None, payload: dict[str, Any]) -> CallTurn | None:
    if not call_id or not payload.get("turn_id"):
        return None
    sequence = int(payload["turn_id"])
    async with AsyncSessionLocal() as db:
        call = await _locked_call(db, call_id)
        if not call or call.status not in WRITABLE_CALL_STATUSES:
            return None
        result = await db.execute(
            select(CallTurn).where(CallTurn.call_id == call_id, CallTurn.sequence == sequence)
        )
        turn = result.scalars().first()
        if turn is None:
            turn = CallTurn(call_id=call_id, sequence=sequence)
            db.add(turn)
            call.turn_count = max(int(call.turn_count or 0), sequence)
        was_interrupted = bool(turn.interrupted)
        mapping = {
            "stt_latency_ms": "stt_latency_ms",
            "llm_latency_ms": "llm_latency_ms",
            "tts_latency_ms": "tts_latency_ms",
            "input_mode": "input_mode",
            "tool_latency_ms": "tool_latency_ms",
            "rag_latency_ms": "rag_latency_ms",
            "speech_ms": "stt_audio_ms",
            "llm_input_tokens": "llm_input_tokens",
            "llm_output_tokens": "llm_output_tokens",
            "tts_characters": "tts_characters",
            "outcome": "outcome",
            "interrupted": "interrupted",
        }
        for source, target in mapping.items():
            if payload.get(source) is not None:
                setattr(turn, target, payload[source])
        if payload.get("answer_audio_ms") is not None:
            turn.end_to_end_latency_ms = payload["answer_audio_ms"]
            turn.first_audio_latency_ms = payload["answer_audio_ms"]
        if payload.get("turn_started_unix_ms") is not None:
            turn.started_at = datetime.fromtimestamp(
                float(payload["turn_started_unix_ms"]) / 1000,
                tz=timezone.utc,
            )
        if not was_interrupted and turn.interrupted:
            call.interruption_count = int(call.interruption_count or 0) + 1
        previous_metrics = turn.metrics or {}
        previous_sample = _perceived_latency_sample(previous_metrics)
        incoming_sample = _perceived_latency_sample(payload)
        turn.metrics = _json_safe({**previous_metrics, **payload})
        cohort_summary = None
        direct_summary = None
        sample_changed = (
            incoming_sample is not None
            and incoming_sample != previous_sample
        )
        if sample_changed:
            # Browser-observed speech-end to first playback is the production
            # end-to-end metric. Keep generation-side answer_audio_ms in the
            # JSON breakdown, but do not present it as perceived latency.
            category, latest_perceived_ms = incoming_sample
            turn.end_to_end_latency_ms = latest_perceived_ms
            all_turns = list(
                (
                    await db.execute(
                        select(CallTurn).where(CallTurn.call_id == call_id)
                    )
                )
                .scalars()
                .all()
            )
            direct_summary = summarize_direct_perceived_latency(
                [candidate.metrics or {} for candidate in all_turns]
            )
            call.avg_latency_ms = direct_summary["average_ms"]
            call.p50_latency_ms = direct_summary["p50_ms"]
            call.p90_latency_ms = direct_summary["p90_ms"]
            records = [candidate.metrics or {} for candidate in all_turns]
            cohort_summary = summarize_perceived_latency(
                records,
                category=category,
            )
            cohort_summary.update(
                summarize_stt_finalization(records, category=category)
            )
        if payload.get("latency_complete"):
            turn.ended_at = utcnow()
        call.updated_at = utcnow()
        await db.commit()
        await db.refresh(turn)
        if cohort_summary is not None:
            logger.info(
                "voice_latency_percentiles call_id={} "
                "cohort={}_voice_browser count={} latest_ms={} "
                "p50_ms={} p90_ms={} stt_native_final_count={} "
                "stt_fallback_count={} stt_fallback_rate_pct={} "
                "stt_final_shorter_count={}",
                call_id,
                incoming_sample[0],
                cohort_summary["count"],
                incoming_sample[1],
                cohort_summary["p50_ms"],
                cohort_summary["p90_ms"],
                cohort_summary["native_final_count"],
                cohort_summary["fallback_count"],
                cohort_summary["fallback_rate_pct"],
                cohort_summary["final_shorter_count"],
            )
        return turn


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return round(ordered[lower], 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower), 1)


def _perceived_latency_sample(
    metrics: dict[str, Any],
) -> tuple[str, float] | None:
    """Return the category and production latency from one eligible client turn."""
    value = metrics.get("user_stop_to_playback_ms")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        return None
    if metrics.get("measurement_source") != "client":
        return None
    if metrics.get("latency_complete") is not True:
        return None
    category = metrics.get("category")
    if metrics.get("input_mode") != "voice" or category not in {"direct", "rag", "tool"}:
        return None
    if metrics.get("interrupted") is True or metrics.get("outcome") == "cancelled":
        return None
    return str(category), float(value)


def _direct_perceived_latency_ms(metrics: dict[str, Any]) -> float | None:
    """Return an eligible direct-turn production latency sample."""
    sample = _perceived_latency_sample(metrics)
    if sample is None or sample[0] != "direct":
        return None
    return sample[1]


def summarize_perceived_latency(
    records: list[dict[str, Any]],
    *,
    category: str,
) -> dict[str, float | int | None]:
    """Summarize one direct/RAG/tool browser-observed voice cohort."""
    values = [
        sample[1]
        for record in records
        if (sample := _perceived_latency_sample(record)) is not None
        and sample[0] == category
    ]
    return {
        "count": len(values),
        "average_ms": round(sum(values) / len(values), 1) if values else None,
        "p50_ms": _percentile(values, 0.5),
        "p90_ms": _percentile(values, 0.9),
    }


def summarize_stt_finalization(
    records: list[dict[str, Any]],
    *,
    category: str,
) -> dict[str, float | int | None]:
    """Summarize native-final/fallback diagnostics for one eligible cohort."""
    finalizations = []
    for record in records:
        sample = _perceived_latency_sample(record)
        if sample is None or sample[0] != category:
            continue
        value = record.get("stt_finalization_ms")
        if isinstance(value, dict) and isinstance(value.get("fallback_forced"), (int, float)):
            finalizations.append(value)
    fallback_count = sum(
        1 for value in finalizations if float(value.get("fallback_forced", 0)) >= 0.5
    )
    final_shorter_count = sum(
        1
        for value in finalizations
        if float(value.get("final_shorter_than_interim", 0) or 0) >= 0.5
    )
    total = len(finalizations)
    return {
        "stt_finalization_count": total,
        "native_final_count": total - fallback_count,
        "fallback_count": fallback_count,
        "fallback_rate_pct": (
            round(fallback_count * 100 / total, 1) if total else None
        ),
        "final_shorter_count": final_shorter_count,
    }


def summarize_numeric_metric(
    records: list[dict[str, Any]],
    metric: str,
) -> dict[str, float | int | None]:
    """Summarize a numeric metric stored in the extensible turn payload."""
    values = [
        float(value)
        for record in records
        if isinstance((value := record.get(metric)), (int, float))
        and not isinstance(value, bool)
        and value >= 0
    ]
    return {
        "count": len(values),
        "p50_ms": _percentile(values, 0.5),
        "p90_ms": _percentile(values, 0.9),
    }


def summarize_direct_perceived_latency(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Summarize the unbiased direct-turn cohort used by logs and call UI."""
    return summarize_perceived_latency(records, category="direct")


async def finalize_call(
    call_id: uuid.UUID | None,
    *,
    status: str = "completed",
    end_reason: str = "client_disconnect",
    ended_by: str = "client",
    ended_at: datetime | None = None,
) -> bool:
    if not call_id or status not in TERMINAL_CALL_STATUSES:
        return False
    async with AsyncSessionLocal() as db:
        call = await _locked_call(db, call_id)
        if not call or is_terminal(call.status):
            return False
        now = ended_at or utcnow()
        duration_ms = max(0.0, (now - call.started_at).total_seconds() * 1000)
        # Finish every read required to build the immutable terminal snapshot
        # before assigning the terminal status. SQLAlchemy autoflushes before
        # SELECTs; assigning status first would split finalization into multiple
        # UPDATEs and the database trigger correctly rejects the later ones.
        turns = list(
            (
                await db.execute(
                    select(CallTurn).where(CallTurn.call_id == call_id)
                )
            )
            .scalars()
            .all()
        )
        direct_summary = summarize_direct_perceived_latency(
            [turn.metrics or {} for turn in turns]
        )
        if direct_summary["count"]:
            call.avg_latency_ms = direct_summary["average_ms"]
            call.p50_latency_ms = direct_summary["p50_ms"]
            call.p90_latency_ms = direct_summary["p90_ms"]
        else:
            # Preserve useful summaries for legacy/non-browser calls that do
            # not have the client playback metric yet.
            latencies = [
                float(turn.end_to_end_latency_ms)
                for turn in turns
                if turn.end_to_end_latency_ms is not None
            ]
            if latencies:
                call.avg_latency_ms = round(sum(latencies) / len(latencies), 1)
                call.p50_latency_ms = _percentile(latencies, 0.5)
                call.p90_latency_ms = _percentile(latencies, 0.9)
        if not (call.summary or "").strip():
            transcript_result = await db.execute(
                select(TranscriptEntry)
                .where(TranscriptEntry.call_id == call_id)
                .order_by(TranscriptEntry.sequence.desc())
                .limit(8)
            )
            entries = list(reversed(transcript_result.scalars().all()))
            summary_parts = [
                f"{'User' if entry.speaker == 'You' else 'Aura'}: "
                f"{' '.join(entry.text.split())}"
                for entry in entries
                if entry.text
            ]
            call.summary = " | ".join(summary_parts)[:1200]
        call.status = status
        call.ended_at = now
        call.end_reason = end_reason[:64]
        call.ended_by = ended_by[:32]
        call.duration_ms = duration_ms
        call.updated_at = now
        await db.commit()
        logger.info(
            "call_lifecycle call_id={} status={} end_reason={} ended_by={} duration_ms={}",
            call_id,
            status,
            call.end_reason,
            call.ended_by,
            round(call.duration_ms or 0.0, 1),
        )
        return True


async def mark_call_ending(call_id: uuid.UUID | None) -> bool:
    """Move a live call into teardown without permitting it to be reopened."""
    if not call_id:
        return False
    async with AsyncSessionLocal() as db:
        call = await _locked_call(db, call_id)
        if not call or is_terminal(call.status) or call.status == "ending":
            return False
        call.status = "ending"
        call.updated_at = utcnow()
        await db.commit()
        return True


async def save_call_summary(call_id: uuid.UUID | None, summary: str) -> bool:
    from core.assistant_output import contains_reserved_tool_markup

    summary = (summary or "").strip()[:3000]
    if not call_id or not summary or contains_reserved_tool_markup(summary):
        return False
    async with AsyncSessionLocal() as db:
        call = await _locked_call(db, call_id)
        if not call or is_terminal(call.status):
            return False
        call.summary = summary
        call.updated_at = utcnow()
        await db.commit()
        return True


class CallEventRecorder:
    """Normalize one diagnostic for both durable history and the live UI."""

    def __init__(
        self,
        call_id: uuid.UUID,
        *,
        provider_context: dict[str, tuple[str | None, str | None]] | None = None,
        live_sender=None,
        turn_id_getter=None,
    ):
        self.call_id = call_id
        self.provider_context = provider_context or {}
        self.live_sender = live_sender
        self.turn_id_getter = turn_id_getter
        self._fallback_tasks: set[asyncio.Task] = set()
        self._live_tasks: set[asyncio.Task] = set()

    async def drain(self) -> None:
        pending = tuple(self._fallback_tasks | self._live_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def record_queue_rejection(self, task_name: str, lane: str) -> None:
        self.record(
            component="persistence",
            code="persistence.queue_rejected",
            severity="error",
            outcome="degraded",
            safe_message="A call record could not be queued for durable persistence.",
            retryable=True,
            details={"task": task_name, "lane": lane},
        )

    def record(
        self,
        *,
        component: str,
        code: str,
        severity: str = "error",
        outcome: str = "failed",
        safe_message: str,
        operator_detail: Any = None,
        request_id: str | None = None,
        duration_ms: float | None = None,
        retryable: bool = False,
        recovered: bool = False,
        fatal: bool = False,
        details: Any = None,
    ) -> dict[str, Any]:
        turn_id = self.turn_id_getter() if self.turn_id_getter else None
        provider, model = self.provider_context.get(component, (None, None))
        safe = sanitize_detail(safe_message)
        fingerprint = event_fingerprint(
            code=code,
            turn_id=turn_id,
            request_id=request_id,
            safe_message=safe,
        )
        payload = {
            "item_type": "event",
            "id": fingerprint,
            "sequence": None,
            "call_id": str(self.call_id),
            "turn_id": turn_id,
            "component": component,
            "code": code,
            "severity": severity,
            "outcome": outcome,
            "message": safe,
            "technical_detail": sanitize_detail(operator_detail) or None,
            "provider": provider,
            "model": model,
            "request_id": request_id,
            "duration_ms": duration_ms,
            "retryable": retryable,
            "recovered": recovered,
            "fatal": fatal,
            "details": _json_safe(sanitize_metadata(details or {}), max_chars=MAX_EVENT_DETAIL_CHARS),
            "fingerprint": fingerprint,
            "created_at": utcnow().isoformat(),
        }
        accepted = task_queue.enqueue(
            save_call_event,
            self.call_id,
            component=component,
            code=code,
            severity=severity,
            outcome=outcome,
            safe_message=safe,
            operator_detail=operator_detail,
            turn_id=turn_id,
            provider=provider,
            model=model,
            request_id=request_id,
            duration_ms=duration_ms,
            retryable=retryable,
            recovered=recovered,
            fatal=fatal,
            details=details,
            fingerprint=fingerprint,
            key=str(self.call_id),
        )
        if not accepted:
            logger.error(
                "call_event_queue status=rejected call_id={} code={}",
                self.call_id,
                code,
            )
            payload["persistence"] = "direct_fallback"
            fallback_task = asyncio.create_task(
                save_call_event(
                    self.call_id,
                    component=component,
                    code=code,
                    severity=severity,
                    outcome=outcome,
                    safe_message=safe,
                    operator_detail=operator_detail,
                    turn_id=turn_id,
                    provider=provider,
                    model=model,
                    request_id=request_id,
                    duration_ms=duration_ms,
                    retryable=retryable,
                    recovered=recovered,
                    fatal=fatal,
                    details=details,
                    fingerprint=fingerprint,
                ),
                name=f"call-event-fallback-{code}",
            )
            self._fallback_tasks.add(fallback_task)
            fallback_task.add_done_callback(self._fallback_tasks.discard)
        if self.live_sender:
            result = self.live_sender(payload)
            if asyncio.iscoroutine(result):
                live_task = asyncio.create_task(
                    result, name=f"call-event-live-{code}"
                )
                self._live_tasks.add(live_task)
                live_task.add_done_callback(self._live_tasks.discard)
        return payload


async def abandon_stale_calls(grace_seconds: int = 30) -> int:
    cutoff = utcnow() - timedelta(seconds=max(1, grace_seconds))
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Call).where(
                Call.status.in_(("initializing", "active", "ending")),
                Call.updated_at < cutoff,
            ).with_for_update(skip_locked=True)
        )
        calls = list(result.scalars().all())
        now = utcnow()
        for call in calls:
            call.status = "abandoned"
            call.ended_at = now
            call.end_reason = "worker_lost"
            call.ended_by = "watchdog"
            call.duration_ms = max(0.0, (now - call.started_at).total_seconds() * 1000)
        await db.commit()
        return len(calls)


async def soft_delete_call(call_id: uuid.UUID, user_id: int, retention_days: int = 30) -> bool:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Call).where(Call.id == call_id, Call.user_id == user_id).with_for_update()
        )
        call = result.scalars().first()
        if not call or not is_terminal(call.status):
            return False
        if call.deleted_at:
            return True
        now = utcnow()
        call.deleted_at = now
        call.purge_after = now + timedelta(days=max(1, retention_days))
        call.updated_at = now
        recording_result = await db.execute(
            select(CallRecording).where(CallRecording.call_id == call_id).with_for_update()
        )
        if recording := recording_result.scalars().first():
            recording.status = "deleted"
            recording.updated_at = now
        await db.commit()
        return True


async def restore_call(call_id: uuid.UUID, user_id: int) -> bool:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Call).where(Call.id == call_id, Call.user_id == user_id).with_for_update()
        )
        call = result.scalars().first()
        if (
            not call
            or not call.deleted_at
            or call.purge_started_at is not None
            or (call.purge_after and call.purge_after <= utcnow())
        ):
            return False
        call.deleted_at = None
        call.purge_after = None
        call.updated_at = utcnow()
        recording_result = await db.execute(
            select(CallRecording).where(CallRecording.call_id == call_id).with_for_update()
        )
        if recording := recording_result.scalars().first():
            recording.status = "available" if recording.object_key else "failed"
            recording.updated_at = utcnow()
        await db.commit()
        return True


async def due_recording_objects_for_purge(db: AsyncSession, now: datetime) -> list[str]:
    result = await db.execute(
        select(CallRecording.object_key)
        .join(Call, Call.id == CallRecording.call_id)
        .where(Call.purge_after.is_not(None), Call.purge_after <= now)
    )
    return [key for key in result.scalars().all() if key]


async def purge_expired_call_rows(db: AsyncSession, now: datetime | None = None) -> int:
    now = now or utcnow()
    result = await db.execute(
        delete(Call).where(Call.purge_after.is_not(None), Call.purge_after <= now).returning(Call.id)
    )
    return len(result.scalars().all())


async def purge_expired_call_row(db: AsyncSession, call_id: uuid.UUID, now: datetime | None = None) -> bool:
    now = now or utcnow()
    result = await db.execute(
        delete(Call)
        .where(
            Call.id == call_id,
            Call.purge_after.is_not(None),
            Call.purge_after <= now,
            Call.purge_started_at.is_not(None),
        )
        .returning(Call.id)
    )
    return result.scalar_one_or_none() is not None
