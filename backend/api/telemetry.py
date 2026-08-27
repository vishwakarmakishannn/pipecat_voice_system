from typing import Literal
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from api.auth import get_current_user
from core.database import get_db
from core.models import Call, User
from core.task_queue import task_queue
from services.calls import save_call_turn
from services.latency_telemetry import persist_voice_latency
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])


class VoiceLatencyTelemetry(BaseModel):
    """Transcript-free browser/server timing record for one response."""

    model_config = ConfigDict(extra="forbid")

    call_id: uuid.UUID
    session_id: str | None = Field(default=None, max_length=200)
    turn_id: int = Field(ge=0)
    turn_started_unix_ms: int | None = Field(default=None, ge=0)
    input_mode: Literal["voice", "text"] = "voice"
    latency_stage: Literal["stt", "llm", "tts"] | None = None
    latency_complete: bool = False
    measurement_source: Literal["server", "client"] = "client"
    category: Literal["direct", "rag", "tool"]
    basis: Literal["user_stopped", "final_stt"] | None = None
    with_tools: bool = False
    rag_used: bool = False
    rag_considered: bool = False
    rag_bypassed: bool = False
    interrupted: bool = False
    outcome: Literal["completed", "cancelled", "recovered", "degraded"] = "completed"
    stt_provider: str | None = Field(default=None, max_length=40)
    stt_model: str | None = Field(default=None, max_length=120)
    llm_provider: str | None = Field(default=None, max_length=40)
    llm_model: str | None = Field(default=None, max_length=120)
    tts_provider: str | None = Field(default=None, max_length=40)
    tts_model: str | None = Field(default=None, max_length=120)
    llm_connection_warmed: bool = False

    stt_latency_ms: float | None = Field(default=None, ge=0, le=300000)
    llm_latency_ms: float | None = Field(default=None, ge=0, le=300000)
    response_preparation_ms: float | None = Field(default=None, ge=0, le=300000)
    server_endpointing_ms: float | None = Field(default=None, ge=0, le=300000)
    turn_release_ms: float | None = Field(default=None, ge=0, le=300000)
    pre_llm_ms: float | None = Field(default=None, ge=0, le=300000)
    llm_ttft_ms: float | None = Field(default=None, ge=0, le=300000)
    tts_latency_ms: float | None = Field(default=None, ge=0, le=300000)
    tool_latency_ms: float | None = Field(default=None, ge=0, le=300000)
    rag_latency_ms: float | None = Field(default=None, ge=0, le=300000)
    llm_ms: float | None = Field(default=None, ge=-5000, le=300000)
    speakable_text_ms: float | None = Field(default=None, ge=-5000, le=300000)
    tts_aggregation_ms: float | None = Field(default=None, ge=-5000, le=300000)
    tts_provider_ms: float | None = Field(default=None, ge=-5000, le=300000)
    speakable_to_audio_ms: float | None = Field(default=None, ge=-5000, le=300000)
    answer_audio_ms: float | None = Field(default=None, ge=-5000, le=300000)
    final_stt_to_audio_ms: float | None = Field(default=None, ge=-5000, le=300000)
    speech_ms: float | None = Field(default=None, ge=0, le=3600000)
    interim_stt_count: int = Field(default=0, ge=0, le=100000)
    final_stt_fragment_count: int = Field(default=0, ge=0, le=100000)
    stt_finalization_ms: dict[str, float] = Field(default_factory=dict)
    vad_diagnostics: dict[str, float] = Field(default_factory=dict)
    stages_ms: dict[str, float] = Field(default_factory=dict)

    client_message_to_audio_ms: float | None = Field(default=None, ge=0, le=300000)
    user_stop_to_playback_ms: float | None = Field(default=None, ge=0, le=300000)
    text_send_to_playback_ms: float | None = Field(default=None, ge=0, le=300000)
    turn_stop_signal_to_playback_ms: float | None = Field(
        default=None, ge=0, le=300000
    )
    endpointing_ms: float | None = Field(default=None, ge=0, le=300000)
    client_speech_ms: float | None = Field(default=None, ge=0, le=3600000)
    tts_signal_to_playback_ms: float | None = Field(default=None, ge=0, le=300000)
    webrtc_jitter_ms: float | None = Field(default=None, ge=0, le=300000)
    jitter_buffer_avg_ms: float | None = Field(default=None, ge=0, le=300000)
    rtt_ms: float | None = Field(default=None, ge=0, le=300000)
    packets_lost: int | None = None
    packets_received: int | None = Field(default=None, ge=0)
    concealed_samples: int | None = Field(default=None, ge=0)
    concealment_events: int | None = Field(default=None, ge=0)

    # Actual browser microphone settings, captured after constraints are
    # applied. These are transcript-free and let accuracy regressions be
    # correlated with browser DSP without changing the live media path.
    capture_reported_latency_ms: float | None = Field(
        default=None, ge=0, le=300000
    )
    capture_sample_rate: int | None = Field(default=None, ge=8000, le=384000)
    capture_channel_count: int | None = Field(default=None, ge=1, le=32)
    capture_echo_cancellation: bool | None = None
    capture_noise_suppression: bool | None = None
    capture_auto_gain_control: bool | None = None

    server_emitted_unix_ms: int | None = Field(default=None, ge=0)
    client_received_unix_ms: int | None = Field(default=None, ge=0)
    playback_detected_unix_ms: int | None = Field(default=None, ge=0)
    playback_signal: str | None = Field(default=None, max_length=80)
    speech_end_signal: str | None = Field(default=None, max_length=80)


@router.post("/voice-latency", status_code=status.HTTP_202_ACCEPTED)
async def record_voice_latency(
    telemetry: VoiceLatencyTelemetry,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Call.status).where(
            Call.id == telemetry.call_id,
            Call.user_id == current_user.id,
            Call.deleted_at.is_(None),
        )
    )
    call_status = result.scalar_one_or_none()
    if call_status is None:
        raise HTTPException(status_code=404, detail="Call not found")
    if call_status not in {"initializing", "active", "ending"}:
        raise HTTPException(status_code=409, detail="Terminal calls are immutable")
    payload = telemetry.model_dump(mode="json")
    call_turn_accepted = task_queue.enqueue(
        save_call_turn,
        telemetry.call_id,
        payload,
        key=str(telemetry.call_id),
    )
    jsonl_accepted = task_queue.enqueue(
        persist_voice_latency,
        current_user.id,
        payload,
        key=str(telemetry.call_id),
    )
    return {"accepted": call_turn_accepted and jsonl_accepted}
