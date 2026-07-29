from typing import Literal

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, ConfigDict, Field

from api.auth import get_current_user
from core.models import User
from core.task_queue import task_queue
from services.latency_telemetry import persist_voice_latency


router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])


class VoiceLatencyTelemetry(BaseModel):
    """Transcript-free browser/server timing record for one response."""

    model_config = ConfigDict(extra="forbid")

    session_id: str | None = Field(default=None, max_length=200)
    turn_id: int = Field(ge=0)
    category: Literal["direct", "rag", "tool"]
    basis: Literal["user_stopped", "final_stt"] | None = None
    with_tools: bool = False
    rag_used: bool = False
    rag_considered: bool = False
    rag_bypassed: bool = False
    llm_provider: str | None = Field(default=None, max_length=40)
    llm_model: str | None = Field(default=None, max_length=120)
    tts_provider: str | None = Field(default=None, max_length=40)
    tts_model: str | None = Field(default=None, max_length=120)
    llm_connection_warmed: bool = False

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

    server_emitted_unix_ms: int | None = Field(default=None, ge=0)
    client_received_unix_ms: int | None = Field(default=None, ge=0)
    playback_detected_unix_ms: int | None = Field(default=None, ge=0)
    playback_signal: str | None = Field(default=None, max_length=80)
    speech_end_signal: str | None = Field(default=None, max_length=80)


@router.post("/voice-latency", status_code=status.HTTP_202_ACCEPTED)
async def record_voice_latency(
    telemetry: VoiceLatencyTelemetry,
    current_user: User = Depends(get_current_user),
):
    accepted = task_queue.enqueue(
        persist_voice_latency,
        current_user.id,
        telemetry.model_dump(mode="json"),
        key=f"voice-latency-{current_user.id}",
    )
    return {"accepted": accepted}
