import json
import uuid

import pytest

from api.telemetry import VoiceLatencyTelemetry
from scripts.summarize_voice_latency import percentile, summarize_records
from services import latency_telemetry
from services.calls import (
    summarize_direct_perceived_latency,
    summarize_perceived_latency,
    summarize_stt_finalization,
)


def test_voice_latency_schema_rejects_transcript_content():
    with pytest.raises(ValueError):
        VoiceLatencyTelemetry(
            call_id=uuid.uuid4(),
            turn_id=1,
            category="direct",
            transcript="private user speech",
        )


def test_voice_latency_schema_accepts_stt_identity():
    telemetry = VoiceLatencyTelemetry(
        call_id=uuid.uuid4(),
        turn_id=1,
        category="direct",
        stt_provider="whisper",
        stt_model="small",
    )

    assert telemetry.stt_provider == "whisper"
    assert telemetry.stt_model == "small"


def test_voice_latency_schema_accepts_server_input_mode():
    telemetry = VoiceLatencyTelemetry(
        call_id=uuid.uuid4(),
        turn_id=1,
        input_mode="text",
        category="direct",
    )

    assert telemetry.input_mode == "text"


def test_voice_latency_schema_accepts_capture_and_vad_diagnostics():
    telemetry = VoiceLatencyTelemetry(
        call_id=uuid.uuid4(),
        turn_id=1,
        category="direct",
        capture_reported_latency_ms=12.5,
        capture_sample_rate=48000,
        capture_channel_count=1,
        capture_echo_cancellation=True,
        capture_noise_suppression=True,
        capture_auto_gain_control=True,
        vad_diagnostics={"confidence_at_stop": 0.12},
    )

    assert telemetry.capture_sample_rate == 48000
    assert telemetry.capture_noise_suppression is True
    assert telemetry.vad_diagnostics["confidence_at_stop"] == 0.12


def test_voice_latency_schema_accepts_pipeline_stage_breakdown():
    telemetry = VoiceLatencyTelemetry(
        call_id=uuid.uuid4(),
        turn_id=1,
        latency_stage="tts",
        latency_complete=True,
        category="direct",
        stt_latency_ms=415.1,
        llm_latency_ms=108.7,
        response_preparation_ms=108.7,
        server_endpointing_ms=430.0,
        turn_release_ms=14.9,
        pre_llm_ms=4.2,
        llm_ttft_ms=72.1,
        tts_latency_ms=3089.2,
        answer_audio_ms=3613.0,
        stt_finalization_ms={
            "force_queue_ms": 2.1,
            "force_update_ms": 91.4,
            "vad_downstream_ms": 418.2,
        },
    )

    assert telemetry.stt_latency_ms == 415.1
    assert telemetry.llm_latency_ms == 108.7
    assert telemetry.response_preparation_ms == 108.7
    assert telemetry.server_endpointing_ms == 430.0
    assert telemetry.turn_release_ms == 14.9
    assert telemetry.pre_llm_ms == 4.2
    assert telemetry.llm_ttft_ms == 72.1
    assert telemetry.tts_latency_ms == 3089.2
    assert telemetry.latency_stage == "tts"
    assert telemetry.latency_complete is True
    assert telemetry.measurement_source == "client"
    assert telemetry.stt_finalization_ms["force_update_ms"] == 91.4


@pytest.mark.anyio
async def test_voice_latency_jsonl_is_persisted(monkeypatch, tmp_path):
    output = tmp_path / "voice.jsonl"
    monkeypatch.setattr(latency_telemetry, "latency_telemetry_path", lambda: output)

    await latency_telemetry.persist_voice_latency(
        7,
        {"turn_id": 3, "category": "direct", "answer_audio_ms": 412.5},
    )

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["schema_version"] == 1
    assert record["user_id"] == 7
    assert record["turn_id"] == 3
    assert record["answer_audio_ms"] == 412.5


def test_latency_summary_separates_category_and_warmth():
    report = summarize_records(
        [
            {
                "category": "direct",
                "llm_connection_warmed": True,
                "answer_audio_ms": 400,
                "stt_latency_ms": 100,
                "llm_latency_ms": 100,
                "tts_latency_ms": 200,
                "stt_finalization_ms": {"fallback_forced": 0},
            },
            {
                "category": "direct",
                "llm_connection_warmed": True,
                "answer_audio_ms": 600,
                "stt_finalization_ms": {
                    "fallback_forced": 1,
                    "final_shorter_than_interim": 1,
                },
            },
            {
                "category": "rag",
                "llm_connection_warmed": False,
                "answer_audio_ms": 900,
            },
        ]
    )

    assert report["direct:warm"]["metrics"]["answer_audio_ms"]["p50"] == 500.0
    assert report["direct:warm"]["metrics"]["answer_audio_ms"]["p90"] == 580.0
    assert report["direct:warm"]["metrics"]["answer_audio_ms"]["p95"] == 590.0
    assert report["direct:warm"]["metrics"]["stt_latency_ms"]["p50"] == 100.0
    assert report["direct:warm"]["metrics"]["llm_latency_ms"]["p50"] == 100.0
    assert report["direct:warm"]["metrics"]["tts_latency_ms"]["p50"] == 200.0
    assert report["rag:cold"]["turns"] == 1
    assert report["direct:all"]["turns"] == 2
    assert report["direct:all"]["stt_finalization"] == {
        "count": 2,
        "native_final_count": 1,
        "fallback_count": 1,
        "fallback_rate_pct": 50.0,
        "final_shorter_count": 1,
    }
    assert percentile([], 0.5) is None


def test_latency_summary_prefers_client_record_without_double_counting():
    base = {
        "user_id": 7,
        "session_id": "session-a",
        "turn_id": 3,
        "category": "direct",
        "llm_connection_warmed": True,
        "answer_audio_ms": 400,
    }
    report = summarize_records(
        [
            {**base, "measurement_source": "server"},
            {
                **base,
                "measurement_source": "client",
                "user_stop_to_playback_ms": 475,
            },
        ]
    )

    assert report["direct:warm"]["turns"] == 1
    assert (
        report["direct:warm"]["metrics"]["user_stop_to_playback_ms"]["p50"]
        == 475.0
    )


def test_direct_perceived_summary_uses_only_eligible_browser_turns():
    base = {
        "measurement_source": "client",
        "latency_complete": True,
        "input_mode": "voice",
        "category": "direct",
        "interrupted": False,
        "outcome": "completed",
    }
    report = summarize_direct_perceived_latency(
        [
            {**base, "user_stop_to_playback_ms": 800},
            {**base, "user_stop_to_playback_ms": 1000},
            {**base, "user_stop_to_playback_ms": 1200},
            {**base, "category": "tool", "user_stop_to_playback_ms": 5000},
            {**base, "input_mode": "text", "user_stop_to_playback_ms": 100},
            {**base, "interrupted": True, "user_stop_to_playback_ms": 200},
        ]
    )

    assert report == {
        "count": 3,
        "average_ms": 1000.0,
        "p50_ms": 1000.0,
        "p90_ms": 1160.0,
    }


def test_latency_cohorts_and_stt_finalization_are_category_scoped():
    base = {
        "measurement_source": "client",
        "latency_complete": True,
        "input_mode": "voice",
        "interrupted": False,
        "outcome": "completed",
    }
    records = [
        {
            **base,
            "category": "direct",
            "user_stop_to_playback_ms": 800,
            "stt_finalization_ms": {"fallback_forced": 0},
        },
        {
            **base,
            "category": "rag",
            "user_stop_to_playback_ms": 1600,
            "stt_finalization_ms": {
                "fallback_forced": 1,
                "final_shorter_than_interim": 1,
            },
        },
        {
            **base,
            "category": "tool",
            "user_stop_to_playback_ms": 4000,
            "stt_finalization_ms": {"fallback_forced": 0},
        },
    ]

    assert summarize_perceived_latency(records, category="rag") == {
        "count": 1,
        "average_ms": 1600.0,
        "p50_ms": 1600.0,
        "p90_ms": 1600.0,
    }
    assert summarize_stt_finalization(records, category="rag") == {
        "stt_finalization_count": 1,
        "native_final_count": 0,
        "fallback_count": 1,
        "fallback_rate_pct": 100.0,
        "final_shorter_count": 1,
    }
