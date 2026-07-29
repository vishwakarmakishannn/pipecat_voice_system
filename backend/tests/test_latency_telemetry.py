import json

import pytest

from api.telemetry import VoiceLatencyTelemetry
from scripts.summarize_voice_latency import percentile, summarize_records
from services import latency_telemetry


def test_voice_latency_schema_rejects_transcript_content():
    with pytest.raises(ValueError):
        VoiceLatencyTelemetry(
            turn_id=1,
            category="direct",
            transcript="private user speech",
        )


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
            },
            {
                "category": "direct",
                "llm_connection_warmed": True,
                "answer_audio_ms": 600,
            },
            {
                "category": "rag",
                "llm_connection_warmed": False,
                "answer_audio_ms": 900,
            },
        ]
    )

    assert report["direct:warm"]["metrics"]["answer_audio_ms"]["p50"] == 400.0
    assert report["direct:warm"]["metrics"]["answer_audio_ms"]["p95"] == 600.0
    assert report["rag:cold"]["turns"] == 1
    assert percentile([], 0.5) is None
