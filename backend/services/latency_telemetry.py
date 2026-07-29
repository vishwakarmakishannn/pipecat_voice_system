"""Best-effort durable storage for privacy-safe voice latency telemetry."""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path


_write_lock = asyncio.Lock()


def latency_telemetry_path() -> Path:
    return Path(
        os.getenv("VOICE_LATENCY_JSONL_PATH", "logs/voice-latency.jsonl")
    ).expanduser()


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"), sort_keys=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(line)
        output.write("\n")


async def persist_voice_latency(user_id: int, payload: dict) -> None:
    record = {
        "schema_version": 1,
        "user_id": user_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    async with _write_lock:
        await asyncio.to_thread(_append_jsonl, latency_telemetry_path(), record)
