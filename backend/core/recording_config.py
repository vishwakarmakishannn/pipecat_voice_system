"""Configuration for durable, private call recordings."""

from __future__ import annotations

import os
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
RECORDING_SAMPLE_RATE = 16_000
RECORDING_CHANNELS = 1
RECORDING_BIT_RATE = 64_000
RECORDING_POLICY_VERSION = "always-on-v1"


def recording_spool_dir() -> Path:
    return Path(
        os.getenv("RECORDING_SPOOL_DIR", str(BACKEND_ROOT / "recording-spool"))
    ).expanduser().resolve()


def local_recording_dir() -> Path:
    return Path(
        os.getenv("RECORDING_STORAGE_DIR", str(BACKEND_ROOT / "recordings"))
    ).expanduser().resolve()


def recording_queue_chunks() -> int:
    return max(2, int(os.getenv("RECORDING_QUEUE_CHUNKS", "12")))


def recording_access_ttl_seconds() -> int:
    return max(30, min(int(os.getenv("RECORDING_ACCESS_TTL_SECONDS", "300")), 3600))

