"""Generate a reproducible baseline containing no credential or customer values."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform

from core.database import AsyncSessionLocal, engine, voice_engine
from knowledge_cli import _corpus_audit


SAFE_CONFIGURATION_KEYS = (
    "APP_VERSION",
    "LLM_PROVIDER",
    "GOOGLE_MODEL",
    "GROQ_MODEL",
    "OPENAI_MODEL",
    "STT_PROVIDER",
    "DEEPGRAM_STT_MODEL",
    "DEEPGRAM_STT_LANGUAGE",
    "DEEPGRAM_STT_NUMERALS",
    "DEEPGRAM_STT_SMART_FORMAT",
    "TTS_PROVIDER",
    "WEB_SEARCH_ENABLED",
    "TURN_STOP_STRATEGY",
    "MSWIPE_ISSUE_CONTRACT_VERSION",
    "MSWIPE_KNOWLEDGE_ENABLED",
    "MSWIPE_KNOWLEDGE_EMBEDDING_PROVIDER",
    "MSWIPE_KNOWLEDGE_EMBEDDING_MODEL",
    "MSWIPE_KNOWLEDGE_CHUNK_POLICY_VERSION",
)


def safe_configuration() -> dict[str, str | None]:
    return {name: os.getenv(name) for name in SAFE_CONFIGURATION_KEYS}


def configuration_hash(configuration: dict[str, str | None]) -> str:
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


async def build_report() -> dict:
    configuration = safe_configuration()
    async with AsyncSessionLocal() as db:
        corpus = await _corpus_audit(db)
    return {
        "schema_version": "safe-baseline-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": platform.python_version()},
        "configuration": configuration,
        "configuration_hash": configuration_hash(configuration),
        "corpus": corpus,
        "privacy": {
            "contains_credentials": False,
            "contains_raw_customer_pii": False,
            "contains_raw_queries": False,
        },
    }


async def main() -> None:
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--output", type=Path)
    args = arguments.parse_args()
    try:
        payload = json.dumps(await build_report(), indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
            print(json.dumps({"output": str(args.output), "status": "created"}))
        else:
            print(payload)
    finally:
        await engine.dispose()
        await voice_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
