"""Summarize persisted voice-latency JSONL into percentile groups."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_METRICS = (
    "answer_audio_ms",
    "user_stop_to_playback_ms",
    "stt_latency_ms",
    "llm_latency_ms",
    "tts_latency_ms",
    "llm_ms",
    "tts_aggregation_ms",
    "tts_provider_ms",
    "endpointing_ms",
    "rtt_ms",
)


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return round(ordered[index], 1)


def summarize_records(records: list[dict]) -> dict:
    # The server writes a guaranteed generation-side record; a browser may
    # later add actual playback/WebRTC metrics for the same turn. Prefer that
    # richer client record without counting the turn twice.
    deduplicated: dict[tuple, dict] = {}
    unkeyed: list[dict] = []
    for record in records:
        turn_id = record.get("turn_id")
        session_id = record.get("session_id")
        if turn_id is None or session_id is None:
            unkeyed.append(record)
            continue
        key = (record.get("user_id"), session_id, turn_id)
        current = deduplicated.get(key)
        source_rank = 2 if record.get("measurement_source") == "client" else 1
        current_rank = (
            2 if current and current.get("measurement_source") == "client" else 1
        )
        if current is None or source_rank >= current_rank:
            deduplicated[key] = record
    records = [*unkeyed, *deduplicated.values()]

    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        category = record.get("category", "unknown")
        warmth = "warm" if record.get("llm_connection_warmed") else "cold"
        groups[f"{category}:{warmth}"].append(record)

    report = {}
    for group, rows in sorted(groups.items()):
        metrics = {}
        for metric in DEFAULT_METRICS:
            values = [
                float(row[metric])
                for row in rows
                if isinstance(row.get(metric), (int, float))
            ]
            if values:
                metrics[metric] = {
                    "count": len(values),
                    "min": round(min(values), 1),
                    "p50": percentile(values, 0.50),
                    "p95": percentile(values, 0.95),
                    "p99": percentile(values, 0.99),
                    "max": round(max(values), 1),
                }
        report[group] = {"turns": len(rows), "metrics": metrics}
    return report


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if isinstance(value, dict):
                records.append(value)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report p50/p95/p99 voice latency by category and warm state."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="logs/voice-latency.jsonl",
        type=Path,
    )
    args = parser.parse_args()
    print(json.dumps(summarize_records(load_jsonl(args.path)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
