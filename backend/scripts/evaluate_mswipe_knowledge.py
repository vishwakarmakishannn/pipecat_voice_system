"""Offline release gate for Mswipe retrieval and routing quality."""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from core.database import engine, voice_engine
from services.knowledge.retrieval import retrieve_knowledge


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
        if not case.get("query") or not case.get("expected_status"):
            raise ValueError(f"Case {line_number} requires query and expected_status")
        cases.append(case)
    if not cases:
        raise ValueError("Evaluation dataset contains no cases")
    return cases


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


async def evaluate(cases: list[dict]) -> dict:
    status_correct = retrieval_cases = hits = reciprocal_rank = 0.0
    no_answer_cases = no_answer_correct = 0
    latencies: list[float] = []
    failures: list[dict] = []
    retrieval_modes: dict[str, int] = {}
    for case in cases:
        started = time.monotonic()
        response = await retrieve_knowledge(
            case["query"],
            answer_type=case.get("answer_type", "fact"),
            requires_live_data=bool(case.get("requires_live_data", False)),
        )
        latencies.append((time.monotonic() - started) * 1000)
        retrieval_modes[response.retrieval_mode] = (
            retrieval_modes.get(response.retrieval_mode, 0) + 1
        )
        status_ok = response.status == case["expected_status"]
        status_correct += int(status_ok)
        expected_keys = list(case.get("expected_unit_keys") or [])
        expected_sources = list(case.get("expected_source_contains") or [])
        expect_no_answer = case["expected_status"] == "no_answer"
        rank = None
        if expected_keys or expected_sources:
            retrieval_cases += 1
            rank = next(
                (
                    index
                    for index, hit in enumerate(response.hits, start=1)
                    if hit.stable_key in expected_keys
                    or any(source in hit.source_uri for source in expected_sources)
                ),
                None,
            )
            if rank:
                hits += 1
                reciprocal_rank += 1 / rank
        if expect_no_answer:
            no_answer_cases += 1
            no_answer_correct += int(response.status != "ok")
        if not status_ok or ((expected_keys or expected_sources) and rank is None):
            failures.append({
                "case_id": case.get("id"),
                "expected_status": case["expected_status"],
                "status": response.status,
                "returned": [hit.stable_key for hit in response.hits],
            })
    return {
        "cases": len(cases),
        "status_accuracy": status_correct / len(cases),
        "recall_at_k": hits / retrieval_cases if retrieval_cases else None,
        "mrr": reciprocal_rank / retrieval_cases if retrieval_cases else None,
        "no_answer_accuracy": no_answer_correct / no_answer_cases if no_answer_cases else None,
        "latency_ms": {
            "p50": round(statistics.median(latencies), 2),
            "p95": round(percentile(latencies, 0.95), 2),
            "max": round(max(latencies), 2),
        },
        "retrieval_modes": retrieval_modes,
        "failures": failures,
    }


async def main() -> None:
    arguments = argparse.ArgumentParser()
    arguments.add_argument("dataset", type=Path)
    arguments.add_argument("--min-status-accuracy", type=float, default=0.95)
    arguments.add_argument("--min-recall", type=float, default=0.85)
    arguments.add_argument("--max-p95-ms", type=float, default=500.0)
    args = arguments.parse_args()
    try:
        report = await evaluate(load_cases(args.dataset))
        print(json.dumps(report, indent=2))
        failed = report["status_accuracy"] < args.min_status_accuracy
        if report["recall_at_k"] is not None:
            failed = failed or report["recall_at_k"] < args.min_recall
        failed = failed or report["latency_ms"]["p95"] > args.max_p95_ms
        if failed:
            raise SystemExit(1)
    finally:
        await engine.dispose()
        await voice_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
