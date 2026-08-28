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
        if not case.get("query") or not case.get("expected_route"):
            raise ValueError(f"Case {line_number} requires query and expected_route")
        cases.append(case)
    if not cases:
        raise ValueError("Evaluation dataset contains no cases")
    return cases


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


async def evaluate(cases: list[dict]) -> dict:
    route_correct = retrieval_cases = hits = reciprocal_rank = 0.0
    no_answer_cases = no_answer_correct = 0
    latencies: list[float] = []
    failures: list[dict] = []
    for case in cases:
        started = time.monotonic()
        response = await retrieve_knowledge(case["query"])
        latencies.append((time.monotonic() - started) * 1000)
        route_ok = response.route.name == case["expected_route"]
        route_correct += int(route_ok)
        expected_keys = list(case.get("expected_unit_keys") or [])
        expect_no_answer = bool(case.get("expect_no_answer", False))
        rank = None
        if expected_keys:
            retrieval_cases += 1
            returned = [hit.stable_key for hit in response.hits]
            rank = next((index for index, key in enumerate(returned, start=1) if key in expected_keys), None)
            if rank:
                hits += 1
                reciprocal_rank += 1 / rank
        if expect_no_answer:
            no_answer_cases += 1
            no_answer_correct += int(response.status != "ok")
        if not route_ok or (expected_keys and rank is None) or (expect_no_answer and response.status == "ok"):
            failures.append({
                "query": case["query"],
                "expected_route": case["expected_route"],
                "actual_route": response.route.name,
                "status": response.status,
                "returned": [hit.stable_key for hit in response.hits],
            })
    return {
        "cases": len(cases),
        "route_accuracy": route_correct / len(cases),
        "recall_at_k": hits / retrieval_cases if retrieval_cases else None,
        "mrr": reciprocal_rank / retrieval_cases if retrieval_cases else None,
        "no_answer_accuracy": no_answer_correct / no_answer_cases if no_answer_cases else None,
        "latency_ms": {
            "p50": round(statistics.median(latencies), 2),
            "p95": round(percentile(latencies, 0.95), 2),
            "max": round(max(latencies), 2),
        },
        "failures": failures,
    }


async def main() -> None:
    arguments = argparse.ArgumentParser()
    arguments.add_argument("dataset", type=Path)
    arguments.add_argument("--min-route-accuracy", type=float, default=0.95)
    arguments.add_argument("--min-recall", type=float, default=0.85)
    args = arguments.parse_args()
    try:
        report = await evaluate(load_cases(args.dataset))
        print(json.dumps(report, indent=2))
        failed = report["route_accuracy"] < args.min_route_accuracy
        if report["recall_at_k"] is not None:
            failed = failed or report["recall_at_k"] < args.min_recall
        if failed:
            raise SystemExit(1)
    finally:
        await engine.dispose()
        await voice_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
