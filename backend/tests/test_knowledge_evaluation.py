import json

import pytest

from scripts.evaluate_mswipe_knowledge import load_cases, percentile


def test_demo_evaluation_cases_have_ids_labels_and_no_placeholders():
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "evals" / "mswipe_knowledge_cases.demo.jsonl"
    cases = load_cases(path)

    assert len(cases) >= 8
    assert len({case["id"] for case in cases}) == len(cases)
    assert {case["expected_status"] for case in cases} == {"ok", "no_answer"}
    assert "replace-with" not in json.dumps(cases)
    assert any(case.get("language") == "hinglish" for case in cases)
    assert any(case.get("kind") == "requires_live_api" for case in cases)
    assert any("hard_negative" in case.get("kind", "") for case in cases)


def test_evaluation_dataset_rejects_missing_status(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"query":"Question"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="expected_status"):
        load_cases(path)


def test_percentile_is_deterministic():
    assert percentile([30, 10, 20, 40], 0.95) == 40
