import asyncio

import pytest

from services.knowledge.units import UnitInput, unit_content_hash, upsert_draft_unit


def make_unit(**overrides):
    values = {
        "stable_key": "mswipe.activation",
        "unit_type": "procedure",
        "title": "Activate a terminal",
        "answer": "Follow the activation steps.",
        "retrieval_text": "Mswipe POS terminal activation steps",
        "source_uri": "https://www.mswipe.com/activation",
        "source_label": "Mswipe activation",
    }
    values.update(overrides)
    return UnitInput(**values)


def test_unit_hash_changes_with_answer_and_not_metadata():
    first = unit_content_hash(make_unit(metadata={"reviewer": "one"}))
    same_content = unit_content_hash(make_unit(metadata={"reviewer": "two"}))
    changed = unit_content_hash(make_unit(answer="A revised approved answer."))
    assert first == same_content
    assert changed != first


def test_unit_validation_runs_before_database_access():
    async def exercise():
        with pytest.raises(ValueError, match="Unsupported"):
            await upsert_draft_unit(object(), make_unit(unit_type="arbitrary_chunk"))
        with pytest.raises(ValueError, match="stable_key"):
            await upsert_draft_unit(object(), make_unit(stable_key="Bad Key"))
        with pytest.raises(ValueError, match="authority"):
            await upsert_draft_unit(object(), make_unit(authority=8))

    asyncio.run(exercise())
