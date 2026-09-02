import asyncio

import pytest

from services.knowledge.units import (
    UnitInput,
    approve_unit,
    unit_content_hash,
    upsert_draft_unit,
)


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


def test_approval_records_reviewed_direct_answer_metadata():
    class Result:
        @staticmethod
        def first():
            return unit

    class ScalarResult:
        @staticmethod
        def scalars():
            return Result()

    class Database:
        async def execute(self, _query):
            return ScalarResult()

        async def commit(self):
            return None

        async def refresh(self, _unit):
            return None

    unit = type("Unit", (), {})()
    unit.status = "draft"
    unit.voice_answer = None
    unit.metadata_json = {}
    unit.approved_by_user_id = None
    unit.approved_at = None
    unit.review_notes = None
    unit.unit_type = "faq"
    unit.title = "Question"
    unit.question = "Question?"
    unit.answer = "A concise reviewed answer."
    unit.retrieval_text = "Question and answer"
    unit.source_uri = "https://www.mswipe.com/help"
    unit.content_hash = "old"

    approved = asyncio.run(
        approve_unit(
            Database(),
            __import__("uuid").uuid4(),
            voice_answer="  A concise reviewed answer.  ",
            atomic_answer=True,
            answerability_reviewed=True,
        )
    )

    assert approved.voice_answer == "A concise reviewed answer."
    assert approved.metadata_json == {
        "atomic_answer": True,
        "answerability_reviewed": True,
        "voice_answer_approved": True,
    }
    assert approved.content_hash != "old"
