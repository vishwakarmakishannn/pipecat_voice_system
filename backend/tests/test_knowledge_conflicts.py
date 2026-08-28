from types import SimpleNamespace
from uuid import uuid4

from services.knowledge.conflicts import _cross_document_pair


def test_sections_from_one_document_are_not_conflict_candidates():
    document_id = uuid4()
    left = SimpleNamespace(document_id=document_id)
    right = SimpleNamespace(document_id=document_id)
    assert not _cross_document_pair(left, right)


def test_separate_or_manual_units_can_be_conflict_candidates():
    assert _cross_document_pair(
        SimpleNamespace(document_id=uuid4()),
        SimpleNamespace(document_id=uuid4()),
    )
    assert _cross_document_pair(
        SimpleNamespace(document_id=None),
        SimpleNamespace(document_id=None),
    )
