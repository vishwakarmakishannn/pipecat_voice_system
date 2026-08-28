import uuid

from core.database import Base
from core.models import (
    KNOWLEDGE_UNIT_TYPES,
    KnowledgeRelease,
    KnowledgeUnit,
    TicketTaxonomyEntry,
)


def test_independent_knowledge_schema_is_registered():
    expected = {
        "knowledge_sources",
        "knowledge_snapshots",
        "knowledge_documents",
        "knowledge_units",
        "knowledge_embeddings",
        "knowledge_aliases",
        "knowledge_releases",
        "knowledge_release_units",
        "knowledge_conflicts",
        "knowledge_jobs",
        "knowledge_feedback",
        "ticket_taxonomy_entries",
    }
    assert expected <= set(Base.metadata.tables)
    assert "rag_files" not in {
        foreign_key.column.table.name
        for name in expected
        for foreign_key in Base.metadata.tables[name].foreign_keys
    }


def test_release_has_one_published_partial_unique_index():
    index = next(
        item for item in KnowledgeRelease.__table__.indexes if item.name == "uq_krel_one_published"
    )
    assert index.unique is True
    assert str(index.dialect_options["postgresql"]["where"]) == "status = 'published'"


def test_typed_unit_and_taxonomy_defaults_are_controlled():
    assert "procedure" in KNOWLEDGE_UNIT_TYPES
    assert "ticket_taxonomy" in KNOWLEDGE_UNIT_TYPES
    unit = KnowledgeUnit(
        stable_key="test.activation",
        unit_type="procedure",
        title="Activation",
        answer="Follow the approved activation process.",
        retrieval_text="Mswipe POS activation",
        source_uri="https://www.mswipe.com/activation",
        source_label="Mswipe activation",
        content_hash="a" * 64,
    )
    taxonomy = TicketTaxonomyEntry(
        ticket_code="Device",
        ticket_subcode="Activation",
        remark="Activation pending",
        content_hash="b" * 64,
    )
    assert unit.status is None  # SQLAlchemy applies scalar defaults at INSERT time.
    assert taxonomy.active is None
    assert isinstance(uuid.uuid4(), uuid.UUID)
