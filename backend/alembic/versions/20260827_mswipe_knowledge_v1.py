"""add independent production Mswipe knowledge subsystem

Revision ID: 20260827_mswipe_knowledge_v1
Revises: 20260814_structured_rag_v2
"""

from collections.abc import Sequence

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa

revision: str = "20260827_mswipe_knowledge_v1"
down_revision: str | Sequence[str] | None = "20260814_structured_rag_v2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "knowledge_sources",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("canonical_uri", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("authority", sa.Integer(), nullable=False),
        sa.Column("audience", sa.String(64), nullable=False),
        sa.Column("language", sa.String(16), nullable=False),
        sa.Column("region", sa.String(32), nullable=False),
        sa.Column("owner", sa.String(255), nullable=True),
        sa.Column("crawl_policy", sa.JSON(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("authority BETWEEN 1 AND 5", name="ck_ks_authority"),
        sa.CheckConstraint(
            "source_type IN ('website','pdf','internal','manual','taxonomy','api')",
            name="ck_ks_type",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("canonical_uri"),
    )
    op.create_index("idx_ks_enabled_type", "knowledge_sources", ["enabled", "source_type"])

    op.create_table(
        "knowledge_snapshots",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_id", sa.Uuid(), nullable=False),
        sa.Column("requested_uri", sa.Text(), nullable=False),
        sa.Column("final_uri", sa.Text(), nullable=True),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("http_status", sa.Integer(), nullable=True),
        sa.Column("content_type", sa.String(255), nullable=True),
        sa.Column("etag", sa.String(255), nullable=True),
        sa.Column("last_modified", sa.String(255), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=True),
        sa.Column("raw_storage_key", sa.Text(), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("warnings", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('queued','fetching','fetched','normalized','failed')",
            name="ck_ksnap_status",
        ),
        sa.ForeignKeyConstraint(["source_id"], ["knowledge_sources.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_id", "content_hash", name="uq_ksnap_source_hash"),
    )
    op.create_index("idx_ksnap_source_created", "knowledge_snapshots", ["source_id", "created_at"])

    op.create_table(
        "knowledge_documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("snapshot_id", sa.Uuid(), nullable=False),
        sa.Column("canonical_uri", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("canonical_markdown", sa.Text(), nullable=False),
        sa.Column("extractor", sa.String(64), nullable=False),
        sa.Column("extractor_version", sa.String(64), nullable=True),
        sa.Column("language", sa.String(16), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("warnings", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["snapshot_id"], ["knowledge_snapshots.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("snapshot_id", name="uq_kdoc_snapshot"),
    )
    op.create_index("idx_kdoc_canonical_uri", "knowledge_documents", ["canonical_uri"])

    op.create_table(
        "knowledge_units",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("stable_key", sa.String(255), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=True),
        sa.Column("unit_type", sa.String(32), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("question", sa.Text(), nullable=True),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("voice_answer", sa.Text(), nullable=True),
        sa.Column("retrieval_text", sa.Text(), nullable=False),
        sa.Column("product", sa.String(128), nullable=True),
        sa.Column("device", sa.String(128), nullable=True),
        sa.Column("topic", sa.String(128), nullable=True),
        sa.Column("issue_family", sa.String(128), nullable=True),
        sa.Column("intents", sa.JSON(), nullable=False),
        sa.Column("audience", sa.String(64), nullable=False),
        sa.Column("language", sa.String(16), nullable=False),
        sa.Column("region", sa.String(32), nullable=False),
        sa.Column("authority", sa.Integer(), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("source_label", sa.String(255), nullable=False),
        sa.Column("effective_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("requires_auth", sa.Boolean(), nullable=False),
        sa.Column("requires_live_api", sa.Boolean(), nullable=False),
        sa.Column("escalation_required", sa.Boolean(), nullable=False),
        sa.Column("ticket_candidates", sa.JSON(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("review_notes", sa.Text(), nullable=True),
        sa.Column("approved_by_user_id", sa.Integer(), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("search_vector", sa.dialects.postgresql.TSVECTOR(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("authority BETWEEN 1 AND 5", name="ck_ku_authority"),
        sa.CheckConstraint("status IN ('draft','approved','retired')", name="ck_ku_status"),
        sa.CheckConstraint(
            "unit_type IN ('faq','procedure','troubleshooting','product_spec','policy','definition','error_code','escalation_rule','contact_information','ticket_taxonomy','developer_reference')",
            name="ck_ku_type",
        ),
        sa.ForeignKeyConstraint(["document_id"], ["knowledge_documents.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["approved_by_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("stable_key"),
    )
    op.create_index("idx_ku_status_type", "knowledge_units", ["status", "unit_type"])
    op.create_index("idx_ku_product_topic", "knowledge_units", ["product", "topic"])
    op.create_index("idx_ku_search_vector", "knowledge_units", ["search_vector"], postgresql_using="gin")

    op.create_table(
        "knowledge_embeddings",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("unit_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        # Alembic revisions must be deterministic across environments. A future
        # dimension change requires a new migration and reindexed release.
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["unit_id"], ["knowledge_units.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("unit_id", "provider", "model", name="uq_kemb_unit_model"),
    )
    op.create_index(
        "idx_kemb_vector",
        "knowledge_embeddings",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "knowledge_aliases",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("canonical", sa.String(255), nullable=False),
        sa.Column("alias", sa.String(255), nullable=False),
        sa.Column("alias_type", sa.String(32), nullable=False),
        sa.Column("product", sa.String(128), nullable=True),
        sa.Column("language", sa.String(16), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("canonical", "alias", "language", name="uq_kalias_value"),
    )
    op.create_index("idx_kalias_alias_active", "knowledge_aliases", ["alias", "active"])

    op.create_table(
        "knowledge_releases",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("corpus_hash", sa.String(64), nullable=True),
        sa.Column("unit_count", sa.Integer(), nullable=False),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("status IN ('draft','published','retired')", name="ck_krel_status"),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version"),
    )
    op.create_index(
        "uq_krel_one_published",
        "knowledge_releases",
        ["status"],
        unique=True,
        postgresql_where=sa.text("status = 'published'"),
    )

    op.create_table(
        "knowledge_release_units",
        sa.Column("release_id", sa.Uuid(), nullable=False),
        sa.Column("unit_id", sa.Uuid(), nullable=False),
        sa.Column("added_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["release_id"], ["knowledge_releases.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["unit_id"], ["knowledge_units.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("release_id", "unit_id", name="pk_krel_unit"),
    )
    op.create_index("idx_krelunit_unit", "knowledge_release_units", ["unit_id"])

    op.create_table(
        "knowledge_conflicts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("left_unit_id", sa.Uuid(), nullable=False),
        sa.Column("right_unit_id", sa.Uuid(), nullable=False),
        sa.Column("conflict_type", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("resolution", sa.Text(), nullable=True),
        sa.Column("resolved_by_user_id", sa.Integer(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("status IN ('open','resolved','ignored')", name="ck_kconf_status"),
        sa.ForeignKeyConstraint(["left_unit_id"], ["knowledge_units.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["right_unit_id"], ["knowledge_units.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["resolved_by_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("left_unit_id", "right_unit_id", "conflict_type", name="uq_kconf_pair"),
    )
    op.create_index("idx_kconf_status", "knowledge_conflicts", ["status", "created_at"])

    op.create_table(
        "knowledge_jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("job_type", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("source_id", sa.Uuid(), nullable=True),
        sa.Column("snapshot_id", sa.Uuid(), nullable=True),
        sa.Column("release_id", sa.Uuid(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("status IN ('queued','running','succeeded','failed')", name="ck_kjob_status"),
        sa.ForeignKeyConstraint(["release_id"], ["knowledge_releases.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["snapshot_id"], ["knowledge_snapshots.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["source_id"], ["knowledge_sources.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_kjob_claim", "knowledge_jobs", ["status", "available_at", "created_at"])

    op.create_table(
        "knowledge_feedback",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("call_id", sa.Uuid(), nullable=True),
        sa.Column("unit_id", sa.Uuid(), nullable=True),
        sa.Column("query_fingerprint", sa.String(64), nullable=False),
        sa.Column("route", sa.String(32), nullable=False),
        sa.Column("outcome", sa.String(32), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["unit_id"], ["knowledge_units.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_kfeedback_created", "knowledge_feedback", ["created_at"])

    op.create_table(
        "ticket_taxonomy_entries",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("ticket_code", sa.String(128), nullable=False),
        sa.Column("ticket_subcode", sa.String(255), nullable=False),
        sa.Column("remark", sa.Text(), nullable=False),
        sa.Column("source_status", sa.String(32), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("source_row", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ticket_code", "ticket_subcode", "remark", name="uq_ticket_taxonomy"),
    )
    op.create_index(
        "idx_ticket_taxonomy_active_code",
        "ticket_taxonomy_entries",
        ["active", "ticket_code", "ticket_subcode"],
    )


def downgrade() -> None:
    op.drop_table("ticket_taxonomy_entries")
    op.drop_table("knowledge_feedback")
    op.drop_table("knowledge_jobs")
    op.drop_table("knowledge_conflicts")
    op.drop_table("knowledge_release_units")
    op.drop_table("knowledge_releases")
    op.drop_table("knowledge_aliases")
    op.drop_table("knowledge_embeddings")
    op.drop_table("knowledge_units")
    op.drop_table("knowledge_documents")
    op.drop_table("knowledge_snapshots")
    op.drop_table("knowledge_sources")
