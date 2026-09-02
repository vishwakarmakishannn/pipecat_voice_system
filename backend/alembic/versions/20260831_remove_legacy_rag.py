"""remove the legacy per-user uploaded-file RAG schema

Revision ID: 20260831_remove_legacy_rag
Revises: 20260827_mswipe_knowledge_v1
"""

from collections.abc import Sequence

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa


revision: str = "20260831_remove_legacy_rag"
down_revision: str | Sequence[str] | None = "20260827_mswipe_knowledge_v1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_table("rag_chunks")
    op.drop_table("rag_files")


def downgrade() -> None:
    op.create_table(
        "rag_files",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column(
            "mime_type",
            sa.String(),
            nullable=False,
            server_default="application/pdf",
        ),
        sa.Column("source_type", sa.String(), nullable=False, server_default="pdf"),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("final_url", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("site_name", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.String(), nullable=True),
        sa.Column(
            "ingestion_version",
            sa.String(32),
            nullable=False,
            server_default="legacy-v1",
        ),
        sa.Column("extractor", sa.String(64), nullable=True),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column(
            "ingestion_warnings",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'::json"),
        ),
        sa.Column("size_bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_rag_files_id", "rag_files", ["id"])
    op.create_index("ix_rag_files_user_id", "rag_files", ["user_id"])
    op.create_index("idx_rag_files_user_status", "rag_files", ["user_id", "status"])

    op.create_table(
        "rag_chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("file_id", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("heading_path", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "metadata_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::json"),
        ),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("search_vector", sa.dialects.postgresql.TSVECTOR(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["file_id"], ["rag_files.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("file_id", "chunk_index", name="uq_rag_chunk_file_index"),
    )
    op.create_index("ix_rag_chunks_id", "rag_chunks", ["id"])
    op.create_index("ix_rag_chunks_user_id", "rag_chunks", ["user_id"])
    op.create_index("ix_rag_chunks_file_id", "rag_chunks", ["file_id"])
    op.create_index("idx_rag_chunks_user_file", "rag_chunks", ["user_id", "file_id"])
    op.create_index(
        "idx_rag_chunks_search_vector",
        "rag_chunks",
        ["search_vector"],
        postgresql_using="gin",
    )
    op.create_index(
        "idx_rag_chunks_embedding",
        "rag_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
