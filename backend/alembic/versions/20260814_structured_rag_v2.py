"""add structured RAG ingestion metadata and multilingual text index

Revision ID: 20260814_structured_rag_v2
Revises: 20260813_review_fixes
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260814_structured_rag_v2"
down_revision: str | Sequence[str] | None = "20260813_review_fixes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "rag_files",
        sa.Column(
            "ingestion_version",
            sa.String(length=32),
            nullable=False,
            server_default="legacy-v1",
        ),
    )
    op.add_column("rag_files", sa.Column("extractor", sa.String(length=64), nullable=True))
    op.add_column("rag_files", sa.Column("quality_score", sa.Float(), nullable=True))
    op.add_column(
        "rag_files",
        sa.Column(
            "ingestion_warnings",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'::json"),
        ),
    )
    op.add_column(
        "rag_chunks",
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "rag_chunks",
        sa.Column(
            "metadata_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::json"),
        ),
    )

    # `simple` keeps identifiers, names, years, and non-English terms instead
    # of applying the English stemmer/stop-word list. Existing rows are rebuilt
    # so old and newly ingested sources use the same query configuration.
    op.execute(
        """
        UPDATE rag_chunks
        SET search_vector = to_tsvector(
            'simple',
            concat_ws(' ', heading_path, content)
        )
        """
    )
    # Existing links may contain structurally lossy v1 chunks. Queue them for
    # the durable worker so deployment repairs the corpus without asking every
    # user to delete and re-add each URL. PDFs keep their existing Docling
    # structure and receive the rebuilt multilingual lexical index above.
    op.execute(
        """
        UPDATE rag_files
        SET status = 'queued', error = NULL, updated_at = now()
        WHERE source_type = 'link' AND status = 'ready'
        """
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE rag_chunks
        SET search_vector = to_tsvector(
            'english',
            concat_ws(' ', heading_path, content)
        )
        """
    )
    op.drop_column("rag_chunks", "metadata_json")
    op.drop_column("rag_chunks", "token_count")
    op.drop_column("rag_files", "ingestion_warnings")
    op.drop_column("rag_files", "quality_score")
    op.drop_column("rag_files", "extractor")
    op.drop_column("rag_files", "ingestion_version")
