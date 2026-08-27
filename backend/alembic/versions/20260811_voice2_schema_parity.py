"""complete Voice 2 ORM/index parity

Revision ID: 20260811_voice2_schema_parity
Revises: 20260811_voice2_metrics
"""

from collections.abc import Sequence

from alembic import op


revision: str = "20260811_voice2_schema_parity"
down_revision: str | Sequence[str] | None = "20260811_voice2_metrics"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_memory_chunks_id ON memory_chunks (id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_memory_chunks_id")
