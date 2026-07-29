"""synchronize legacy schema metadata

Revision ID: 20260717_schema_sync
Revises: 20260717_latency_indexes
Create Date: 2026-07-17
"""

from typing import Sequence, Union

from alembic import op


revision: str = "20260717_schema_sync"
down_revision: Union[str, Sequence[str], None] = "20260717_latency_indexes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


TIMESTAMP_COLUMNS = {
    "conversations": ("created_at", "updated_at"),
    "issues": ("created_at", "updated_at"),
    "memory_chunks": ("created_at", "updated_at"),
    "messages": ("created_at",),
    "rag_chunks": ("created_at", "updated_at"),
    "rag_files": ("created_at", "updated_at"),
    "user_memories": ("created_at", "updated_at"),
}


def upgrade() -> None:
    for table, columns in TIMESTAMP_COLUMNS.items():
        for column in columns:
            op.execute(
                f"ALTER TABLE {table} ALTER COLUMN {column} TYPE TIMESTAMPTZ "
                f"USING {column} AT TIME ZONE 'UTC'"
            )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_memory_user_updated "
        "ON user_memories (user_id, updated_at)"
    )
    # Legacy installs used IVFFlat with too many lists for small datasets.
    # HNSW performs well without a training threshold and matches fresh installs.
    op.execute("DROP INDEX IF EXISTS idx_memory_chunks_embedding")
    op.execute(
        "CREATE INDEX idx_memory_chunks_embedding ON memory_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_memory_chunks_embedding")
    op.execute(
        "CREATE INDEX idx_memory_chunks_embedding ON memory_chunks "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )
    op.execute("DROP INDEX IF EXISTS idx_user_memory_user_updated")
    for table, columns in TIMESTAMP_COLUMNS.items():
        for column in columns:
            op.execute(
                f"ALTER TABLE {table} ALTER COLUMN {column} TYPE TIMESTAMP "
                f"USING {column} AT TIME ZONE 'UTC'"
            )
