"""make RAG ingestion status a durable queue

Revision ID: 20260810_durable_rag_queue
Revises: 20260720_pgvector_iterative_scan
Create Date: 2026-08-10
"""

from typing import Sequence, Union

from alembic import op


revision: str = "20260810_durable_rag_queue"
down_revision: Union[str, Sequence[str], None] = (
    "20260720_pgvector_iterative_scan"
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE rag_files ALTER COLUMN status SET DEFAULT 'queued'")
    # The old API used an in-memory task queue. At deployment there can be no
    # surviving owner for these rows, so return them to the durable worker.
    op.execute("UPDATE rag_files SET status = 'queued' WHERE status = 'processing'")


def downgrade() -> None:
    op.execute("UPDATE rag_files SET status = 'processing' WHERE status = 'queued'")
    op.execute("ALTER TABLE rag_files ALTER COLUMN status SET DEFAULT 'processing'")
