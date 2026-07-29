"""enable filtered HNSW iterative scans

Revision ID: 20260720_pgvector_iterative_scan
Revises: 20260717_schema_sync
Create Date: 2026-07-20
"""

from typing import Sequence, Union

from alembic import op


revision: str = "20260720_pgvector_iterative_scan"
down_revision: Union[str, Sequence[str], None] = "20260717_schema_sync"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Iterative HNSW scanning (pgvector 0.8+) is designed for approximate
    # indexes whose results are narrowed by tenant/status predicates.
    op.execute("ALTER EXTENSION vector UPDATE")


def downgrade() -> None:
    # Extension downgrades are intentionally unsupported by PostgreSQL.
    pass
