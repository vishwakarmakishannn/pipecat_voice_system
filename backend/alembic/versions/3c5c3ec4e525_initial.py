"""initial

Revision ID: 3c5c3ec4e525
Revises: 
Create Date: 2026-07-15 18:23:01.089400

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3c5c3ec4e525'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create a complete baseline schema for fresh deployments."""
    from core.models import Base

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    Base.metadata.create_all(bind=op.get_bind())
    op.execute("CREATE INDEX IF NOT EXISTS idx_rag_files_user_status ON rag_files (user_id, status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_rag_chunks_user_file ON rag_chunks (user_id, file_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_rag_chunks_search_vector ON rag_chunks USING gin (search_vector)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding ON rag_chunks USING hnsw (embedding vector_cosine_ops)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_memory_chunks_embedding ON memory_chunks USING hnsw (embedding vector_cosine_ops)")


def downgrade() -> None:
    """Remove the baseline schema."""
    from core.models import Base

    Base.metadata.drop_all(bind=op.get_bind())
