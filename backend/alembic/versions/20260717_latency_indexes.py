"""ensure latency-critical indexes

Revision ID: 20260717_latency_indexes
Revises: 3c5c3ec4e525
Create Date: 2026-07-17
"""

from typing import Sequence, Union

from alembic import op


revision: str = "20260717_latency_indexes"
down_revision: Union[str, Sequence[str], None] = "3c5c3ec4e525"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE INDEX IF NOT EXISTS idx_conversation_user_updated ON conversations (user_id, updated_at)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_message_conversation_created ON messages (conversation_id, created_at)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_rag_files_user_status ON rag_files (user_id, status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_rag_chunks_user_file ON rag_chunks (user_id, file_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_rag_chunks_search_vector ON rag_chunks USING gin (search_vector)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding ON rag_chunks USING hnsw (embedding vector_cosine_ops)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_memory_chunks_embedding ON memory_chunks USING hnsw (embedding vector_cosine_ops)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_memory_chunks_embedding")
    op.execute("DROP INDEX IF EXISTS idx_rag_chunks_embedding")
    op.execute("DROP INDEX IF EXISTS idx_rag_chunks_search_vector")
    op.execute("DROP INDEX IF EXISTS idx_rag_chunks_user_file")
    op.execute("DROP INDEX IF EXISTS idx_rag_files_user_status")
    op.execute("DROP INDEX IF EXISTS idx_message_conversation_created")
    op.execute("DROP INDEX IF EXISTS idx_conversation_user_updated")
