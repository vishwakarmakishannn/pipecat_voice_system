"""tighten call filtering and memory provenance

Revision ID: 20260811_voice2_memory_fk
Revises: 20260811_voice2_immutability
"""

from alembic import op


revision = "20260811_voice2_memory_fk"
down_revision = "20260811_voice2_immutability"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_foreign_key(
        "fk_user_memories_source_transcript",
        "user_memories",
        "transcript_entries",
        ["source_transcript_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "idx_calls_user_deleted_started",
        "calls",
        ["user_id", "deleted_at", "started_at"],
    )
    op.create_index(
        "idx_calls_user_llm_started",
        "calls",
        ["user_id", "llm_provider", "llm_model", "started_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_calls_user_llm_started", table_name="calls")
    op.drop_index("idx_calls_user_deleted_started", table_name="calls")
    op.drop_constraint(
        "fk_user_memories_source_transcript",
        "user_memories",
        type_="foreignkey",
    )
