"""index every provider/model call-list filter

Revision ID: 20260811_voice2_provider_indexes
Revises: 20260811_voice2_schema_parity
"""

from collections.abc import Sequence

from alembic import op


revision: str = "20260811_voice2_provider_indexes"
down_revision: str | Sequence[str] | None = "20260811_voice2_schema_parity"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "idx_calls_user_stt_started",
        "calls",
        ["user_id", "stt_provider", "stt_model", "started_at"],
    )
    op.create_index(
        "idx_calls_user_tts_started",
        "calls",
        ["user_id", "tts_provider", "tts_model", "started_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_calls_user_tts_started", table_name="calls")
    op.drop_index("idx_calls_user_stt_started", table_name="calls")
