"""add aggregate interruption count

Revision ID: 20260811_voice2_metrics
Revises: 20260811_voice2_memory_fk
"""

from alembic import op
import sqlalchemy as sa


revision = "20260811_voice2_metrics"
down_revision = "20260811_voice2_memory_fk"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "calls",
        sa.Column("interruption_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.alter_column("calls", "interruption_count", server_default=None)


def downgrade() -> None:
    op.drop_column("calls", "interruption_count")
