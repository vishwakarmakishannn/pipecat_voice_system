"""serialize call purge and enforce terminal audit immutability

Revision ID: 20260813_review_fixes
Revises: 20260811_voice2_provider_indexes
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260813_review_fixes"
down_revision: str | Sequence[str] | None = "20260811_voice2_provider_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


CHILD_TABLES = ("call_turns", "transcript_entries", "call_operations", "call_events")


def upgrade() -> None:
    op.add_column(
        "calls",
        sa.Column("purge_started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_calls_purge_started", "calls", ["purge_started_at"])
    op.add_column(
        "transcript_entries",
        sa.Column("persistence_id", sa.String(length=64), nullable=True),
    )
    op.create_unique_constraint(
        "uq_transcript_persistence_id",
        "transcript_entries",
        ["call_id", "persistence_id"],
    )
    op.add_column(
        "call_operations",
        sa.Column("persistence_id", sa.String(length=64), nullable=True),
    )
    op.create_unique_constraint(
        "uq_call_operation_persistence_id",
        "call_operations",
        ["call_id", "persistence_id"],
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION reject_terminal_call_child_write()
        RETURNS trigger AS $$
        DECLARE parent_status text;
        DECLARE target_call_id uuid;
        BEGIN
            target_call_id := CASE WHEN TG_OP = 'DELETE' THEN OLD.call_id ELSE NEW.call_id END;
            IF TG_OP = 'DELETE'
               AND current_setting('aura.allow_call_purge', true) = 'on' THEN
                RETURN OLD;
            END IF;
            SELECT status INTO parent_status FROM calls WHERE id = target_call_id;
            IF parent_status IN ('completed', 'failed', 'cancelled', 'abandoned') THEN
                RAISE EXCEPTION 'terminal call % is immutable', target_call_id
                    USING ERRCODE = '55000';
            END IF;
            RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    for table in CHILD_TABLES:
        op.execute(f"DROP TRIGGER IF EXISTS trg_{table}_terminal_write ON {table}")
        op.execute(
            f"""
            CREATE TRIGGER trg_{table}_terminal_write
            BEFORE INSERT OR UPDATE OR DELETE ON {table}
            FOR EACH ROW EXECUTE FUNCTION reject_terminal_call_child_write()
            """
        )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION reject_terminal_call_mutation()
        RETURNS trigger AS $$
        BEGIN
            IF OLD.status IN ('completed', 'failed', 'cancelled', 'abandoned')
               AND (
                   to_jsonb(NEW) - ARRAY['deleted_at', 'purge_after', 'purge_started_at', 'updated_at']
               ) IS DISTINCT FROM (
                   to_jsonb(OLD) - ARRAY['deleted_at', 'purge_after', 'purge_started_at', 'updated_at']
               ) THEN
                RAISE EXCEPTION 'terminal call % is immutable', OLD.id
                    USING ERRCODE = '55000';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute("DROP TRIGGER IF EXISTS trg_calls_terminal_reopen ON calls")
    op.execute(
        """
        CREATE TRIGGER trg_calls_terminal_mutation
        BEFORE UPDATE ON calls
        FOR EACH ROW EXECUTE FUNCTION reject_terminal_call_mutation()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_calls_terminal_mutation ON calls")
    op.execute("DROP FUNCTION IF EXISTS reject_terminal_call_mutation()")
    op.execute(
        """
        CREATE TRIGGER trg_calls_terminal_reopen
        BEFORE UPDATE OF status ON calls
        FOR EACH ROW EXECUTE FUNCTION reject_terminal_call_reopen()
        """
    )
    for table in CHILD_TABLES:
        op.execute(f"DROP TRIGGER IF EXISTS trg_{table}_terminal_write ON {table}")
        op.execute(
            f"""
            CREATE TRIGGER trg_{table}_terminal_write
            BEFORE INSERT OR UPDATE ON {table}
            FOR EACH ROW EXECUTE FUNCTION reject_terminal_call_child_write()
            """
        )
    op.drop_index("idx_calls_purge_started", table_name="calls")
    op.drop_constraint(
        "uq_call_operation_persistence_id", "call_operations", type_="unique"
    )
    op.drop_column("call_operations", "persistence_id")
    op.drop_constraint(
        "uq_transcript_persistence_id", "transcript_entries", type_="unique"
    )
    op.drop_column("transcript_entries", "persistence_id")
    op.drop_column("calls", "purge_started_at")
