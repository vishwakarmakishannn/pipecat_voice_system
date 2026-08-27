"""enforce terminal call immutability in PostgreSQL

Revision ID: 20260811_voice2_immutability
Revises: 20260811_voice_system_2
"""

from alembic import op


revision = "20260811_voice2_immutability"
down_revision = "20260811_voice_system_2"
branch_labels = None
depends_on = None


CHILD_TABLES = ("call_turns", "transcript_entries", "call_operations", "call_events")


def upgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION reject_terminal_call_child_write()
        RETURNS trigger AS $$
        DECLARE parent_status text;
        BEGIN
            SELECT status INTO parent_status FROM calls WHERE id = NEW.call_id;
            IF parent_status IN ('completed', 'failed', 'cancelled', 'abandoned') THEN
                RAISE EXCEPTION 'terminal call % is immutable', NEW.call_id
                    USING ERRCODE = '55000';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    for table in CHILD_TABLES:
        op.execute(
            f"""
            CREATE TRIGGER trg_{table}_terminal_write
            BEFORE INSERT OR UPDATE ON {table}
            FOR EACH ROW EXECUTE FUNCTION reject_terminal_call_child_write()
            """
        )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION reject_terminal_call_reopen()
        RETURNS trigger AS $$
        BEGIN
            IF OLD.status IN ('completed', 'failed', 'cancelled', 'abandoned')
               AND NEW.status IS DISTINCT FROM OLD.status THEN
                RAISE EXCEPTION 'terminal call % cannot be reopened', OLD.id
                    USING ERRCODE = '55000';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_calls_terminal_reopen
        BEFORE UPDATE OF status ON calls
        FOR EACH ROW EXECUTE FUNCTION reject_terminal_call_reopen()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_calls_terminal_reopen ON calls")
    op.execute("DROP FUNCTION IF EXISTS reject_terminal_call_reopen()")
    for table in CHILD_TABLES:
        op.execute(f"DROP TRIGGER IF EXISTS trg_{table}_terminal_write ON {table}")
    op.execute("DROP FUNCTION IF EXISTS reject_terminal_call_child_write()")

