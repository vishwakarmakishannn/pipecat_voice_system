"""replace reopenable conversations with immutable voice calls

Revision ID: 20260811_voice_system_2
Revises: 20260810_durable_rag_queue
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision: str = "20260811_voice_system_2"
down_revision: str | Sequence[str] | None = "20260810_durable_rag_queue"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


LEGACY_DATA_TABLES = (
    "messages",
    "conversations",
    "memory_chunks",
    "user_memories",
    "rag_chunks",
    "rag_files",
    "rag_ingestion_jobs",
    "issues",
    "users",
)


def _refuse_implicit_data_loss() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())
    populated: list[str] = []
    for table_name in LEGACY_DATA_TABLES:
        if table_name not in existing:
            continue
        count = bind.execute(sa.text(f'SELECT count(*) FROM "{table_name}"')).scalar_one()
        if count:
            populated.append(f"{table_name}={count}")
    if populated:
        raise RuntimeError(
            "Voice System 2.0 is a destructive migration and legacy data still exists "
            f"({', '.join(populated)}). Run `uv run python -m scripts.reset_voice2_database "
            "--confirm RESET_ALL_APPLICATION_DATA` against the intended database first."
        )


def upgrade() -> None:
    _refuse_implicit_data_loss()

    op.drop_table("memory_chunks")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.alter_column(
        "user_memories",
        "source_message_id",
        new_column_name="source_transcript_id",
        existing_type=sa.Integer(),
        type_=sa.BigInteger(),
        existing_nullable=True,
    )

    op.create_table(
        "calls",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("transport", sa.String(32), nullable=False),
        sa.Column("direction", sa.String(16), nullable=False),
        sa.Column("runner_session_id", sa.String(128), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("connected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("end_reason", sa.String(64), nullable=True),
        sa.Column("ended_by", sa.String(32), nullable=True),
        sa.Column("stt_provider", sa.String(64), nullable=True),
        sa.Column("stt_model", sa.String(255), nullable=True),
        sa.Column("stt_language", sa.String(32), nullable=True),
        sa.Column("llm_provider", sa.String(64), nullable=True),
        sa.Column("llm_model", sa.String(255), nullable=True),
        sa.Column("tts_provider", sa.String(64), nullable=True),
        sa.Column("tts_model", sa.String(255), nullable=True),
        sa.Column("tts_voice", sa.String(255), nullable=True),
        sa.Column("tts_language", sa.String(32), nullable=True),
        sa.Column("input_sample_rate", sa.Integer(), nullable=True),
        sa.Column("output_sample_rate", sa.Integer(), nullable=True),
        sa.Column("recording_sample_rate", sa.Integer(), nullable=False),
        sa.Column("provider_config", sa.JSON(), nullable=False),
        sa.Column("endpointing_config", sa.JSON(), nullable=False),
        sa.Column("pipeline_config", sa.JSON(), nullable=False),
        sa.Column("prompt_version", sa.String(128), nullable=True),
        sa.Column("prompt_hash", sa.String(64), nullable=True),
        sa.Column("tool_schema_hash", sa.String(64), nullable=True),
        sa.Column("rag_config_version", sa.String(128), nullable=True),
        sa.Column("application_version", sa.String(128), nullable=True),
        sa.Column("turn_count", sa.Integer(), nullable=False),
        sa.Column("tool_call_count", sa.Integer(), nullable=False),
        sa.Column("error_count", sa.Integer(), nullable=False),
        sa.Column("warning_count", sa.Integer(), nullable=False),
        sa.Column("next_timeline_sequence", sa.Integer(), nullable=False),
        sa.Column("avg_latency_ms", sa.Float(), nullable=True),
        sa.Column("p50_latency_ms", sa.Float(), nullable=True),
        sa.Column("p90_latency_ms", sa.Float(), nullable=True),
        sa.Column("recording_policy_version", sa.String(64), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("purge_after", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('initializing','active','ending','completed','failed','cancelled','abandoned')",
            name="ck_calls_status",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_calls_user_started", "calls", ["user_id", "started_at"])
    op.create_index("idx_calls_user_status_started", "calls", ["user_id", "status", "started_at"])
    op.create_index("idx_calls_purge_after", "calls", ["purge_after"])
    op.create_index("idx_calls_runner_session", "calls", ["runner_session_id"])

    op.create_table(
        "call_turns",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("call_id", sa.Uuid(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("input_mode", sa.String(16), nullable=False),
        sa.Column("outcome", sa.String(24), nullable=False),
        sa.Column("interrupted", sa.Boolean(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stt_latency_ms", sa.Float(), nullable=True),
        sa.Column("llm_latency_ms", sa.Float(), nullable=True),
        sa.Column("tts_latency_ms", sa.Float(), nullable=True),
        sa.Column("tool_latency_ms", sa.Float(), nullable=True),
        sa.Column("rag_latency_ms", sa.Float(), nullable=True),
        sa.Column("first_audio_latency_ms", sa.Float(), nullable=True),
        sa.Column("end_to_end_latency_ms", sa.Float(), nullable=True),
        sa.Column("llm_input_tokens", sa.Integer(), nullable=True),
        sa.Column("llm_output_tokens", sa.Integer(), nullable=True),
        sa.Column("stt_audio_ms", sa.Float(), nullable=True),
        sa.Column("tts_characters", sa.Integer(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("call_id", "sequence", name="uq_call_turn_sequence"),
    )
    op.create_index("idx_call_turns_call_sequence", "call_turns", ["call_id", "sequence"])

    op.create_table(
        "transcript_entries",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("call_id", sa.Uuid(), nullable=False),
        sa.Column("turn_id", sa.Integer(), nullable=True),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("speaker", sa.String(24), nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("audio_offset_ms", sa.Float(), nullable=True),
        sa.Column("audio_end_offset_ms", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("is_final", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("call_id", "sequence", name="uq_transcript_call_sequence"),
    )
    op.create_index("idx_transcript_call_sequence", "transcript_entries", ["call_id", "sequence"])
    op.create_index("idx_transcript_call_turn", "transcript_entries", ["call_id", "turn_id"])

    op.create_table(
        "call_operations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("call_id", sa.Uuid(), nullable=False),
        sa.Column("turn_id", sa.Integer(), nullable=True),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("operation_type", sa.String(16), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("arguments", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("request_id", sa.String(255), nullable=True),
        sa.Column("error_code", sa.String(128), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("call_id", "sequence", name="uq_call_operation_sequence"),
    )
    op.create_index("idx_call_operations_call_started", "call_operations", ["call_id", "started_at"])
    op.create_index("idx_call_operations_call_turn", "call_operations", ["call_id", "turn_id"])

    op.create_table(
        "call_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("call_id", sa.Uuid(), nullable=False),
        sa.Column("turn_id", sa.Integer(), nullable=True),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("component", sa.String(32), nullable=False),
        sa.Column("code", sa.String(128), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("outcome", sa.String(24), nullable=False),
        sa.Column("safe_message", sa.Text(), nullable=False),
        sa.Column("operator_detail", sa.Text(), nullable=True),
        sa.Column("provider", sa.String(64), nullable=True),
        sa.Column("model", sa.String(255), nullable=True),
        sa.Column("request_id", sa.String(255), nullable=True),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("retryable", sa.Boolean(), nullable=False),
        sa.Column("recovered", sa.Boolean(), nullable=False),
        sa.Column("fatal", sa.Boolean(), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("call_id", "fingerprint", name="uq_call_event_fingerprint"),
        sa.UniqueConstraint("call_id", "sequence", name="uq_call_event_sequence"),
    )
    op.create_index("idx_call_events_call_created", "call_events", ["call_id", "created_at"])
    op.create_index("idx_call_events_code_created", "call_events", ["code", "created_at"])

    op.create_table(
        "call_recordings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("call_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("object_key", sa.Text(), nullable=True),
        sa.Column("spool_path", sa.Text(), nullable=True),
        sa.Column("mime_type", sa.String(64), nullable=False),
        sa.Column("codec", sa.String(32), nullable=False),
        sa.Column("channels", sa.Integer(), nullable=False),
        sa.Column("sample_rate", sa.Integer(), nullable=False),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("checksum_sha256", sa.String(64), nullable=True),
        sa.Column("failure_code", sa.String(128), nullable=True),
        sa.Column("failure_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('recording','processing','available','failed','deleted')",
            name="ck_call_recordings_status",
        ),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("call_id"),
    )
    op.create_index("idx_call_recordings_status_updated", "call_recordings", ["status", "updated_at"])

    op.create_table(
        "memory_chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("call_id", sa.Uuid(), nullable=False),
        sa.Column("transcript_start_id", sa.BigInteger(), nullable=False),
        sa.Column("transcript_end_id", sa.BigInteger(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "call_id", "transcript_start_id", "transcript_end_id",
            name="uq_memory_chunk_transcript_window",
        ),
    )
    op.create_index(
        "idx_memory_chunks_embedding",
        "memory_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    raise RuntimeError("Voice System 2.0 is a destructive baseline and cannot be downgraded")
