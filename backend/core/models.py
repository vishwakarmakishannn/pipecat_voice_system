import uuid

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime, timezone
from core.database import Base
from core.memory_config import MEMORY_EMBEDDING_DIMENSION

def utcnow():
    return datetime.now(timezone.utc)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    
    calls = relationship("Call", back_populates="user", cascade="all, delete-orphan")
    memories = relationship("UserMemory", back_populates="user", cascade="all, delete-orphan")
    memory_chunks = relationship("MemoryChunk", back_populates="user", cascade="all, delete-orphan")
    rag_files = relationship("RagFile", back_populates="user", cascade="all, delete-orphan")
    rag_chunks = relationship("RagChunk", back_populates="user", cascade="all, delete-orphan")

TERMINAL_CALL_STATUSES = ("completed", "failed", "cancelled", "abandoned")


class Call(Base):
    __tablename__ = "calls"
    __table_args__ = (
        CheckConstraint(
            "status IN ('initializing','active','ending','completed','failed','cancelled','abandoned')",
            name="ck_calls_status",
        ),
        Index("idx_calls_user_started", "user_id", "started_at"),
        Index("idx_calls_user_status_started", "user_id", "status", "started_at"),
        Index("idx_calls_user_deleted_started", "user_id", "deleted_at", "started_at"),
        Index("idx_calls_user_stt_started", "user_id", "stt_provider", "stt_model", "started_at"),
        Index("idx_calls_user_llm_started", "user_id", "llm_provider", "llm_model", "started_at"),
        Index("idx_calls_user_tts_started", "user_id", "tts_provider", "tts_model", "started_at"),
        Index("idx_calls_purge_after", "purge_after"),
        Index("idx_calls_runner_session", "runner_session_id"),
    )

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), nullable=False, default="New call")
    summary = Column(Text, nullable=False, default="")
    status = Column(String(24), nullable=False, default="initializing")
    transport = Column(String(32), nullable=False, default="webrtc")
    direction = Column(String(16), nullable=False, default="web")
    runner_session_id = Column(String(128), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    connected_at = Column(DateTime(timezone=True), nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Float, nullable=True)
    end_reason = Column(String(64), nullable=True)
    ended_by = Column(String(32), nullable=True)

    stt_provider = Column(String(64), nullable=True)
    stt_model = Column(String(255), nullable=True)
    stt_language = Column(String(32), nullable=True)
    llm_provider = Column(String(64), nullable=True)
    llm_model = Column(String(255), nullable=True)
    tts_provider = Column(String(64), nullable=True)
    tts_model = Column(String(255), nullable=True)
    tts_voice = Column(String(255), nullable=True)
    tts_language = Column(String(32), nullable=True)
    input_sample_rate = Column(Integer, nullable=True)
    output_sample_rate = Column(Integer, nullable=True)
    recording_sample_rate = Column(Integer, nullable=False, default=16000)

    provider_config = Column(JSON, nullable=False, default=dict)
    endpointing_config = Column(JSON, nullable=False, default=dict)
    pipeline_config = Column(JSON, nullable=False, default=dict)
    prompt_version = Column(String(128), nullable=True)
    prompt_hash = Column(String(64), nullable=True)
    tool_schema_hash = Column(String(64), nullable=True)
    rag_config_version = Column(String(128), nullable=True)
    application_version = Column(String(128), nullable=True)

    turn_count = Column(Integer, nullable=False, default=0)
    tool_call_count = Column(Integer, nullable=False, default=0)
    error_count = Column(Integer, nullable=False, default=0)
    warning_count = Column(Integer, nullable=False, default=0)
    interruption_count = Column(Integer, nullable=False, default=0)
    next_timeline_sequence = Column(Integer, nullable=False, default=0)
    avg_latency_ms = Column(Float, nullable=True)
    p50_latency_ms = Column(Float, nullable=True)
    p90_latency_ms = Column(Float, nullable=True)
    recording_policy_version = Column(String(64), nullable=False, default="always-on-v1")

    deleted_at = Column(DateTime(timezone=True), nullable=True)
    purge_after = Column(DateTime(timezone=True), nullable=True)
    purge_started_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    user = relationship("User", back_populates="calls")
    turns = relationship("CallTurn", back_populates="call", cascade="all, delete-orphan")
    transcripts = relationship("TranscriptEntry", back_populates="call", cascade="all, delete-orphan")
    operations = relationship("CallOperation", back_populates="call", cascade="all, delete-orphan")
    events = relationship("CallEvent", back_populates="call", cascade="all, delete-orphan")
    recording = relationship("CallRecording", back_populates="call", cascade="all, delete-orphan", uselist=False)
    memory_chunks = relationship("MemoryChunk", back_populates="call", cascade="all, delete-orphan")


class CallTurn(Base):
    __tablename__ = "call_turns"
    __table_args__ = (
        UniqueConstraint("call_id", "sequence", name="uq_call_turn_sequence"),
        Index("idx_call_turns_call_sequence", "call_id", "sequence"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    call_id = Column(Uuid(as_uuid=True), ForeignKey("calls.id", ondelete="CASCADE"), nullable=False)
    sequence = Column(Integer, nullable=False)
    input_mode = Column(String(16), nullable=False, default="voice")
    outcome = Column(String(24), nullable=False, default="completed")
    interrupted = Column(Boolean, nullable=False, default=False)
    started_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    stt_latency_ms = Column(Float, nullable=True)
    llm_latency_ms = Column(Float, nullable=True)
    tts_latency_ms = Column(Float, nullable=True)
    tool_latency_ms = Column(Float, nullable=True)
    rag_latency_ms = Column(Float, nullable=True)
    first_audio_latency_ms = Column(Float, nullable=True)
    end_to_end_latency_ms = Column(Float, nullable=True)
    llm_input_tokens = Column(Integer, nullable=True)
    llm_output_tokens = Column(Integer, nullable=True)
    stt_audio_ms = Column(Float, nullable=True)
    tts_characters = Column(Integer, nullable=True)
    metrics = Column(JSON, nullable=False, default=dict)

    call = relationship("Call", back_populates="turns")


class TranscriptEntry(Base):
    __tablename__ = "transcript_entries"
    __table_args__ = (
        UniqueConstraint("call_id", "sequence", name="uq_transcript_call_sequence"),
        Index("idx_transcript_call_sequence", "call_id", "sequence"),
        Index("idx_transcript_call_turn", "call_id", "turn_id"),
        UniqueConstraint("call_id", "persistence_id", name="uq_transcript_persistence_id"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    call_id = Column(Uuid(as_uuid=True), ForeignKey("calls.id", ondelete="CASCADE"), nullable=False)
    turn_id = Column(Integer, nullable=True)
    sequence = Column(Integer, nullable=False)
    speaker = Column(String(24), nullable=False)
    source = Column(String(32), nullable=False)
    text = Column(Text, nullable=False)
    audio_offset_ms = Column(Float, nullable=True)
    audio_end_offset_ms = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    is_final = Column(Boolean, nullable=False, default=True)
    persistence_id = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    call = relationship("Call", back_populates="transcripts")


class CallOperation(Base):
    __tablename__ = "call_operations"
    __table_args__ = (
        UniqueConstraint("call_id", "sequence", name="uq_call_operation_sequence"),
        Index("idx_call_operations_call_started", "call_id", "started_at"),
        Index("idx_call_operations_call_turn", "call_id", "turn_id"),
        UniqueConstraint("call_id", "persistence_id", name="uq_call_operation_persistence_id"),
    )

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id = Column(Uuid(as_uuid=True), ForeignKey("calls.id", ondelete="CASCADE"), nullable=False)
    turn_id = Column(Integer, nullable=True)
    sequence = Column(Integer, nullable=False)
    operation_type = Column(String(16), nullable=False)
    name = Column(String(255), nullable=False)
    status = Column(String(24), nullable=False, default="completed")
    arguments = Column(JSON, nullable=False, default=dict)
    result = Column(JSON, nullable=True)
    request_id = Column(String(255), nullable=True)
    error_code = Column(String(128), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Float, nullable=True)
    persistence_id = Column(String(64), nullable=True)

    call = relationship("Call", back_populates="operations")


class CallEvent(Base):
    __tablename__ = "call_events"
    __table_args__ = (
        Index("idx_call_events_call_created", "call_id", "created_at"),
        Index("idx_call_events_code_created", "code", "created_at"),
        UniqueConstraint("call_id", "sequence", name="uq_call_event_sequence"),
        UniqueConstraint("call_id", "fingerprint", name="uq_call_event_fingerprint"),
    )

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id = Column(Uuid(as_uuid=True), ForeignKey("calls.id", ondelete="CASCADE"), nullable=False)
    turn_id = Column(Integer, nullable=True)
    sequence = Column(Integer, nullable=False)
    component = Column(String(32), nullable=False)
    code = Column(String(128), nullable=False)
    severity = Column(String(16), nullable=False)
    outcome = Column(String(24), nullable=False)
    safe_message = Column(Text, nullable=False)
    operator_detail = Column(Text, nullable=True)
    provider = Column(String(64), nullable=True)
    model = Column(String(255), nullable=True)
    request_id = Column(String(255), nullable=True)
    duration_ms = Column(Float, nullable=True)
    retryable = Column(Boolean, nullable=False, default=False)
    recovered = Column(Boolean, nullable=False, default=False)
    fatal = Column(Boolean, nullable=False, default=False)
    details = Column(JSON, nullable=False, default=dict)
    fingerprint = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    call = relationship("Call", back_populates="events")


class CallRecording(Base):
    __tablename__ = "call_recordings"
    __table_args__ = (
        CheckConstraint("status IN ('recording','processing','available','failed','deleted')", name="ck_call_recordings_status"),
        Index("idx_call_recordings_status_updated", "status", "updated_at"),
    )

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id = Column(Uuid(as_uuid=True), ForeignKey("calls.id", ondelete="CASCADE"), nullable=False, unique=True)
    status = Column(String(24), nullable=False, default="recording")
    object_key = Column(Text, nullable=True)
    spool_path = Column(Text, nullable=True)
    mime_type = Column(String(64), nullable=False, default="audio/mpeg")
    codec = Column(String(32), nullable=False, default="mp3")
    channels = Column(Integer, nullable=False, default=1)
    sample_rate = Column(Integer, nullable=False, default=16000)
    duration_ms = Column(Float, nullable=True)
    size_bytes = Column(BigInteger, nullable=True)
    checksum_sha256 = Column(String(64), nullable=True)
    failure_code = Column(String(128), nullable=True)
    failure_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    call = relationship("Call", back_populates="recording")

class UserMemory(Base):
    __tablename__ = "user_memories"
    __table_args__ = (
        UniqueConstraint("user_id", "fact_type", "key", "value", name="uq_user_memory_fact_value"),
        Index("idx_user_memory_user_updated", "user_id", "updated_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    fact_type = Column(String, default="profile", nullable=False)
    key = Column(String, nullable=False)
    value = Column(Text, nullable=False)
    confidence = Column(Float, default=1.0)
    durability = Column(String, default="stable", nullable=False)
    status = Column(String, default="active", nullable=False)
    source_transcript_id = Column(
        BigInteger,
        ForeignKey("transcript_entries.id", ondelete="CASCADE"),
        nullable=True,
    )
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user = relationship("User", back_populates="memories")

class MemoryChunk(Base):
    __tablename__ = "memory_chunks"
    __table_args__ = (
        UniqueConstraint("call_id", "transcript_start_id", "transcript_end_id", name="uq_memory_chunk_transcript_window"),
        Index(
            "idx_memory_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    call_id = Column(Uuid(as_uuid=True), ForeignKey("calls.id", ondelete="CASCADE"), nullable=False)
    transcript_start_id = Column(BigInteger, nullable=False)
    transcript_end_id = Column(BigInteger, nullable=False)
    chunk_text = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    embedding = Column(Vector(MEMORY_EMBEDDING_DIMENSION), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user = relationship("User", back_populates="memory_chunks")
    call = relationship("Call", back_populates="memory_chunks")


class RagFile(Base):
    __tablename__ = "rag_files"
    __table_args__ = (Index("idx_rag_files_user_status", "user_id", "status"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    storage_path = Column(Text, nullable=False)
    mime_type = Column(String, nullable=False, default="application/pdf")
    source_type = Column(String, nullable=False, default="pdf")
    url = Column(Text, nullable=True)
    final_url = Column(Text, nullable=True)
    title = Column(Text, nullable=True)
    site_name = Column(Text, nullable=True)
    content_hash = Column(String, nullable=True)
    ingestion_version = Column(String(32), nullable=False, default="structured-v2")
    extractor = Column(String(64), nullable=True)
    quality_score = Column(Float, nullable=True)
    ingestion_warnings = Column(JSON, nullable=False, default=list)
    size_bytes = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False, default="queued")
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user = relationship("User", back_populates="rag_files")
    chunks = relationship("RagChunk", back_populates="file", cascade="all, delete-orphan")


class RagChunk(Base):
    __tablename__ = "rag_chunks"
    __table_args__ = (
        UniqueConstraint("file_id", "chunk_index", name="uq_rag_chunk_file_index"),
        Index("idx_rag_chunks_user_file", "user_id", "file_id"),
        Index("idx_rag_chunks_search_vector", "search_vector", postgresql_using="gin"),
        Index(
            "idx_rag_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    file_id = Column(Integer, ForeignKey("rag_files.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    heading_path = Column(Text, nullable=True)
    content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False, default=0)
    metadata_json = Column(JSON, nullable=False, default=dict)
    embedding = Column(Vector(MEMORY_EMBEDDING_DIMENSION), nullable=True)
    search_vector = Column(TSVECTOR, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user = relationship("User", back_populates="rag_chunks")
    file = relationship("RagFile", back_populates="chunks")


class Issue(Base):
    __tablename__ = "issues"

    id = Column(Integer, primary_key=True, index=True)
    cust_id = Column(String, nullable=False)
    email = Column(String, nullable=False)
    mobile = Column(String, nullable=False)
    device_id = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="raised")
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
