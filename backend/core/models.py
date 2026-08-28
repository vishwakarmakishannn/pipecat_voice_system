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
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime, timezone
from core.database import Base
from core.memory_config import MEMORY_EMBEDDING_DIMENSION
from core.knowledge_config import KNOWLEDGE_EMBEDDING_DIMENSION

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
        Index("idx_calls_purge_started", "purge_started_at"),
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


KNOWLEDGE_UNIT_TYPES = (
    "faq",
    "procedure",
    "troubleshooting",
    "product_spec",
    "policy",
    "definition",
    "error_code",
    "escalation_rule",
    "contact_information",
    "ticket_taxonomy",
    "developer_reference",
)


class KnowledgeSource(Base):
    __tablename__ = "knowledge_sources"
    __table_args__ = (
        CheckConstraint("authority BETWEEN 1 AND 5", name="ck_ks_authority"),
        CheckConstraint(
            "source_type IN ('website','pdf','internal','manual','taxonomy','api')",
            name="ck_ks_type",
        ),
        Index("idx_ks_enabled_type", "enabled", "source_type"),
    )

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    canonical_uri = Column(Text, nullable=False, unique=True)
    source_type = Column(String(32), nullable=False)
    authority = Column(Integer, nullable=False, default=3)
    audience = Column(String(64), nullable=False, default="customer")
    language = Column(String(16), nullable=False, default="en")
    region = Column(String(32), nullable=False, default="IN")
    owner = Column(String(255), nullable=True)
    crawl_policy = Column(JSON, nullable=False, default=dict)
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class KnowledgeSnapshot(Base):
    __tablename__ = "knowledge_snapshots"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued','fetching','fetched','normalized','failed')",
            name="ck_ksnap_status",
        ),
        UniqueConstraint("source_id", "content_hash", name="uq_ksnap_source_hash"),
        Index("idx_ksnap_source_created", "source_id", "created_at"),
    )

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(Uuid(as_uuid=True), ForeignKey("knowledge_sources.id", ondelete="CASCADE"), nullable=False)
    requested_uri = Column(Text, nullable=False)
    final_uri = Column(Text, nullable=True)
    status = Column(String(24), nullable=False, default="queued")
    http_status = Column(Integer, nullable=True)
    content_type = Column(String(255), nullable=True)
    etag = Column(String(255), nullable=True)
    last_modified = Column(String(255), nullable=True)
    content_hash = Column(String(64), nullable=True)
    raw_storage_key = Column(Text, nullable=True)
    size_bytes = Column(BigInteger, nullable=False, default=0)
    quality_score = Column(Float, nullable=True)
    warnings = Column(JSON, nullable=False, default=list)
    error = Column(Text, nullable=True)
    fetched_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"
    __table_args__ = (
        UniqueConstraint("snapshot_id", name="uq_kdoc_snapshot"),
        Index("idx_kdoc_canonical_uri", "canonical_uri"),
    )

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    snapshot_id = Column(Uuid(as_uuid=True), ForeignKey("knowledge_snapshots.id", ondelete="CASCADE"), nullable=False)
    canonical_uri = Column(Text, nullable=False)
    title = Column(Text, nullable=False)
    canonical_markdown = Column(Text, nullable=False)
    extractor = Column(String(64), nullable=False)
    extractor_version = Column(String(64), nullable=True)
    language = Column(String(16), nullable=False, default="en")
    metadata_json = Column(JSON, nullable=False, default=dict)
    quality_score = Column(Float, nullable=True)
    warnings = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class KnowledgeUnit(Base):
    __tablename__ = "knowledge_units"
    __table_args__ = (
        CheckConstraint(
            "unit_type IN (" + ",".join(f"'{value}'" for value in KNOWLEDGE_UNIT_TYPES) + ")",
            name="ck_ku_type",
        ),
        CheckConstraint("status IN ('draft','approved','retired')", name="ck_ku_status"),
        CheckConstraint("authority BETWEEN 1 AND 5", name="ck_ku_authority"),
        Index("idx_ku_status_type", "status", "unit_type"),
        Index("idx_ku_product_topic", "product", "topic"),
        Index("idx_ku_search_vector", "search_vector", postgresql_using="gin"),
    )

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    stable_key = Column(String(255), nullable=False, unique=True)
    document_id = Column(Uuid(as_uuid=True), ForeignKey("knowledge_documents.id", ondelete="SET NULL"), nullable=True)
    unit_type = Column(String(32), nullable=False)
    title = Column(Text, nullable=False)
    question = Column(Text, nullable=True)
    answer = Column(Text, nullable=False)
    voice_answer = Column(Text, nullable=True)
    retrieval_text = Column(Text, nullable=False)
    product = Column(String(128), nullable=True)
    device = Column(String(128), nullable=True)
    topic = Column(String(128), nullable=True)
    issue_family = Column(String(128), nullable=True)
    intents = Column(JSON, nullable=False, default=list)
    audience = Column(String(64), nullable=False, default="customer")
    language = Column(String(16), nullable=False, default="en")
    region = Column(String(32), nullable=False, default="IN")
    authority = Column(Integer, nullable=False, default=3)
    source_uri = Column(Text, nullable=False)
    source_label = Column(String(255), nullable=False)
    effective_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    requires_auth = Column(Boolean, nullable=False, default=False)
    requires_live_api = Column(Boolean, nullable=False, default=False)
    escalation_required = Column(Boolean, nullable=False, default=False)
    ticket_candidates = Column(JSON, nullable=False, default=list)
    metadata_json = Column(JSON, nullable=False, default=dict)
    version = Column(Integer, nullable=False, default=1)
    status = Column(String(16), nullable=False, default="draft")
    review_notes = Column(Text, nullable=True)
    approved_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    approved_at = Column(DateTime(timezone=True), nullable=True)
    content_hash = Column(String(64), nullable=False)
    search_vector = Column(TSVECTOR, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class KnowledgeEmbedding(Base):
    __tablename__ = "knowledge_embeddings"
    __table_args__ = (
        UniqueConstraint("unit_id", "provider", "model", name="uq_kemb_unit_model"),
        Index(
            "idx_kemb_vector",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    unit_id = Column(Uuid(as_uuid=True), ForeignKey("knowledge_units.id", ondelete="CASCADE"), nullable=False)
    provider = Column(String(32), nullable=False)
    model = Column(String(255), nullable=False)
    dimension = Column(Integer, nullable=False)
    content_hash = Column(String(64), nullable=False)
    embedding = Column(Vector(KNOWLEDGE_EMBEDDING_DIMENSION), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class KnowledgeAlias(Base):
    __tablename__ = "knowledge_aliases"
    __table_args__ = (
        UniqueConstraint("canonical", "alias", "language", name="uq_kalias_value"),
        Index("idx_kalias_alias_active", "alias", "active"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    canonical = Column(String(255), nullable=False)
    alias = Column(String(255), nullable=False)
    alias_type = Column(String(32), nullable=False, default="stt")
    product = Column(String(128), nullable=True)
    language = Column(String(16), nullable=False, default="en")
    priority = Column(Integer, nullable=False, default=0)
    active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class KnowledgeRelease(Base):
    __tablename__ = "knowledge_releases"
    __table_args__ = (
        CheckConstraint("status IN ('draft','published','retired')", name="ck_krel_status"),
        Index(
            "uq_krel_one_published",
            "status",
            unique=True,
            postgresql_where=text("status = 'published'"),
        ),
    )

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version = Column(String(64), nullable=False, unique=True)
    status = Column(String(16), nullable=False, default="draft")
    description = Column(Text, nullable=True)
    corpus_hash = Column(String(64), nullable=True)
    unit_count = Column(Integer, nullable=False, default=0)
    created_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    retired_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class KnowledgeReleaseUnit(Base):
    __tablename__ = "knowledge_release_units"
    __table_args__ = (
        PrimaryKeyConstraint("release_id", "unit_id", name="pk_krel_unit"),
        Index("idx_krelunit_unit", "unit_id"),
    )

    release_id = Column(Uuid(as_uuid=True), ForeignKey("knowledge_releases.id", ondelete="CASCADE"), nullable=False)
    unit_id = Column(Uuid(as_uuid=True), ForeignKey("knowledge_units.id", ondelete="RESTRICT"), nullable=False)
    added_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class KnowledgeConflict(Base):
    __tablename__ = "knowledge_conflicts"
    __table_args__ = (
        CheckConstraint("status IN ('open','resolved','ignored')", name="ck_kconf_status"),
        UniqueConstraint("left_unit_id", "right_unit_id", "conflict_type", name="uq_kconf_pair"),
        Index("idx_kconf_status", "status", "created_at"),
    )

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    left_unit_id = Column(Uuid(as_uuid=True), ForeignKey("knowledge_units.id", ondelete="CASCADE"), nullable=False)
    right_unit_id = Column(Uuid(as_uuid=True), ForeignKey("knowledge_units.id", ondelete="CASCADE"), nullable=False)
    conflict_type = Column(String(32), nullable=False)
    status = Column(String(16), nullable=False, default="open")
    details = Column(JSON, nullable=False, default=dict)
    resolution = Column(Text, nullable=True)
    resolved_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class KnowledgeJob(Base):
    __tablename__ = "knowledge_jobs"
    __table_args__ = (
        CheckConstraint("status IN ('queued','running','succeeded','failed')", name="ck_kjob_status"),
        Index("idx_kjob_claim", "status", "available_at", "created_at"),
    )

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_type = Column(String(32), nullable=False)
    status = Column(String(16), nullable=False, default="queued")
    source_id = Column(Uuid(as_uuid=True), ForeignKey("knowledge_sources.id", ondelete="CASCADE"), nullable=True)
    snapshot_id = Column(Uuid(as_uuid=True), ForeignKey("knowledge_snapshots.id", ondelete="SET NULL"), nullable=True)
    release_id = Column(Uuid(as_uuid=True), ForeignKey("knowledge_releases.id", ondelete="SET NULL"), nullable=True)
    payload = Column(JSON, nullable=False, default=dict)
    attempts = Column(Integer, nullable=False, default=0)
    available_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    claimed_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class KnowledgeFeedback(Base):
    __tablename__ = "knowledge_feedback"
    __table_args__ = (Index("idx_kfeedback_created", "created_at"),)

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id = Column(Uuid(as_uuid=True), ForeignKey("calls.id", ondelete="SET NULL"), nullable=True)
    unit_id = Column(Uuid(as_uuid=True), ForeignKey("knowledge_units.id", ondelete="SET NULL"), nullable=True)
    query_fingerprint = Column(String(64), nullable=False)
    route = Column(String(32), nullable=False)
    outcome = Column(String(32), nullable=False)
    details = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class TicketTaxonomyEntry(Base):
    __tablename__ = "ticket_taxonomy_entries"
    __table_args__ = (
        UniqueConstraint("ticket_code", "ticket_subcode", "remark", name="uq_ticket_taxonomy"),
        Index("idx_ticket_taxonomy_active_code", "active", "ticket_code", "ticket_subcode"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ticket_code = Column(String(128), nullable=False)
    ticket_subcode = Column(String(255), nullable=False)
    remark = Column(Text, nullable=False)
    source_status = Column(String(32), nullable=True)
    active = Column(Boolean, nullable=False, default=True)
    content_hash = Column(String(64), nullable=False)
    source_row = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


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
