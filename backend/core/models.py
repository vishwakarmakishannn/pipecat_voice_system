from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, Index
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
    
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    memories = relationship("UserMemory", back_populates="user", cascade="all, delete-orphan")
    memory_chunks = relationship("MemoryChunk", back_populates="user", cascade="all, delete-orphan")
    rag_files = relationship("RagFile", back_populates="user", cascade="all, delete-orphan")
    rag_chunks = relationship("RagChunk", back_populates="user", cascade="all, delete-orphan")

class Conversation(Base):
    __tablename__ = "conversations"

    __table_args__ = (
        Index("idx_conversation_user_updated", "user_id", "updated_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, default="New conversation")
    summary = Column(Text, nullable=True, default="")
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    memory_chunks = relationship("MemoryChunk", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"

    __table_args__ = (
        Index("idx_message_conversation_created", "conversation_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)  # 'You' or 'Aura'
    content = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    conversation = relationship("Conversation", back_populates="messages")

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
    source_message_id = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user = relationship("User", back_populates="memories")

class MemoryChunk(Base):
    __tablename__ = "memory_chunks"
    __table_args__ = (
        UniqueConstraint("conversation_id", "message_start_id", "message_end_id", name="uq_memory_chunk_message_window"),
        Index(
            "idx_memory_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    message_start_id = Column(Integer, nullable=False)
    message_end_id = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    embedding = Column(Vector(MEMORY_EMBEDDING_DIMENSION), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user = relationship("User", back_populates="memory_chunks")
    conversation = relationship("Conversation", back_populates="memory_chunks")


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
    size_bytes = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False, default="processing")
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
