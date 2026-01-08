"""
SQLAlchemy database models
"""
from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import (
    Column, String, Integer, BigInteger, DateTime, ForeignKey,
    Text, DECIMAL, ARRAY, Boolean
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.database import Base


class Workspace(Base):
    """Workspace model for multi-tenancy"""
    __tablename__ = "workspaces"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    workspace_type = Column(String(50), default="personal")  # personal, team, hybrid
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    resources = relationship("Resource", back_populates="workspace", cascade="all, delete-orphan")
    chunks = relationship("Chunk", back_populates="workspace", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="workspace", cascade="all, delete-orphan")


class Resource(Base):
    """Resource model for tracking documents"""
    __tablename__ = "resources"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"))
    filename = Column(String(512), nullable=False)
    file_type = Column(String(50), nullable=False)
    content_hash = Column(String(64), unique=True, nullable=False)
    is_duplicate_of = Column(UUID(as_uuid=True), ForeignKey("resources.id", ondelete="SET NULL"), nullable=True)
    file_size_bytes = Column(BigInteger, nullable=False)
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    resource_metadata = Column("metadata", JSONB, default={})
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    workspace = relationship("Workspace", back_populates="resources")
    chunks = relationship("Chunk", back_populates="resource", cascade="all, delete-orphan")
    source_quality = relationship("SourceQuality", back_populates="resource", uselist=False, cascade="all, delete-orphan")
    embedding_tasks = relationship("EmbeddingTask", back_populates="resource", cascade="all, delete-orphan")


class Chunk(Base):
    """Chunk model for document segments with embeddings"""
    __tablename__ = "chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    resource_id = Column(UUID(as_uuid=True), ForeignKey("resources.id", ondelete="CASCADE"))
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"))
    content = Column(Text, nullable=False)
    embedding = Column(Vector(768))  # 768-dimensional vector
    chunk_index = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=False)
    chunk_metadata = Column(JSONB, default={})
    citation_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    resource = relationship("Resource", back_populates="chunks")
    workspace = relationship("Workspace", back_populates="chunks")


class Conversation(Base):
    """Conversation model for chat history"""
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"))
    title = Column(String(512))
    model_name = Column(String(100), nullable=False)
    system_prompt = Column(Text)
    conversation_metadata = Column("metadata", JSONB, default={})
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    workspace = relationship("Workspace", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    """Message model for chat messages"""
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"))
    role = Column(String(50), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    citations = Column(JSONB, default=[])
    source_chunks = Column(ARRAY(UUID(as_uuid=True)), default=[])
    model_metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")


class EmbeddingTask(Base):
    """Embedding task model for tracking async processing"""
    __tablename__ = "embedding_tasks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    resource_id = Column(UUID(as_uuid=True), ForeignKey("resources.id", ondelete="CASCADE"))
    task_id = Column(String(255), unique=True, nullable=False)
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    chunks_processed = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    error_message = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    resource = relationship("Resource", back_populates="embedding_tasks")


class SourceQuality(Base):
    """Source quality model for ranking"""
    __tablename__ = "source_quality"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    resource_id = Column(UUID(as_uuid=True), ForeignKey("resources.id", ondelete="CASCADE"), unique=True)
    quality_score = Column(DECIMAL(3, 2), default=0.5)
    specificity_score = Column(DECIMAL(3, 2), default=0.5)
    recency_weight = Column(DECIMAL(3, 2), default=0.5)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    resource = relationship("Resource", back_populates="source_quality")