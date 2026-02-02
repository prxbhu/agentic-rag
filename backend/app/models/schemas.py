"""
Pydantic schemas for request/response validation
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


# Workspace Schemas
class WorkspaceBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    workspace_type: str = Field(default="personal")


class WorkspaceCreate(WorkspaceBase):
    pass


class WorkspaceResponse(WorkspaceBase):
    id: UUID
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


# Resource Schemas
class ResourceBase(BaseModel):
    filename: str
    file_type: str


class ResourceCreate(ResourceBase):
    workspace_id: UUID
    content_hash: str
    file_size_bytes: int


class ResourceResponse(ResourceBase):
    id: UUID
    workspace_id: UUID
    status: str
    chunk_count: Optional[int] = 0
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(from_attributes=True)


class ResourceUploadResponse(BaseModel):
    status: str
    message: str
    resource_id: UUID
    task_id: Optional[str] = None
    chunks_count: Optional[int] = None
    poll_url: Optional[str] = None


# Chunk Schemas
class ChunkBase(BaseModel):
    content: str
    chunk_index: int
    token_count: int


class ChunkCreate(ChunkBase):
    resource_id: UUID
    workspace_id: UUID
    chunk_metadata: Dict[str, Any] = Field(default_factory=dict)


class ChunkResponse(ChunkBase):
    id: UUID
    resource_id: UUID
    citation_count: int
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(from_attributes=True)


class ChunkWithScore(ChunkResponse):
    relevance_score: float
    semantic_score: Optional[float] = None
    bm25_score: Optional[float] = None


# Conversation Schemas
class ConversationBase(BaseModel):
    title: Optional[str] = None
    system_prompt: Optional[str] = None


class ConversationCreate(ConversationBase):
    workspace_id: UUID


class ConversationResponse(ConversationBase):
    id: UUID
    workspace_id: UUID
    model_name: str
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
    model_provider: Optional[str]

class ConversationWithMessages(ConversationResponse):
    messages: List["MessageResponse"] = Field(default_factory=list)


# Message Schemas
class MessageBase(BaseModel):
    content: str


class MessageCreate(MessageBase):
    workspace_id: UUID


class MessageResponse(MessageBase):
    id: UUID
    conversation_id: UUID
    role: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class MessageWithMetadata(MessageResponse):
    model_metadata: Dict[str, Any] = Field(default_factory=dict)
    source_chunks: List[UUID] = []


# Citation Schemas
class Citation(BaseModel):
    chunk_id: UUID
    resource_id: UUID
    content_preview: str
    relevance_score: float


class CitationVerification(BaseModel):
    passed: bool
    issues: List[Dict[str, Any]] = []
    total_citations: int
    total_sources: int


# Search Schemas
class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1)
    workspace_id: UUID
    limit: int = Field(default=10, ge=1, le=100)


class SearchResult(BaseModel):
    chunk_id: UUID
    content: str
    resource_id: UUID
    filename: str
    combined_score: float
    semantic_score: Optional[float] = None
    bm25_score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int


# Embedding Task Schemas
class EmbeddingTaskStatus(BaseModel):
    resource_id: UUID
    filename: str
    status: str
    progress: Dict[str, Any]
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


# RAG Pipeline Schemas
class RAGRequest(BaseModel):
    query: str = Field(..., min_length=1)
    workspace_id: UUID
    conversation_id: Optional[UUID] = None
    max_sources: int = Field(default=5, ge=1, le=20)


class RAGResponse(BaseModel):
    query: str
    response: str
    citations: List[Citation]
    metadata: Dict[str, Any]
    verification: Optional[CitationVerification] = None


# Health Check Schemas
class HealthStatus(BaseModel):
    status: str
    version: str
    service: str


class DetailedHealthStatus(HealthStatus):
    components: Dict[str, Dict[str, Any]]
    system: Dict[str, Any]


# Error Response Schema
class ErrorResponse(BaseModel):
    error: str
    message: str
    detail: Optional[Any] = None


# Query Expansion Schema
class QueryExpansion(BaseModel):
    original_query: str
    expanded_queries: List[str]


# Context Assembly Schema
class AssembledContext(BaseModel):
    context: str
    sources_used: List[UUID]
    token_count: int
    primary_sources: int
    supporting_sources: int


# Ranking Schema
class RankedResult(BaseModel):
    chunk_id: UUID
    content: str
    final_score: float
    base_relevance: float
    citation_frequency: float
    recency_score: float
    specificity_score: float
    source_quality: float
    has_conflict: bool = False


# Statistics Schema
class SystemStatistics(BaseModel):
    total_resources: int
    total_chunks: int
    total_conversations: int
    total_messages: int
    most_cited_chunks: List[Dict[str, Any]]


# Update forward references
ConversationWithMessages.model_rebuild()