"""
Configuration management for the RAG system
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql://postgres:IAmCrazy1@localhost:5432/rag_db",
        description="PostgreSQL connection string"
    )
    
    # Redis
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string"
    )
    
    # Celery
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Celery broker URL"
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379/0",
        description="Celery result backend URL"
    )
    
    # Embedding Model
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="HuggingFace embedding model"
    )
    EMBEDDING_DIMENSION: int = Field(
        default=768,
        description="Embedding vector dimension"
    )
    DEFAULT_CHUNK_SIZE: int = Field(
        default=512,
        description="Default chunk size in tokens"
    )
    CHUNK_OVERLAP: int = Field(
        default=50,
        description="Overlap between chunks in tokens"
    )
    BATCH_SIZE: int = Field(
        default=32,
        description="Batch size for embedding generation"
    )
    
    # LLM Configuration
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )
    CHAT_MODEL: str = Field(
        default="gemma3n:e2b",
        description="Default chat model"
    )
    GEMINI_API_KEY: Optional[str] = Field(
        default=None,
        description="Google Gemini API key (optional)"
    )
    
    # Hardware Configuration
    ENABLE_GPU: str = Field(
        default="auto",
        description="Enable GPU: 'true', 'false', or 'auto'"
    )
    FORCE_CPU: bool = Field(
        default=False,
        description="Force CPU-only mode"
    )
    
    # Search Configuration
    HYBRID_SEARCH_SEMANTIC_WEIGHT: float = Field(
        default=0.6,
        description="Weight for semantic search in hybrid mode"
    )
    HYBRID_SEARCH_BM25_WEIGHT: float = Field(
        default=0.4,
        description="Weight for BM25 search in hybrid mode"
    )
    MAX_SEARCH_RESULTS: int = Field(
        default=20,
        description="Maximum number of search results to retrieve"
    )
    
    # Ranking Configuration
    BASE_RELEVANCE_WEIGHT: float = 0.40
    CITATION_FREQ_WEIGHT: float = 0.15
    RECENCY_WEIGHT: float = 0.15
    SPECIFICITY_WEIGHT: float = 0.15
    SOURCE_QUALITY_WEIGHT: float = 0.15
    
    # Context Assembly
    DEFAULT_TOKEN_BUDGET: int = Field(
        default=30000,
        description="Token budget for context assembly"
    )
    PRIMARY_SOURCES_RATIO: float = Field(
        default=0.75,
        description="Ratio of tokens for primary sources"
    )
    SUPPORTING_CONTEXT_RATIO: float = Field(
        default=0.20,
        description="Ratio of tokens for supporting context"
    )
    METADATA_RATIO: float = Field(
        default=0.05,
        description="Ratio of tokens for metadata"
    )
    
    # API Configuration
    API_TITLE: str = "Agentic RAG System"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Production-ready RAG with LangChain and LangGraph"
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8000"]
    
    # File Upload
    MAX_UPLOAD_SIZE_MB: int = Field(
        default=50,
        description="Maximum file upload size in MB"
    )
    ALLOWED_EXTENSIONS: set = {
        "pdf", "docx", "doc", "txt", "md", "xlsx", "xls", "csv"
    }
    UPLOAD_DIR: str = Field(
        default="./uploads",
        description="Directory for uploaded files"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding="utf-8"
        case_sensitive = True
        extra="allow"


# Global settings instance
settings = Settings()


# Create upload directory if it doesn't exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)