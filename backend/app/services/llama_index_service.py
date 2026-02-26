import os
from urllib.parse import urlparse
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url

from app.config import settings

def init_llamaindex():
    """Initialize Global LlamaIndex settings and return the Vector Store."""
    
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=settings.EMBEDDING_MODEL
    )
    
    Settings.llm = Ollama(
        model=settings.CHAT_MODEL, 
        base_url=settings.OLLAMA_BASE_URL, 
        request_timeout=300.0
    )
    
    Settings.chunk_size = settings.DEFAULT_CHUNK_SIZE
    Settings.chunk_overlap = settings.CHUNK_OVERLAP

    db_url = make_url(settings.DATABASE_URL.replace("+asyncpg", ""))
    
    vector_store = PGVectorStore.from_params(
        database=db_url.database,
        host=db_url.host,
        password=db_url.password,
        port=db_url.port,
        user=db_url.username,
        table_name="llama_chunks", 
        embed_dim=settings.EMBEDDING_DIMENSION,
    )
    
    return vector_store