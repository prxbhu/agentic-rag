# Production-Ready Agentic RAG System

A complete, production-grade Retrieval-Augmented Generation (RAG) system built with LangChain, LangGraph, and open-source models.

## 🚀 Features

### Core Capabilities
- **Async-First Architecture**: Celery workers handle embeddings; API responds immediately
- **Hybrid Search**: Semantic + BM25 with reciprocal rank fusion
- **Multi-Factor Ranking**: Combines relevance, recency, specificity, citation frequency, and source quality
- **Citation Verification**: Post-generation validation of LLM claims against sources
- **Hardware Adaptation**: Auto-detects GPU/CPU, adjusts models and timeouts
- **Content Deduplication**: SHA-256 hashing prevents re-processing identical documents

### Tech Stack
- **Backend**: FastAPI + Python 3.11+
- **Database**: PostgreSQL 16 + pgvector
- **Vector Search**: HNSW indexing for <200ms searches
- **Task Queue**: Celery + Redis
- **LLM**: Ollama (Mistral 7B 4-bit quantized) or Google Gemini
- **Embeddings**: sentence-transformers/all-mpnet-base-v2 (768-dim)
- **Orchestration**: LangGraph for agentic workflows
- **Frontend**: React 19 + TypeScript + Vite

## 📋 Prerequisites

- Docker & Docker Compose
- 8GB+ RAM (16GB recommended)
- Optional: NVIDIA GPU with CUDA support for faster inference

## 🛠️ Quick Start

### 1. Clone and Setup

```bash
# Clone repository
git clone <your-repo-url>
cd agentic-rag-system

# Create environment file
cp .env.example .env

# Edit .env with your settings
nano .env
```

### 2. Environment Configuration

```bash
# .env
POSTGRES_PASSWORD=your_secure_password
GEMINI_API_KEY=your_gemini_key_optional

# Hardware settings
ENABLE_GPU=auto  # auto, true, or false
FORCE_CPU=false  # Set to true to force CPU-only mode

# Model configuration
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
CHAT_MODEL=mistral:7b-instruct-q4_0
```

### 3. Launch Stack

```bash
# Start all services
docker-compose up -d

# Wait for models to download (first run only, ~2GB)
docker-compose logs -f ollama

# Once Ollama is ready, pull the model
docker exec -it rag_ollama ollama pull mistral:7b-instruct-q4_0
```

### 4. Verify Services

```bash
# Check all services are healthy
docker-compose ps

# Test API
curl http://localhost:8000/api/health

# View logs
docker-compose logs -f backend
docker-compose logs -f celery_worker
```

### 5. Access Application

- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **API Base**: http://localhost:8000/api

## 📁 Project Structure

```
agentic-rag-system/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app
│   │   ├── config.py               # Configuration
│   │   ├── database.py             # DB connection
│   │   ├── api/                    # REST endpoints
│   │   │   ├── resources.py        # Document upload
│   │   │   ├── conversations.py    # Chat endpoints
│   │   │   └── health.py           # Health checks
│   │   ├── services/               # Business logic
│   │   │   ├── ingestion.py        # Document parsing
│   │   │   ├── embedding.py        # Vector generation
│   │   │   ├── search.py           # Hybrid search
│   │   │   ├── ranking.py          # Multi-factor ranking
│   │   │   ├── llm_service.py      # LLM abstraction
│   │   │   ├── citation.py         # Citation verification
│   │   │   └── hardware.py         # GPU/CPU detection
│   │   ├── agents/                 # LangGraph agents
│   │   │   ├── rag_agent.py        # Main RAG orchestration
│   │   │   ├── tools.py            # LangChain tools
│   │   │   └── prompts.py          # Prompt templates
│   │   ├── models/                 # Data models
│   │   │   ├── schemas.py          # Pydantic models
│   │   │   └── database_models.py  # SQLAlchemy models
│   │   └── tasks/                  # Celery tasks
│   │       ├── celery_app.py       # Celery config
│   │       └── embedding_tasks.py  # Async workers
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── ChatInterface.tsx
│   │   │   ├── DocumentUpload.tsx
│   │   │   └── SourceCitations.tsx
│   │   └── api/
│   │       └── client.ts           # API client
│   ├── package.json
│   └── Dockerfile
├── database/
│   ├── init.sql                    # Schema + pgvector
│   └── migrations/
├── docker-compose.yml              # 7-service stack
└── README.md
```

## 🔧 API Usage

### Upload Documents

```bash
curl -X POST "http://localhost:8000/api/resources/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "workspace_id=<workspace-uuid>"
```

### Check Embedding Status

```bash
curl "http://localhost:8000/api/resources/<resource-id>/embedding-status"
```

### Send Chat Message

```bash
curl -X POST "http://localhost:8000/api/conversations/<conv-id>/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "What are the key findings?",
    "workspace_id": "<workspace-uuid>"
  }'
```

## 🎯 RAG Pipeline Flow

```
1. Query Expansion
   ├─> Generate 3-5 query variants
   └─> Include synonyms and reformulations

2. Hybrid Search
   ├─> Semantic search (pgvector cosine similarity)
   ├─> BM25 keyword search (PostgreSQL FTS)
   └─> Reciprocal rank fusion merge

3. Multi-Factor Ranking
   ├─> Base relevance (40%)
   ├─> Citation frequency (15%)
   ├─> Recency (15%)
   ├─> Specificity (15%)
   └─> Source quality (15%)

4. Context Assembly
   ├─> Token budget: 2000 default
   ├─> Primary sources: 60%
   ├─> Supporting context: 30%
   └─> Metadata: 10%

5. LLM Generation
   ├─> Streaming response
   ├─> Strict citation requirements
   └─> Anti-hallucination prompts

6. Citation Verification
   ├─> Extract [Source N] references
   ├─> Validate against source chunks
   └─> Flag mismatches
```

## 🔍 Hardware Optimization

### GPU Detection

The system automatically detects and optimizes for:
- **NVIDIA GPUs**: Uses CUDA acceleration
- **AMD GPUs**: Uses ROCm support
- **Apple Silicon**: Uses Metal acceleration
- **CPU-only**: Adjusts batch sizes and timeouts

### Model Selection

| Hardware | Model |
|----------|-------|
| GPU + 16GB+ RAM | mistral:7b-instruct |
| GPU + 8GB RAM | mistral:7b-instruct-q4_0 |
| CPU + 16GB RAM | mistral:7b-instruct-q4_0 |
| CPU + 8GB RAM | gemma:2b-instruct-q4_0 |
| CPU + <8GB RAM | phi:2.7b-instruct-q4_0 |

## 📊 Performance Tuning

### Database Optimization

```sql
-- Adjust HNSW index parameters for your workload
CREATE INDEX idx_chunks_embedding ON chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (
  m = 16,              -- Connections per layer (higher = better recall, more memory)
  ef_construction = 64 -- Build quality (higher = better index, slower build)
);

-- Query-time tuning
SET hnsw.ef_search = 100;  -- Higher = better recall, slower search
```

### Celery Workers

```bash
# Adjust concurrency based on CPU cores
# In docker-compose.yml, celery_worker service:
command: celery -A app.tasks.celery_app worker --loglevel=info --concurrency=4
```

### Embedding Batch Size

```python
# Automatically adjusted based on hardware
# Manual override in config.py:
BATCH_SIZE = 64  # GPU with 16GB+ RAM
BATCH_SIZE = 32  # GPU with 8GB RAM
BATCH_SIZE = 16  # CPU mode
```

## 🐛 Troubleshooting

### Ollama Model Not Found

```bash
docker exec -it rag_ollama ollama pull mistral:7b-instruct-q4_0
```

### Out of Memory

```bash
# Reduce batch size in .env
BATCH_SIZE=16

# Use smaller model
CHAT_MODEL=phi:2.7b-instruct-q4_0

# Force CPU mode
FORCE_CPU=true
```

### Slow Searches

```sql
-- Rebuild HNSW index
REINDEX INDEX CONCURRENTLY idx_chunks_embedding;

-- Or adjust ef_search
SET hnsw.ef_search = 50;  -- Lower = faster, less accurate
```

### Celery Tasks Stuck

```bash
# Restart worker
docker-compose restart celery_worker

# Purge queue
docker exec -it rag_redis redis-cli FLUSHDB
```

## 📈 Monitoring

### Health Checks

```bash
# Overall system health
curl http://localhost:8000/api/health

# Database connection
curl http://localhost:8000/api/health/db

# Ollama status
curl http://localhost:11434/api/tags
```

### Logs

```bash
# Backend logs
docker-compose logs -f backend

# Celery worker logs
docker-compose logs -f celery_worker

# Database logs
docker-compose logs -f postgres
```

### Metrics

Access Celery Flower (optional):
```bash
docker run -p 5555:5555 mher/flower:latest \
  --broker=redis://localhost:6379/0
```

## 🔒 Security Considerations

1. **Change default passwords** in `.env`
2. **Use HTTPS** in production
3. **Enable authentication** for API endpoints
4. **Restrict CORS origins** in `config.py`
5. **Scan uploaded files** for malware
6. **Rate limit** API endpoints
7. **Sanitize user inputs** to prevent injection attacks

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

## 📚 References

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [Ollama Models](https://ollama.ai/library)
- [Sentence Transformers](https://www.sbert.net/)

## 💬 Support

- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Email: support@example.com