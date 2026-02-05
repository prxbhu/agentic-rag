# Production-Ready Agentic RAG System

A complete, production-grade Retrieval-Augmented Generation (RAG) system built with LangChain, LangGraph, and open-source models. This system features an autonomous agentic workflow capable of self-reflection, query decomposition, and answer refinement.

## 🚀 Features

### Core Capabilities
- **Agentic Workflow**: Uses LangGraph for orchestrating complex reasoning loops (Decomposition → Search → Reflect → Refine).
- **Advanced Ingestion**: Powered by **Docling** for high-fidelity parsing of PDFs (with OCR), Office docs (DOCX, XLSX, PPTX), images, and HTML.
- **Async-First Architecture**: Celery workers handle embeddings; API responds immediately
- **Hybrid Search**: Semantic (pgvector) + BM25 (PostgreSQL FTS) with reciprocal rank fusion.
- **Advanced Reranking**: Multi-stage reranking using Cross-Encoders (BGE-Reranker) and MMR for diversity.
- **Multi-Factor Ranking**: Combines relevance, recency, specificity, citation frequency, and source quality
- **Citation Verification**: Post-generation validation to prevent hallucinations by verifying claims against source text.
- **Self-Correction**: The agent evaluates its own answers and enters a refinement loop if quality standards aren't met.
- **Hardware Adaptation**: Auto-detects NVIDIA/AMD GPUs or Apple Metal to optimize model loading.
- **Content Deduplication**: SHA-256 hashing prevents re-processing identical documents

### Tech Stack
- **Backend**: FastAPI + Python 3.12+
- **Database**: PostgreSQL 16 + pgvector (with async SQLAlchemy)
- **Vector Search**: HNSW indexing for <200ms searches
- **Task Queue**: Celery + Redis
- **LLM Support**: 
  - **Ollama** (Local inference)
  - **Google Vertex AI** (Gemini models)
  - **vLLM** (High-throughput serving)
- **Ingestion**: Docling (IBM) for layout-aware document parsing
- **Embeddings**: sentence-transformers/all-mpnet-base-v2 (768-dim)
- **Orchestration**: LangGraph state machines
- **Frontend**: React 18 + TypeScript + Vite + Tailwind
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

# Run the automated setup script
chmod +x quick_setup.sh
./quick_setup.sh

cd backend

# Create environment file
cp .env.example .env

# Edit .env with your settings
nano .env
```

### 2. Environment Configuration

```bash
# .env
# Database & Queue
DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_db
REDIS_URL=redis://localhost:6379/0
GEMINI_API_KEY=your_gemini_key_optional

# LLM Selection
OLLAMA_BASE_URL=http://localhost:11434
# To use Google Vertex AI (Gemini):
GOOGLE_APPLICATION_CREDENTIALS=./backend/vertex.json
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
# To use vLLM:
VLLM_BASE_URL=http://your-vllm-instance:8000

# Hardware settings
ENABLE_GPU=auto 
FORCE_CPU=false  # Set to true to force CPU-only mode

# Model configuration
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
CHAT_MODEL=mistral:7b-instruct-q4_0
```

### 3. Environment Setup
```bash
cd backend
uv venv
uv pip install -r requirements.txt
cd..
cd frontend
npm i
```

### 4. Launch Stack Locally

```bash
# Start backend services
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload         

# Start frontend services
cd frontend
npm run dev
```

### 4. Launch Stack through Docker

```bash
# Start all services
docker-compose up -d

# Wait for models to download (first run only, ~2GB)
docker-compose logs -f ollama

# Once Ollama is ready, pull the model
docker exec -it rag_ollama ollama pull mistral:7b-instruct-q4_0
```

### 5. Verify Services

```bash
# Check all services are healthy
docker-compose ps

# Test API
curl http://localhost:8000/api/health

# View logs
docker-compose logs -f backend
docker-compose logs -f celery_worker
```

### 6. Access Application

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
│   │   │   ├── workspaces.py    # Workspace endpoints
│   │   │   └── health.py           # Health checks
│   │   ├── services/               # Business logic
│   │   │   ├── ingestion.py        # Document parsing
│   │   │   ├── embedding.py        # Vector generation
│   │   │   ├── enhanced_rag.py     # Reranking & Query Decomposition
│   │   │   ├── search.py           # Hybrid search
│   │   │   ├── ranking.py          # Scoring logic
│   │   │   ├── llm_service.py      # LLM abstraction
│   │   │   ├── citation.py         # Citation verification
│   │   │   └── hardware.py         # GPU/CPU detection
│   │   ├── agents/                 # LangGraph agents
│   │   │   ├── rag_agent.py        # Main RAG orchestration
│   │   │   └── tools.py            # LangChain tools
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
├── frontend/
├── docker-compose.yml              # 7-service stack
└── README.md
```

## 🔧 API Usage

### Create Workspace

```bash
curl -X POST "http://localhost:8000/api/workspaces/" \
  -H "Content-Type: application/json" \
  -d '{"name": "Research Project", "workspace_type": "personal"}'
```

### Upload Documents
Supports PDF, DOCX, XLSX, PPTX, Images, etc.
```bash
curl -X POST "http://localhost:8000/api/resources/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@paper.pdf" \
  -F "workspace_id=<workspace-uuid>"
```

### Check Embedding Status

```bash
curl "http://localhost:8000/api/resources/<resource-id>/embedding-status"
```

### Stream Chat Response

```bash
curl -X POST "http://localhost:8000/api/conversations/<conv-id>/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Analyze the revenue growth mentioned in the documents.",
    "workspace_id": "<workspace-uuid>"
  }'
```

## 🎯 RAG Pipeline Flow

```
1. Query Expansion
   ├─> Generate 3-5 query variants
   └─> Include synonyms and reformulations

2. Hybrid Search
   ├─> Hybrid Search (Vector + Keyword)
   ├─> Advanced Reranking (Cross-Encoder / Hybrid Score)
   └─> Diversity Check (MMR)

3. Context Assembly
   ├─> Token Budgeting (allocate tokens for primary vs supporting sources)
   └─> Metadata Injection (recency, source quality)
4. LLM Generation
   └─> LLM generates initial answer 
   └─> Strict citation requirements

5. Verification & Reflection
   ├─> Verify Citations: Check if [Source X] actually supports the claim
   ├─> Self-Reflection: LLM grades its own answer (0-10 score)
   │    └─> If Score < 6: Loop back to Refinement
   └─> Refine Response: Fix missing citations or logic gaps
```

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
celery -A app.tasks.celery_app worker \         
  --loglevel=info \
  --pool=solo \
  -Q embeddings
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
docker exec -it rag_ollama ollama pull gemma3:4b
```

### Out of Memory

```bash
# Reduce batch size in .env
BATCH_SIZE=16

# Use smaller model
CHAT_MODEL=gemma3:1b

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
6. **Rate limit** API endpoints
7. **Sanitize user inputs** to prevent injection attacks

## 📝 License

MIT License - see LICENSE file for details.


## 📚 References

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [Ollama Models](https://ollama.ai/library)
- [Sentence Transformers](https://www.sbert.net/)
- [Docling](https://docling-project.github.io/docling/)