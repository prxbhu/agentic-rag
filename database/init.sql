-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Workspaces for multi-tenancy
CREATE TABLE IF NOT EXISTS workspaces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    workspace_type VARCHAR(50) DEFAULT 'personal', -- personal, team, hybrid
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

DROP TRIGGER IF EXISTS trg_workspaces_updated_at ON workspaces;
CREATE TRIGGER trg_workspaces_updated_at
BEFORE UPDATE ON workspaces
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Resources table for tracking documents
CREATE TABLE IF NOT EXISTS resources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    filename VARCHAR(512) NOT NULL,
    file_type VARCHAR(50) NOT NULL, -- pdf, docx, txt, md, xlsx
    content_hash VARCHAR(64) UNIQUE NOT NULL, -- SHA-256 for deduplication
    is_duplicate_of UUID REFERENCES resources(id) ON DELETE SET NULL,
    file_size_bytes BIGINT NOT NULL,
    status VARCHAR(50) DEFAULT 'pending', -- pending, processing, completed, failed
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_resources_workspace ON resources(workspace_id);
CREATE INDEX IF NOT EXISTS idx_resources_hash ON resources(content_hash);
CREATE INDEX IF NOT EXISTS idx_resources_status ON resources(status);

DROP TRIGGER IF EXISTS trg_resources_updated_at ON resources;
CREATE TRIGGER trg_resources_updated_at
BEFORE UPDATE ON resources
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Chunks table optimized for vector retrieval
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_id UUID REFERENCES resources(id) ON DELETE CASCADE,
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(768), -- sentence-transformers/all-mpnet-base-v2 dimension
    parent_content TEXT, -- Content of the parent chunk (larger context)
    parent_chunk_id UUID, -- Link to parent (logical grouping)
    chunk_index INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    chunk_metadata JSONB DEFAULT '{}', -- stores section, page_num, etc.
    citation_count INTEGER DEFAULT 0, -- tracks usage frequency
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_chunks_resource ON chunks(resource_id);
CREATE INDEX IF NOT EXISTS idx_chunks_workspace ON chunks(workspace_id);
CREATE INDEX IF NOT EXISTS idx_chunks_citation_count ON chunks(citation_count DESC);

-- Full-text search index for BM25 hybrid retrieval
CREATE INDEX IF NOT EXISTS idx_chunks_content_fts ON chunks 
    USING gin(to_tsvector('english', content));

-- Conversations for chat history
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    title VARCHAR(512),
    model_name VARCHAR(100) NOT NULL,
    system_prompt TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversations_workspace ON conversations(workspace_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created ON conversations(created_at DESC);

DROP TRIGGER IF EXISTS trg_conversations_updated_at ON conversations;
CREATE TRIGGER trg_conversations_updated_at
BEFORE UPDATE ON conversations
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Messages with source tracking
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL, -- user, assistant, system
    content TEXT NOT NULL,
    citations JSONB DEFAULT '[]', -- [{chunk_id, source_name, confidence}]
    source_chunks UUID[] DEFAULT '{}', -- array of chunk IDs used
    model_metadata JSONB DEFAULT '{}', -- tokens, latency, etc.
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at DESC);

-- Embedding tasks for tracking async processing
CREATE TABLE IF NOT EXISTS embedding_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_id UUID REFERENCES resources(id) ON DELETE CASCADE,
    task_id VARCHAR(255) UNIQUE NOT NULL, -- Celery task ID
    status VARCHAR(50) DEFAULT 'pending', -- pending, processing, completed, failed
    chunks_processed INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_embedding_tasks_resource ON embedding_tasks(resource_id);
CREATE INDEX IF NOT EXISTS idx_embedding_tasks_status ON embedding_tasks(status);

-- Source quality scores (for ranking)
CREATE TABLE IF NOT EXISTS source_quality (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_id UUID REFERENCES resources(id) ON DELETE CASCADE,
    quality_score DECIMAL(3, 2) DEFAULT 0.5, -- 0.0 to 1.0
    specificity_score DECIMAL(3, 2) DEFAULT 0.5,
    recency_weight DECIMAL(3, 2) DEFAULT 0.5,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_source_quality_resource ON source_quality(resource_id);

DROP TRIGGER IF EXISTS trg_source_quality_updated_at ON source_quality;
CREATE TRIGGER trg_source_quality_updated_at
BEFORE UPDATE ON source_quality
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_resources_updated_at
    BEFORE UPDATE ON resources
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_workspaces_updated_at
    BEFORE UPDATE ON workspaces
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create default workspace
INSERT INTO workspaces (name, workspace_type) 
VALUES ('Default Workspace', 'personal')
ON CONFLICT DO NOTHING;


-- Function for hybrid search (semantic + BM25)
CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding vector(768),
    query_text TEXT,
    workspace_filter UUID,
    result_limit INT DEFAULT 20,
    semantic_weight DECIMAL DEFAULT 0.6,
    bm25_weight DECIMAL DEFAULT 0.4
)
RETURNS TABLE (
    chunk_id UUID,
    content TEXT,
    resource_id UUID,
    combined_score DECIMAL,
    semantic_score DECIMAL,
    bm25_score DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    WITH semantic_results AS (
        SELECT 
            c.id,
            c.content,
            c.resource_id,
            CAST(1 - (c.embedding <=> query_embedding) AS DECIMAL) AS similarity
        FROM chunks c
        WHERE c.workspace_id = workspace_filter
        ORDER BY c.embedding <=> query_embedding
        LIMIT result_limit * 2
    ),
    bm25_results AS (
        SELECT 
            c.id,
            c.content,
            c.resource_id,
            CAST(
                ts_rank_cd(
                    to_tsvector('english', c.content), 
                    plainto_tsquery('english', query_text)
                ) AS DECIMAL
            ) AS rank
        FROM chunks c
        WHERE c.workspace_id = workspace_filter
          AND to_tsvector('english', c.content) @@ plainto_tsquery('english', query_text)
        ORDER BY rank DESC
        LIMIT result_limit * 2
    )
    SELECT 
        COALESCE(s.id, b.id) AS chunk_id,
        COALESCE(s.content, b.content) AS content,
        COALESCE(s.resource_id, b.resource_id) AS resource_id,
        CAST(
            (COALESCE(s.similarity, 0) * semantic_weight + 
             COALESCE(b.rank, 0) * bm25_weight)
        AS DECIMAL) AS combined_score,
        COALESCE(s.similarity, 0) AS semantic_score,
        COALESCE(b.rank, 0) AS bm25_score
    FROM semantic_results s
    FULL OUTER JOIN bm25_results b ON s.id = b.id
    ORDER BY combined_score DESC
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;