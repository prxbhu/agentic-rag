"""
Database initialization script
Run this to create all tables and indexes
"""
import asyncio
import logging
from pathlib import Path
from sqlalchemy import text

import sqlparse
from app.database import engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INIT_SQL_PATH = Path(__file__).parent.parent / "database" / "init.sql"

async def init_database():
    """Initialize database schema from init.sql"""

    logger.info("Starting database initialization...")
    
    init_sql = INIT_SQL_PATH.read_text()

    statements = [
        stmt.strip()
        for stmt in sqlparse.split(init_sql)
        if stmt.strip()
    ]
    
    async with engine.begin() as conn:
        for stmt in statements:
            logger.debug(f"Executing SQL:\n{stmt[:120]}...")
            await conn.execute(text(stmt))

        await conn.commit()

    logger.info("Database initialization completed successfully!")

async def drop_all_tables():
    """Drop all tables (use with caution!)"""
    logger.warning("Dropping all tables...")

    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
        await conn.execute(text("GRANT ALL ON SCHEMA public TO postgres"))
        await conn.execute(text("GRANT ALL ON SCHEMA public TO public"))

        await conn.commit()

    logger.info("All tables dropped")




# async def init_database():
#     """Initialize database with all tables and extensions"""
    
#     logger.info("Starting database initialization...")
    
#     async with engine.begin() as conn:
#         # Enable pgvector extension
#         logger.info("Enabling pgvector extension...")
#         await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
#         await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
        
#         # Create all tables
#         logger.info("Creating tables...")
#         await conn.run_sync(Base.metadata.create_all)
        
#         # Create HNSW index for vector similarity search
#         logger.info("Creating HNSW index for embeddings...")
#         await conn.execute(text("""
#             CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks 
#             USING hnsw (embedding vector_cosine_ops)
#             WITH (m = 16, ef_construction = 64)
#         """))
        
#         # Create full-text search index for BM25
#         logger.info("Creating full-text search index...")
#         await conn.execute(text("""
#             CREATE INDEX IF NOT EXISTS idx_chunks_content_fts ON chunks 
#             USING gin(to_tsvector('english', content))
#         """))
        
#         # Create additional indexes
#         logger.info("Creating additional indexes...")
#         await conn.execute(text("""
#             CREATE INDEX IF NOT EXISTS idx_resources_workspace ON resources(workspace_id)
#         """))
#         await conn.execute(text("""
#             CREATE INDEX IF NOT EXISTS idx_resources_hash ON resources(content_hash)
#         """))
#         await conn.execute(text("""
#             CREATE INDEX IF NOT EXISTS idx_chunks_resource ON chunks(resource_id)
#         """))
#         await conn.execute(text("""
#             CREATE INDEX IF NOT EXISTS idx_chunks_workspace ON chunks(workspace_id)
#         """))
#         await conn.execute(text("""
#             CREATE INDEX IF NOT EXISTS idx_chunks_citation_count ON chunks(citation_count DESC)
#         """))
        
#         # Create hybrid search function
#         logger.info("Creating hybrid search function...")
#         await conn.execute(text("""
#             CREATE OR REPLACE FUNCTION hybrid_search(
#                 query_embedding vector(768),
#                 query_text TEXT,
#                 workspace_filter UUID,
#                 result_limit INT DEFAULT 20,
#                 semantic_weight DECIMAL DEFAULT 0.6,
#                 bm25_weight DECIMAL DEFAULT 0.4
#             )
#             RETURNS TABLE (
#                 chunk_id UUID,
#                 content TEXT,
#                 resource_id UUID,
#                 combined_score DECIMAL,
#                 semantic_score DECIMAL,
#                 bm25_score DECIMAL
#             ) AS $$
#             BEGIN
#                 RETURN QUERY
#                 WITH semantic_results AS (
#                     SELECT 
#                         c.id,
#                         c.content,
#                         c.resource_id,
#                         1 - (c.embedding <=> query_embedding) AS similarity
#                     FROM chunks c
#                     WHERE c.workspace_id = workspace_filter
#                     ORDER BY c.embedding <=> query_embedding
#                     LIMIT result_limit * 2
#                 ),
#                 bm25_results AS (
#                     SELECT 
#                         c.id,
#                         c.content,
#                         c.resource_id,
#                         ts_rank_cd(to_tsvector('english', c.content), 
#                                   plainto_tsquery('english', query_text)) AS rank
#                     FROM chunks c
#                     WHERE c.workspace_id = workspace_filter
#                         AND to_tsvector('english', c.content) @@ plainto_tsquery('english', query_text)
#                     ORDER BY rank DESC
#                     LIMIT result_limit * 2
#                 )
#                 SELECT 
#                     COALESCE(s.id, b.id) AS chunk_id,
#                     COALESCE(s.content, b.content) AS content,
#                     COALESCE(s.resource_id, b.resource_id) AS resource_id,
#                     (COALESCE(s.similarity, 0) * semantic_weight + 
#                      COALESCE(b.rank, 0) * bm25_weight) AS combined_score,
#                     COALESCE(s.similarity, 0) AS semantic_score,
#                     COALESCE(b.rank, 0) AS bm25_score
#                 FROM semantic_results s
#                 FULL OUTER JOIN bm25_results b ON s.id = b.id
#                 ORDER BY combined_score DESC
#                 LIMIT result_limit;
#             END;
#             $$ LANGUAGE plpgsql;
#         """))
        
#         # Create default workspace
#         logger.info("Creating default workspace...")
#         await conn.execute(text("""
#             INSERT INTO workspaces (id, name, workspace_type) 
#             VALUES (uuid_generate_v4(), 'Default Workspace', 'personal')
#             ON CONFLICT DO NOTHING
#         """))
        
#         await conn.commit()
    
#     logger.info("Database initialization completed successfully!")


# async def drop_all_tables():
#     """Drop all tables (use with caution!)"""
#     logger.warning("Dropping all tables...")
    
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.drop_all)
    
#     logger.info("All tables dropped")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--drop":
        asyncio.run(drop_all_tables())
    
    asyncio.run(init_database())