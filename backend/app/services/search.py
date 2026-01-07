"""
Hybrid search service combining semantic and BM25 search
"""
import logging
from typing import List, Dict, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.config import settings
from app.services.embedding import embedding_service

logger = logging.getLogger(__name__)


class SearchService:
    """Service for hybrid search (semantic + BM25)"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def semantic_search(
        self,
        query: str,
        workspace_id: UUID,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector similarity
        
        Args:
            query: Search query
            workspace_id: Workspace to search in
            limit: Maximum number of results
            
        Returns:
            List of search results with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = embedding_service.embed_text(query)
            
            # Convert to PostgreSQL vector format
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
            
            # Semantic search query
            sql = text("""
                SELECT 
                    c.id as chunk_id,
                    c.content,
                    c.resource_id,
                    c.chunk_metadata,
                    r.filename,
                    1 - (c.embedding <=> :query_embedding::vector) as similarity
                FROM chunks c
                JOIN resources r ON c.resource_id = r.id
                WHERE c.workspace_id = :workspace_id
                ORDER BY c.embedding <=> :query_embedding::vector
                LIMIT :limit
            """)
            
            result = await self.db.execute(
                sql,
                {
                    "query_embedding": embedding_str,
                    "workspace_id": str(workspace_id),
                    "limit": limit
                }
            )
            
            rows = result.fetchall()
            
            results = []
            for row in rows:
                results.append({
                    "chunk_id": row.chunk_id,
                    "content": row.content,
                    "resource_id": row.resource_id,
                    "filename": row.filename,
                    "metadata": row.chunk_metadata,
                    "semantic_score": float(row.similarity)
                })
            
            logger.info(f"Semantic search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise
    
    async def keyword_search(
        self,
        query: str,
        workspace_id: UUID,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword search using PostgreSQL full-text search (BM25-like)
        
        Args:
            query: Search query
            workspace_id: Workspace to search in
            limit: Maximum number of results
            
        Returns:
            List of search results with BM25 scores
        """
        try:
            sql = text("""
                SELECT 
                    c.id as chunk_id,
                    c.content,
                    c.resource_id,
                    c.chunk_metadata,
                    r.filename,
                    ts_rank_cd(
                        to_tsvector('english', c.content),
                        plainto_tsquery('english', :query)
                    ) as bm25_score
                FROM chunks c
                JOIN resources r ON c.resource_id = r.id
                WHERE c.workspace_id = :workspace_id
                    AND to_tsvector('english', c.content) @@ plainto_tsquery('english', :query)
                ORDER BY bm25_score DESC
                LIMIT :limit
            """)
            
            result = await self.db.execute(
                sql,
                {
                    "query": query,
                    "workspace_id": str(workspace_id),
                    "limit": limit
                }
            )
            
            rows = result.fetchall()
            
            results = []
            for row in rows:
                results.append({
                    "chunk_id": row.chunk_id,
                    "content": row.content,
                    "resource_id": row.resource_id,
                    "filename": row.filename,
                    "metadata": row.chunk_metadata,
                    "bm25_score": float(row.bm25_score)
                })
            
            logger.info(f"Keyword search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            raise
    
    async def hybrid_search(
        self,
        query: str,
        workspace_id: UUID,
        limit: int = 20,
        semantic_weight: float = None,
        bm25_weight: float = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword search
        Uses reciprocal rank fusion to merge results
        
        Args:
            query: Search query
            workspace_id: Workspace to search in
            limit: Maximum number of results
            semantic_weight: Weight for semantic scores (default from config)
            bm25_weight: Weight for BM25 scores (default from config)
            
        Returns:
            List of merged and ranked search results
        """
        if semantic_weight is None:
            semantic_weight = settings.HYBRID_SEARCH_SEMANTIC_WEIGHT
        if bm25_weight is None:
            bm25_weight = settings.HYBRID_SEARCH_BM25_WEIGHT
        
        try:
            # Use the database function for efficient hybrid search
            query_embedding = embedding_service.embed_text(query)
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
            
            sql = text("""
                SELECT * FROM hybrid_search(
                    :query_embedding::vector,
                    :query_text,
                    :workspace_id,
                    :result_limit,
                    :semantic_weight,
                    :bm25_weight
                )
            """)
            
            result = await self.db.execute(
                sql,
                {
                    "query_embedding": embedding_str,
                    "query_text": query,
                    "workspace_id": str(workspace_id),
                    "result_limit": limit,
                    "semantic_weight": semantic_weight,
                    "bm25_weight": bm25_weight
                }
            )
            
            rows = result.fetchall()
            
            # Fetch full chunk details
            results = []
            for row in rows:
                chunk_sql = text("""
                    SELECT 
                        c.id, c.content, c.resource_id, c.chunk_metadata,
                        c.citation_count, r.filename
                    FROM chunks c
                    JOIN resources r ON c.resource_id = r.id
                    WHERE c.id = :chunk_id
                """)
                
                chunk_result = await self.db.execute(
                    chunk_sql,
                    {"chunk_id": str(row.chunk_id)}
                )
                chunk = chunk_result.fetchone()
                
                if chunk:
                    results.append({
                        "chunk_id": chunk.id,
                        "content": chunk.content,
                        "resource_id": chunk.resource_id,
                        "filename": chunk.filename,
                        "metadata": chunk.chunk_metadata,
                        "citation_count": chunk.citation_count,
                        "combined_score": float(row.combined_score),
                        "semantic_score": float(row.semantic_score),
                        "bm25_score": float(row.bm25_score)
                    })
            
            logger.info(f"Hybrid search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to manual merging if database function fails
            return await self._fallback_hybrid_search(
                query, workspace_id, limit, semantic_weight, bm25_weight
            )
    
    async def _fallback_hybrid_search(
        self,
        query: str,
        workspace_id: UUID,
        limit: int,
        semantic_weight: float,
        bm25_weight: float
    ) -> List[Dict[str, Any]]:
        """
        Fallback hybrid search using reciprocal rank fusion in Python
        """
        logger.info("Using fallback hybrid search implementation")
        
        # Perform both searches
        semantic_results = await self.semantic_search(query, workspace_id, limit * 2)
        keyword_results = await self.keyword_search(query, workspace_id, limit * 2)
        
        # Reciprocal rank fusion
        k = 60  # Constant for RRF
        scores = {}
        
        # Add semantic scores
        for rank, result in enumerate(semantic_results, 1):
            chunk_id = result["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0) + semantic_weight / (k + rank)
        
        # Add BM25 scores
        for rank, result in enumerate(keyword_results, 1):
            chunk_id = result["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0) + bm25_weight / (k + rank)
        
        # Merge results
        all_results = {r["chunk_id"]: r for r in semantic_results + keyword_results}
        
        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        merged_results = []
        for chunk_id in sorted_ids[:limit]:
            result = all_results[chunk_id]
            result["combined_score"] = scores[chunk_id]
            merged_results.append(result)
        
        return merged_results