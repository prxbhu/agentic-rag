"""
Celery tasks for asynchronous embedding generation
"""
import logging
from typing import List, Dict
from uuid import UUID
from datetime import datetime

from app.tasks.celery_app import celery_app
from app.services.embedding import embedding_service
from app.database import get_db_session
from sqlalchemy import text

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60
)
def generate_embeddings_task(self, resource_id: str, chunks: List[Dict]):
    """
    Celery task to generate embeddings for document chunks
    
    Args:
        resource_id: UUID of the resource
        chunks: List of chunk dictionaries with 'id' and 'content'
    """
    task_id = self.request.id
    logger.info(f"Task {task_id}: Starting embedding generation for {len(chunks)} chunks")
    
    try:
        # Update task status to processing
        _update_task_status(
            resource_id=resource_id,
            task_id=task_id,
            status="processing",
            total_chunks=len(chunks)
        )
        
        # Extract texts for batch processing
        texts = [chunk["content"] for chunk in chunks]
        
        # Generate embeddings in batches
        embeddings = embedding_service.embed_batch(texts)
        
        # Store embeddings in database
        _store_embeddings(chunks, embeddings)
        
        # Update task status to completed
        _update_task_status(
            resource_id=resource_id,
            task_id=task_id,
            status="completed",
            chunks_processed=len(chunks)
        )
        
        # Update resource status
        _update_resource_status(resource_id, "completed")
        
        logger.info(f"Task {task_id}: Successfully generated {len(chunks)} embeddings")
        
        return {
            "resource_id": resource_id,
            "chunks_processed": len(chunks),
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Task {task_id}: Embedding generation failed: {e}")
        
        # Update task status to failed
        _update_task_status(
            resource_id=resource_id,
            task_id=task_id,
            status="failed",
            error_message=str(e)
        )
        
        # Update resource status
        _update_resource_status(resource_id, "failed")
        
        # Retry if possible
        raise self.retry(exc=e)


def _update_task_status(
    resource_id: str,
    task_id: str,
    status: str,
    total_chunks: int = None,
    chunks_processed: int = None,
    error_message: str = None
):
    """Update embedding task status in database"""
    with get_db_session() as db:
        updates = ["status = :status"]
        params = {
            "resource_id": resource_id,
            "task_id": task_id,
            "status": status
        }
        
        if status == "processing":
            updates.append("started_at = :started_at")
            params["started_at"] = datetime.utcnow()
        
        if status == "completed":
            updates.append("completed_at = :completed_at")
            params["completed_at"] = datetime.utcnow()
        
        if total_chunks is not None:
            updates.append("total_chunks = :total_chunks")
            params["total_chunks"] = total_chunks
        
        if chunks_processed is not None:
            updates.append("chunks_processed = :chunks_processed")
            params["chunks_processed"] = chunks_processed
        
        if error_message is not None:
            updates.append("error_message = :error_message")
            params["error_message"] = error_message
        
        sql = text(f"""
            UPDATE embedding_tasks
            SET {', '.join(updates)}
            WHERE resource_id = :resource_id AND task_id = :task_id
        """)
        
        db.execute(sql, params)
        db.commit()


def _store_embeddings(chunks: List[Dict], embeddings: List[List[float]]):
    """Store embeddings in the database"""
    with get_db_session() as db:
        for chunk, embedding in zip(chunks, embeddings):
            # Convert embedding to PostgreSQL vector format
            embedding_str = f"[{','.join(map(str, embedding))}]"
            
            sql = text("""
                UPDATE chunks
                SET embedding = :embedding::vector
                WHERE id = :chunk_id
            """)
            
            db.execute(sql, {
                "chunk_id": chunk["id"],
                "embedding": embedding_str
            })
        
        db.commit()


def _update_resource_status(resource_id: str, status: str):
    """Update resource processing status"""
    with get_db_session() as db:
        sql = text("""
            UPDATE resources
            SET status = :status, updated_at = :updated_at
            WHERE id = :resource_id
        """)
        
        db.execute(sql, {
            "resource_id": resource_id,
            "status": status,
            "updated_at": datetime.utcnow()
        })
        db.commit()


@celery_app.task
def cleanup_old_tasks():
    """
    Periodic task to clean up old completed/failed embedding tasks
    Runs daily to keep the database clean
    """
    logger.info("Running cleanup of old embedding tasks")
    
    try:
        with get_db_session() as db:
            # Delete tasks older than 30 days
            sql = text("""
                DELETE FROM embedding_tasks
                WHERE (status = 'completed' OR status = 'failed')
                    AND created_at < NOW() - INTERVAL '30 days'
            """)
            
            result = db.execute(sql)
            db.commit()
            
            deleted_count = result.rowcount
            logger.info(f"Cleaned up {deleted_count} old tasks")
            
            return {"deleted_count": deleted_count}
            
    except Exception as e:
        logger.error(f"Cleanup task failed: {e}")
        raise


@celery_app.task
def reindex_chunks():
    """
    Periodic task to rebuild HNSW index for optimal performance
    Should be run weekly or when index performance degrades
    """
    logger.info("Running HNSW index rebuild")
    
    try:
        with get_db_session() as db:
            # Rebuild the HNSW index
            sql = text("""
                REINDEX INDEX CONCURRENTLY idx_chunks_embedding
            """)
            
            db.execute(sql)
            db.commit()
            
            logger.info("HNSW index rebuild completed")
            
            return {"status": "completed"}
            
    except Exception as e:
        logger.error(f"Index rebuild failed: {e}")
        raise