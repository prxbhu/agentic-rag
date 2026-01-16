"""
API endpoints for resource management (document upload and processing)
"""
import json
import logging
import hashlib
from typing import Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.config import settings
from app.services.ingestion import IngestionService
from app.tasks.embedding_tasks import generate_embeddings_task

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    workspace_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload and process a document
    
    Returns resource_id and task_id for tracking embedding generation
    """
    
    if workspace_id == "":
        workspace_id = None

    if workspace_id is not None:
        try:
            workspace_id = UUID(workspace_id)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail="workspace_id must be a valid UUID"
            )
    
    try:
        # Validate file type
        file_ext = file.filename.split(".")[-1].lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Supported: {settings.ALLOWED_EXTENSIONS}"
            )
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # Validate file size
        if file_size > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE_MB}MB"
            )
        
        # Calculate content hash for deduplication
        content_hash = hashlib.sha256(content).hexdigest()
        
        # Check for duplicates
        duplicate_check = await db.execute(
            text("""
                SELECT id, is_duplicate_of 
                FROM resources 
                WHERE content_hash = :hash
            """),
            {"hash": content_hash}
        )
        existing = duplicate_check.fetchone()
        
        if existing:
            logger.info(f"Duplicate file detected: {content_hash}")
            return JSONResponse(
                status_code=200,
                content={
                    "status": "duplicate",
                    "message": "This document has already been uploaded",
                    "resource_id": str(existing.id),
                    "original_resource_id": str(existing.is_duplicate_of or existing.id)
                }
            )
        
        # Get or create default workspace
        if not workspace_id:
            workspace_result = await db.execute(
                text("SELECT id FROM workspaces WHERE workspace_type = 'personal' LIMIT 1")
            )
            workspace_row = workspace_result.fetchone()
            workspace_id = str(workspace_row.id) if workspace_row else str(uuid4())
        
        # Create resource record
        resource_id = uuid4()
        await db.execute(
            text("""
                INSERT INTO resources (id, workspace_id, filename, file_type, 
                                     content_hash, file_size_bytes, status)
                VALUES (:id, :workspace_id, :filename, :file_type, 
                        :content_hash, :file_size, 'processing')
            """),
            {
                "id": str(resource_id),
                "workspace_id": str(workspace_id),
                "filename": file.filename,
                "file_type": file_ext,
                "content_hash": content_hash,
                "file_size": file_size
            }
        )
        await db.commit()
        
        # Process document into chunks
        ingestion_service = IngestionService()
        chunks = await ingestion_service.process_document(
            content=content,
            filename=file.filename,
            file_type=file_ext,
            resource_id=resource_id,
            workspace_id=UUID(str(workspace_id))
        )
        
        logger.info(f"Document processed into {len(chunks)} chunks")
        
        # Store chunks in database (without embeddings initially)
        for chunk in chunks:
            await db.execute(
            text("""
                INSERT INTO chunks (
                    id, resource_id, workspace_id, content,
                    chunk_index, token_count, chunk_metadata
                )
                VALUES (
                    :id, :resource_id, :workspace_id, :content,
                    :chunk_index, :token_count,
                    CAST(:metadata AS JSONB)
                )
            """),
            {
                "id": str(chunk["id"]),
                "resource_id": str(resource_id),
                "workspace_id": str(workspace_id),
                "content": chunk["content"],
                "chunk_index": chunk["chunk_index"],
                "token_count": chunk["token_count"],
                "metadata": json.dumps(chunk["metadata"]) 
            }
        )

        await db.commit()
        
        
        task_id = str(uuid4())
        # Create task tracking record
        await db.execute(
            text("""
                INSERT INTO embedding_tasks (id, resource_id, task_id, status, 
                                           total_chunks, chunks_processed)
                VALUES (:id, :resource_id, :task_id, 'pending', :total_chunks, 0)
            """),
            {
                "id": str(uuid4()),
                "resource_id": str(resource_id),
                "task_id": task_id,
                "total_chunks": len(chunks)
            }
        )
        await db.commit()
        
         # Start async embedding task
        task = generate_embeddings_task.apply_async(
                args=[str(resource_id), [{"id": str(c["id"]), "content": c["content"]} for c in chunks]],
                task_id=task_id
)
        
        return {
            "status": "processing",
            "message": "Document uploaded and processing started",
            "resource_id": str(resource_id),
            "task_id": task.id,
            "chunks_count": len(chunks),
            "poll_url": f"/api/resources/{resource_id}/embedding-status"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{resource_id}/embedding-status")
async def get_embedding_status(
    resource_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the status of embedding generation for a resource
    
    Poll this endpoint to check when embeddings are complete
    """
    try:
        result = await db.execute(
            text("""
                SELECT 
                    et.status, 
                    et.chunks_processed, 
                    et.total_chunks,
                    et.error_message,
                    et.started_at,
                    et.completed_at,
                    r.filename
                FROM embedding_tasks et
                JOIN resources r ON et.resource_id = r.id
                WHERE et.resource_id = :resource_id
                ORDER BY et.created_at DESC
                LIMIT 1
            """),
            {"resource_id": str(resource_id)}
        )
        
        row = result.fetchone()
        
        if not row:
            raise HTTPException(
                status_code=404,
                detail="Resource not found or no embedding task exists"
            )
        
        return {
            "resource_id": str(resource_id),
            "filename": row.filename,
            "status": row.status,
            "progress": {
                "chunks_processed": row.chunks_processed,
                "total_chunks": row.total_chunks,
                "percentage": (row.chunks_processed / row.total_chunks * 100) 
                             if row.total_chunks > 0 else 0
            },
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
            "error_message": row.error_message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{resource_id}")
async def get_resource(
    resource_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get resource details"""
    try:
        result = await db.execute(
            text("""
                SELECT 
                    r.id, r.filename, r.file_type, r.file_size_bytes,
                    r.status, r.created_at, r.metadata,
                    COUNT(c.id) as chunk_count
                FROM resources r
                LEFT JOIN chunks c ON r.id = c.resource_id
                WHERE r.id = :resource_id
                GROUP BY r.id
            """),
            {"resource_id": str(resource_id)}
        )
        
        row = result.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Resource not found")
        
        return {
            "id": str(row.id),
            "filename": row.filename,
            "file_type": row.file_type,
            "file_size_bytes": row.file_size_bytes,
            "status": row.status,
            "chunk_count": row.chunk_count,
            "created_at": row.created_at.isoformat(),
            "metadata": row.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get resource failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{resource_id}")
async def delete_resource(
    resource_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Delete a resource and all associated chunks"""
    try:
        # Check if resource exists
        result = await db.execute(
            text("SELECT id FROM resources WHERE id = :resource_id"),
            {"resource_id": str(resource_id)}
        )
        
        if not result.fetchone():
            raise HTTPException(status_code=404, detail="Resource not found")
        
        # Delete resource (cascades to chunks and tasks)
        await db.execute(
            text("DELETE FROM resources WHERE id = :resource_id"),
            {"resource_id": str(resource_id)}
        )
        await db.commit()
        
        return {"status": "deleted", "resource_id": str(resource_id)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete resource failed: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))