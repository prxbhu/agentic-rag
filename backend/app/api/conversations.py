"""
API endpoints for conversations and chat messages
"""
import json 
import logging
from typing import Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.config import settings
from app.agents.rag_agent import RAGAgent
from app.services.search import SearchService
from app.services.ranking import RankingService
from app.services.citation import CitationService

logger = logging.getLogger(__name__)

router = APIRouter()


class CreateConversationRequest(BaseModel):
    workspace_id: str
    title: Optional[str] = None
    system_prompt: Optional[str] = None


class SendMessageRequest(BaseModel):
    content: str
    workspace_id: str


class Message(BaseModel):
    id: str
    role: str
    content: str
    citations: list
    created_at: str


@router.post("")
async def create_conversation(
    request: CreateConversationRequest,
    db: AsyncSession = Depends(get_db)
):
    """Create a new conversation"""
    try:
        conversation_id = uuid4()
        
        await db.execute(
            text("""
                INSERT INTO conversations (id, workspace_id, title, model_name, system_prompt)
                VALUES (:id, :workspace_id, :title, :model_name, :system_prompt)
            """),
            {
                "id": str(conversation_id),
                "workspace_id": request.workspace_id,
                "title": request.title or "New Conversation",
                "model_name": settings.CHAT_MODEL,
                "system_prompt": request.system_prompt
            }
        )
        #await db.commit()
        
        return {
            "id": str(conversation_id),
            "workspace_id": request.workspace_id,
            "title": request.title or "New Conversation",
            "created_at": "now"
        }
        
    except Exception as e:
        logger.error(f"Create conversation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get conversation details and message history"""
    try:
        # Get conversation
        conv_result = await db.execute(
            text("""
                SELECT id, workspace_id, title, model_name, created_at
                FROM conversations
                WHERE id = :conversation_id
            """),
            {"conversation_id": str(conversation_id)}
        )
        conv_row = conv_result.fetchone()
        
        if not conv_row:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get messages
        messages_result = await db.execute(
            text("""
                SELECT id, role, content, citations, created_at
                FROM messages
                WHERE conversation_id = :conversation_id
                ORDER BY created_at ASC
            """),
            {"conversation_id": str(conversation_id)}
        )
        
        messages = []
        for row in messages_result.fetchall():
            messages.append({
                "id": str(row.id),
                "role": row.role,
                "content": row.content,
                "citations": row.citations or [],
                "created_at": row.created_at.isoformat()
            })
        
        return {
            "id": str(conv_row.id),
            "workspace_id": str(conv_row.workspace_id),
            "title": conv_row.title,
            "model_name": conv_row.model_name,
            "created_at": conv_row.created_at.isoformat(),
            "messages": messages
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get conversation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{conversation_id}/messages")
async def send_message(
    conversation_id: UUID,
    request: SendMessageRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Send a message and get RAG response
    
    This endpoint triggers the full RAG pipeline:
    1. Query expansion
    2. Hybrid search
    3. Multi-factor ranking
    4. Context assembly
    5. LLM generation
    6. Citation verification
    """
    try:
        # Verify conversation exists
        conv_result = await db.execute(
            text("SELECT id FROM conversations WHERE id = :id"),
            {"id": str(conversation_id)}
        )
        if not conv_result.fetchone():
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Store user message
        user_message_id = uuid4()
        await db.execute(
            text("""
                INSERT INTO messages (id, conversation_id, role, content)
                VALUES (:id, :conversation_id, 'user', :content)
            """),
            {
                "id": str(user_message_id),
                "conversation_id": str(conversation_id),
                "content": request.content
            }
        )
        # await db.commit()
        
        # Initialize services
        search_service = SearchService(db)
        ranking_service = RankingService(db)
        citation_service = CitationService(db)
        
        # Initialize RAG agent
        rag_agent = RAGAgent(search_service, ranking_service, citation_service)
        
        # Run RAG pipeline
        logger.info(f"Running RAG pipeline for conversation {conversation_id}")
        result = await rag_agent.run(
            query=request.content,
            workspace_id=UUID(request.workspace_id),
            conversation_id=conversation_id
        )
        
        # Store assistant message
        assistant_message_id = uuid4()
        await db.execute(
            text("""
                INSERT INTO messages (id, conversation_id, role, content, 
                                    citations, source_chunks, model_metadata)
                VALUES (:id, :conversation_id, 'assistant', :content, 
                        CAST(:citations AS JSONB), :source_chunks, CAST(:metadata AS JSONB))
            """),
            {
                "id": str(assistant_message_id),
                "conversation_id": str(conversation_id),
                "content": result["response"],
                "citations": json.dumps(result["citations"], default=str),
                "source_chunks": [str(c["chunk_id"]) for c in result["citations"]],
                "metadata": json.dumps(result["metadata"], default=str)
            }
        )
        # await db.commit()
        
        # Update citation counts
        for citation in result["citations"]:
            await ranking_service.update_citation_count(citation["chunk_id"])
        
        return {
            "message_id": str(assistant_message_id),
            "role": "assistant",
            "content": result["response"],
            "citations": result["citations"],
            "metadata": result["metadata"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Send message failed: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}/export")
async def export_conversation(
    conversation_id: UUID,
    format: str = "json",
    db: AsyncSession = Depends(get_db)
):
    """
    Export conversation as JSON or Markdown
    """
    try:
        # Get conversation with messages
        conv_data = await get_conversation(conversation_id, db)
        
        if format == "markdown":
            # Convert to markdown
            md_lines = [
                f"# {conv_data['title']}",
                f"\n*Created: {conv_data['created_at']}*\n",
                "---\n"
            ]
            
            for msg in conv_data["messages"]:
                role_prefix = "**You:**" if msg["role"] == "user" else "**Assistant:**"
                md_lines.append(f"\n{role_prefix}\n\n{msg['content']}\n")
                
                if msg["citations"]:
                    md_lines.append("\n*Sources:*\n")
                    for citation in msg["citations"]:
                        md_lines.append(f"- {citation.get('content_preview', 'N/A')}\n")
            
            markdown = "\n".join(md_lines)
            
            return StreamingResponse(
                iter([markdown]),
                media_type="text/markdown",
                headers={
                    "Content-Disposition": f"attachment; filename=conversation_{conversation_id}.md"
                }
            )
        else:
            # Return as JSON
            return conv_data
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export conversation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Delete a conversation and all its messages"""
    try:
        result = await db.execute(
            text("DELETE FROM conversations WHERE id = :id RETURNING id"),
            {"id": str(conversation_id)}
        )
        
        if not result.fetchone():
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # await db.commit()
        
        return {"status": "deleted", "conversation_id": str(conversation_id)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete conversation failed: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))