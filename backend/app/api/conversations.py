"""
API endpoints for conversations and chat messages
"""
import json 
import logging
from typing import Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage

from app.database import get_db
from app.config import settings
from app.agents.rag_agent import RAGAgent
from app.services.search import SearchService
from app.services.ranking import RankingService
from app.services.citation import CitationService
from app.services.hardware import HardwareDetector
from app.services.llm_service import get_llm_service

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


async def generate_conversation_title(conversation_id: str, user_content: str, assistant_content: str):
    """Background task to generate a title for the conversation"""
    try:
        # We need a new database session for the background task
        from app.database import AsyncSessionLocal
        
        llm = get_llm_service()
        prompt = f"""Summarize the following interaction into a short, concise title (max 5 words). Do not use quotes.
        User: {user_content[:200]}
        Assistant: {assistant_content[:200]}
        Title:"""
        
        messages = [HumanMessage(content=prompt)]
        title = await llm.generate(messages, temperature=0.7)
        title = title.strip().strip('"')
        
        logger.info(f"Generated title for {conversation_id}: {title}")
        
        async with AsyncSessionLocal() as session:
            try:
                await session.execute(
                    text("UPDATE conversations SET title = :title WHERE id = :id"),
                    {"title": title, "id": conversation_id}
                )
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database update failed for title generation: {e}")
            
    except Exception as e:
        logger.error(f"Failed to auto-generate title: {e}")


@router.post("")
async def create_conversation(
    request: CreateConversationRequest,
    db: AsyncSession = Depends(get_db)
):
    """Create a new conversation"""
    try:
        conversation_id = uuid4()
        
        # Use optimal model from hardware detector instead of settings default
        model_name = HardwareDetector.get_optimal_model()
        
        await db.execute(
            text("""
                INSERT INTO conversations (id, workspace_id, title, model_name, system_prompt)
                VALUES (:id, :workspace_id, :title, :model_name, :system_prompt)
            """),
            {
                "id": str(conversation_id),
                "workspace_id": request.workspace_id,
                "title": request.title or "New Conversation",
                "model_name": model_name,
                "system_prompt": request.system_prompt
            }
        )
        # Commit is handled by the dependency or explicit commit here
        await db.commit()
        
        return {
            "id": str(conversation_id),
            "workspace_id": request.workspace_id,
            "title": request.title or "New Conversation",
            "model_name": model_name,
            "created_at": "now"
        }
        
    except Exception as e:
        logger.error(f"Create conversation failed: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def list_conversations(
    workspace_id: Optional[UUID] = None,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """List conversations, optionally filtered by workspace"""
    try:
        query = """
            SELECT id, workspace_id, title, model_name, created_at
            FROM conversations
        """
        params = {"limit": limit, "offset": offset}
        
        if workspace_id:
            query += " WHERE workspace_id = :workspace_id"
            params["workspace_id"] = str(workspace_id)
            
        query += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
        
        result = await db.execute(text(query), params)
        
        conversations = []
        for row in result.fetchall():
            conversations.append({
                "id": str(row.id),
                "workspace_id": str(row.workspace_id),
                "title": row.title,
                "model_name": row.model_name,
                "created_at": row.created_at.isoformat()
            })
        
        return {"conversations": conversations}
        
    except Exception as e:
        logger.error(f"List conversations failed: {e}")
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

@router.get("/{conversation_id}/messages")
async def get_messages(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get all messages in a conversation"""
    try:
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
        
        return {"messages": messages}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get messages failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{conversation_id}/messages")
async def send_message(
    conversation_id: UUID,
    request: SendMessageRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    async def generator():
        try:
            # Verify conversation exists
            conv_result = await db.execute(
                text("SELECT id, title FROM conversations WHERE id = :id"),
                {"id": str(conversation_id)}
            )
            conv = conv_result.fetchone()
            if not conv:
                yield json.dumps({"type": "error", "data": "Conversation not found"}) + "\n"
                return

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
            await db.commit()

            # Initialize services + agent
            search_service = SearchService(db)
            ranking_service = RankingService(db)
            citation_service = CitationService(db)
            rag_agent = RAGAgent(search_service, ranking_service, citation_service)

            # Stream tokens from LLM
            assistant_text = ""
            final_payload = None

            async for event in rag_agent.run_stream(
                query=request.content,
                workspace_id=UUID(request.workspace_id),
                conversation_id=conversation_id
            ):
                if event["type"] == "content":
                    assistant_text += event["data"]

                if event["type"] == "end":
                    final_payload = event

                yield json.dumps(event, default=str) + "\n"

            # Save assistant message after streaming ends
            if final_payload:
                assistant_message_id = uuid4()

                await db.execute(
                    text("""
                        INSERT INTO messages (id, conversation_id, role, content, citations, source_chunks, model_metadata)
                        VALUES (:id, :conversation_id, 'assistant', :content,
                                CAST(:citations AS JSONB), :source_chunks, CAST(:metadata AS JSONB))
                    """),
                    {
                        "id": str(assistant_message_id),
                        "conversation_id": str(conversation_id),
                        "content": assistant_text,
                        "citations": json.dumps(final_payload.get("citations", []), default=str),
                        "source_chunks": [str(c["chunk_id"]) for c in final_payload.get("citations", [])],
                        "metadata": json.dumps(final_payload.get("metadata", {}), default=str),
                    }
                )
                await db.commit()

                # Send message_id in final event
                yield json.dumps({
                    "type": "message_id",
                    "data": str(assistant_message_id)
                }) + "\n"

                # Title generation (first exchange)
                if conv.title == "New Conversation":
                    background_tasks.add_task(
                        generate_conversation_title,
                        str(conversation_id),
                        request.content,
                        assistant_text
                    )

        except Exception as e:
            logger.exception("Streaming send_message failed")
            yield json.dumps({"type": "error", "data": str(e)}) + "\n"

    return StreamingResponse(generator(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

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
        
        await db.commit()
        
        return {"status": "deleted", "conversation_id": str(conversation_id)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete conversation failed: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))