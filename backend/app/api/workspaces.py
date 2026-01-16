from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select
from typing import List
from uuid import UUID
import logging

from app.database import get_db
from app.models.database_models import Workspace
from app.models.schemas import WorkspaceCreate, WorkspaceResponse

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/", response_model=List[WorkspaceResponse])
async def list_workspaces(db: AsyncSession = Depends(get_db)):
    """List all available workspaces"""
    result = await db.execute(select(Workspace))
    return result.scalars().all()

@router.post("/", response_model=WorkspaceResponse)
async def create_workspace(
    workspace: WorkspaceCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new workspace"""
    # Check if workspace with same name exists
    result = await db.execute(select(Workspace).where(Workspace.name == workspace.name))
    existing = result.scalars().first()
    if existing:
        logger.info(f"Workspace with name {workspace.name} already exists")
        raise HTTPException(status_code=400, detail="Workspace with this name already exists")
    
    db_workspace = Workspace(
        name=workspace.name,
        workspace_type=workspace.workspace_type
    )
    db.add(db_workspace)
    await db.commit()
    await db.refresh(db_workspace)
    return db_workspace

@router.get("/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(
    workspace_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get workspace details"""
    result = await db.execute(select(Workspace).where(Workspace.id == workspace_id))
    workspace = result.scalars().first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace
