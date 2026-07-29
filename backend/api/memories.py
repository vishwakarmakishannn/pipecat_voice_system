from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from api.auth import get_current_user
from core.database import get_db
from core.models import User, UserMemory


router = APIRouter(prefix="/api/memories", tags=["memories"])


class MemoryResponse(BaseModel):
    id: int
    fact_type: str
    key: str
    value: str
    confidence: float | None
    durability: str
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


@router.get("", response_model=List[MemoryResponse])
async def get_memories(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(UserMemory)
        .where(UserMemory.user_id == current_user.id, UserMemory.status == "active")
        .order_by(UserMemory.fact_type.asc(), UserMemory.key.asc(), UserMemory.updated_at.desc())
    )
    return result.scalars().all()


@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(UserMemory).where(
            UserMemory.id == memory_id,
            UserMemory.user_id == current_user.id,
        )
    )
    memory = result.scalars().first()
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    await db.delete(memory)
    await db.commit()
    return {"message": "Deleted successfully"}


@router.delete("")
async def delete_all_memories(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(UserMemory).where(UserMemory.user_id == current_user.id))
    memories = result.scalars().all()
    for memory in memories:
        await db.delete(memory)
    await db.commit()
    return {"message": "Deleted successfully", "count": len(memories)}
