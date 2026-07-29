from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

from core.database import get_db
from core.models import User, Conversation, Message
from api.auth import get_current_user
from services.memory import process_saved_message

router = APIRouter(prefix="/api/conversations", tags=["conversations"])

class MessageCreate(BaseModel):
    role: Literal["You", "Aura", "ToolCall", "RagCall"]
    content: str = Field(..., min_length=1, max_length=100000)

class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class ConversationCreate(BaseModel):
    title: Optional[str] = Field("New conversation", max_length=255)

class ConversationResponse(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

@router.get("", response_model=List[ConversationResponse])
async def get_conversations(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc(), Conversation.created_at.desc())
    )
    return result.scalars().all()

@router.post("", response_model=ConversationResponse)
async def create_conversation(conv_data: ConversationCreate, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    new_conv = Conversation(user_id=current_user.id, title=conv_data.title)
    db.add(new_conv)
    await db.commit()
    await db.refresh(new_conv)
    return new_conv

@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: int, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == current_user.id))
    conv = result.scalars().first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    await db.delete(conv)
    await db.commit()
    return {"message": "Deleted successfully"}

@router.get("/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_messages(conversation_id: int, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == current_user.id))
    if not result.scalars().first():
        raise HTTPException(status_code=404, detail="Conversation not found")
        
    msg_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    return msg_result.scalars().all()

@router.post("/{conversation_id}/messages", response_model=MessageResponse)
async def add_message(conversation_id: int, message_data: MessageCreate, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == current_user.id))
    conv = result.scalars().first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
        
    new_msg = Message(conversation_id=conversation_id, role=message_data.role, content=message_data.content)
    db.add(new_msg)
    await db.flush()
    await process_saved_message(db, conv, new_msg)
    await db.commit()
    await db.refresh(new_msg)
    return new_msg
