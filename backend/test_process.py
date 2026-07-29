import asyncio
from core.database import AsyncSessionLocal
from sqlalchemy import select
from core.models import Message
from services.memory import _process_saved_message_background

async def test():
    async with AsyncSessionLocal() as db:
        msg = (await db.execute(select(Message).where(Message.id == 211))).scalars().first()
        if not msg:
            print("Message not found")
            return
        print(f"Processing message {msg.id} for conversation {msg.conversation_id}")
        await _process_saved_message_background(msg.conversation_id, msg.id)
        print("Done processing")

if __name__ == "__main__":
    asyncio.run(test())
