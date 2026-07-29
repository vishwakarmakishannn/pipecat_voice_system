import asyncio
import logging
from services.rag import process_rag_file
from core.database import AsyncSessionLocal
from core.models import RagFile
from sqlalchemy import select

logging.basicConfig(level=logging.INFO)

async def test():
    print("Fetching rag file...")
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(RagFile).where(RagFile.id == 14))
        rag_file = result.scalars().first()
        if not rag_file:
            print("File not found")
            return
        
        print(f"File status: {rag_file.status}, path: {rag_file.storage_path}")

    print("Running process_rag_file...")
    await process_rag_file(14)
    print("Done")

if __name__ == "__main__":
    asyncio.run(test())
