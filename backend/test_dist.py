import asyncio
import os
import sys

from sqlalchemy import select, text
from core.database import AsyncSessionLocal
from core.models import RagChunk, RagFile
from services.memory import embed_text

async def main():
    query = "What is the name of client? From issue PDF?"
    print(f"Query: {query}")
    embedding = await embed_text(query)
    if not embedding:
        print("Failed to embed query")
        return

    print("Embedding generated")
    
    async with AsyncSessionLocal() as db:
        distance_expr = RagChunk.embedding.cosine_distance(embedding).label("distance")
        result = await db.execute(
            select(RagChunk.id, RagChunk.file_id, RagFile.title, RagFile.filename, distance_expr)
            .join(RagFile, RagFile.id == RagChunk.file_id)
            .where(RagChunk.embedding.is_not(None))
            .order_by(distance_expr.asc())
            .limit(10)
        )
        
        print("--- VECTOR DISTANCES ---")
        for idx, (chunk_id, file_id, title, filename, distance) in enumerate(result.all(), 1):
            print(f"[{idx}] chunk={chunk_id} file={file_id} file='{filename}' dist={distance:.4f} sim={1.0-distance:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
