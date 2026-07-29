import asyncio
from sqlalchemy import select, func, literal_column
from core.database import AsyncSessionLocal
from core.models import RagChunk, RagFile

async def main():
    query = '"top 5 documentaries"'
    async with AsyncSessionLocal() as db:
        or_query = " OR ".join(w for w in query.replace("?", "").replace(".", "").split())
        print(f"OR Query: {or_query}")
        ts_query = func.websearch_to_tsquery(literal_column("'english'"), or_query)
        text_rank = func.ts_rank_cd(RagChunk.search_vector, ts_query).label("text_rank")
        text_result = await db.execute(
            select(RagChunk.id, RagChunk.file_id, text_rank, ts_query)
            .join(RagFile, RagFile.id == RagChunk.file_id)
            .where(
                RagChunk.search_vector.op("@@")(ts_query)
            )
            .order_by(text_rank.desc())
            .limit(10)
        )
        print("--- TEXT SEARCH ---")
        for chunk_id, file_id, rank_val, ts_q in text_result.all():
            print(f"chunk={chunk_id} file={file_id} rank={rank_val} query={ts_q}")

if __name__ == "__main__":
    asyncio.run(main())
