import asyncio
from core.database import AsyncSessionLocal
from services.rag import retrieve_rag_chunks, format_rag_context

async def main():
    query = "What is the name of client? From issue PDF?"
    user_id = 2  # The user ID from the logs
    
    print(f"Query: {query}")
    chunks = await retrieve_rag_chunks(user_id, query)
    print(f"Retrieved {len(chunks)} chunks")
    
    for idx, chunk in enumerate(chunks, 1):
        print(f"[{idx}] id={chunk.id} file={chunk.filename} score={chunk.score} dist={chunk.vector_similarity} rank={chunk.text_rank}")

    print("\n--- CONTEXT ---")
    print(format_rag_context(chunks))

if __name__ == "__main__":
    asyncio.run(main())
