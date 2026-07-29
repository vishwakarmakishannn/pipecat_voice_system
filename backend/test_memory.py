import asyncio
from services.memory import classify_memory_events

async def test():
    user_text = "You can call Mirage, by the way."
    assistant_text = ""
    events = await classify_memory_events(user_text, assistant_text)
    print("Extracted events:", events)

if __name__ == "__main__":
    asyncio.run(test())
