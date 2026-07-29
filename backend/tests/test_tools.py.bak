import asyncio
from pipecat.processors.aggregators.llm_context import LLMContext

async def tavily_search(query: str, max_results: int = 5):
    """Search the web using Tavily.
    
    Args:
        query: The search query.
        max_results: Max number of results.
    """
    pass

def test():
    context = LLMContext(tools=[tavily_search], messages=[])
    print(context.tools)
    
if __name__ == "__main__":
    test()
