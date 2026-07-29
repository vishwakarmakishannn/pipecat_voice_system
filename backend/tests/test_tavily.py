import asyncio
from dotenv import load_dotenv
load_dotenv(override=True)

from tools.tavily import tavily_search
from pipecat.services.llm_service import FunctionCallParams

async def main():
    params = FunctionCallParams(
        function_name="tavily_search",
        tool_call_id="123",
        args={"query": "Taylor Swift latest album"}
    )
    result = await tavily_search(params, query="Taylor Swift latest album", max_results=1)
    print("Result:", result)

if __name__ == "__main__":
    asyncio.run(main())
