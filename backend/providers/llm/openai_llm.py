import os
import asyncio

from pipecat.services.openai.llm import OpenAILLMService

from core.prompt_config import load_system_prompt
from core.tool_config import tool_timeout_seconds
from core.llm_config import first_token_timeout_seconds, total_timeout_seconds
from .stream_timeout import LLMStreamDeadlineError, bounded_openai_stream


class LatencyBoundOpenAILLMService(OpenAILLMService):
    async def _stream_content(self, context):
        first_timeout = first_token_timeout_seconds()
        total_timeout = total_timeout_seconds()
        started = asyncio.get_running_loop().time()
        try:
            stream = await asyncio.wait_for(super()._stream_content(context), timeout=first_timeout)
        except TimeoutError as exc:
            raise LLMStreamDeadlineError("OpenAI stream creation deadline exceeded") from exc
        elapsed = asyncio.get_running_loop().time() - started
        return bounded_openai_stream(
            stream,
            max(0.001, first_timeout - elapsed),
            max(0.001, total_timeout - elapsed),
        )


def get_openai_llm():
    """Build the sole OpenAI service used by a voice pipeline."""
    return LatencyBoundOpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=LatencyBoundOpenAILLMService.Settings(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            system_instruction=load_system_prompt(),
        ),
        function_call_timeout_secs=tool_timeout_seconds(),
        enable_async_tool_cancellation=True,
        retry_on_timeout=False,
    )
