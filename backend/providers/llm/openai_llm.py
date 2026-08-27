import os
import asyncio
import time
import uuid

from pipecat.services.openai.llm import OpenAILLMService

from core.prompt_config import load_system_prompt
from core.tool_config import tool_timeout_seconds
from core.llm_config import (
    first_token_timeout_seconds,
    timeout_recovery_text,
    total_timeout_seconds,
)
from .stream_timeout import (
    bounded_openai_stream,
    openai_recovery_stream,
    recovering_openai_stream,
    report_llm_deadline,
)


class LatencyBoundOpenAILLMService(OpenAILLMService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diagnostic_callback = None

    async def _stream_content(self, context):
        first_timeout = first_token_timeout_seconds()
        total_timeout = total_timeout_seconds()
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        model = self._settings.model
        recovery_text = timeout_recovery_text()
        try:
            stream = await asyncio.wait_for(super()._stream_content(context), timeout=first_timeout)
        except TimeoutError:
            report_llm_deadline(
                self.diagnostic_callback,
                code="llm.stream_creation_timeout",
                request_id=request_id,
                duration_ms=round((time.monotonic() - started) * 1000, 1),
                recovery_text=recovery_text,
            )
            return openai_recovery_stream(recovery_text, model)
        elapsed = time.monotonic() - started
        bounded = bounded_openai_stream(
            stream,
            max(0.001, first_timeout - elapsed),
            max(0.001, total_timeout - elapsed),
        )
        return recovering_openai_stream(
            bounded,
            recovery_text=recovery_text,
            model=model,
            request_id=request_id,
            started_at=started,
            diagnostic_callback=self.diagnostic_callback,
        )


def get_openai_llm(*, system_instruction: str | None = None):
    """Build the sole OpenAI service used by a voice pipeline."""
    return LatencyBoundOpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=LatencyBoundOpenAILLMService.Settings(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            system_instruction=system_instruction or load_system_prompt(),
        ),
        function_call_timeout_secs=tool_timeout_seconds(),
        enable_async_tool_cancellation=True,
        retry_on_timeout=False,
    )
