import asyncio
import os
import time
import uuid

from loguru import logger
from google.genai.types import Candidate, Content, GenerateContentResponse, Part
from pipecat.services.google.llm import GoogleLLMService
from core.llm_config import (
    first_token_timeout_seconds,
    timeout_recovery_text,
)
from core.prompt_config import load_system_prompt
from core.tool_config import tool_timeout_seconds


class FirstTokenTimeoutError(TimeoutError):
    """Raised when a live LLM stream produces no meaningful first chunk."""


class LatencyBoundGoogleLLMService(GoogleLLMService):
    def __init__(
        self,
        *,
        first_token_timeout_secs: float,
        timeout_message: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._first_token_timeout_secs = first_token_timeout_secs
        self._timeout_message = timeout_message

    @staticmethod
    def _chunk_has_output(chunk) -> bool:
        for candidate in getattr(chunk, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                if (
                    getattr(part, "text", None)
                    or getattr(part, "function_call", None)
                    or getattr(part, "inline_data", None)
                ):
                    return True
        return False

    @classmethod
    async def _first_output_stream(cls, stream, timeout_secs: float):
        iterator = stream.__aiter__()
        buffered = []
        deadline = asyncio.get_running_loop().time() + timeout_secs
        try:
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError
                try:
                    chunk = await asyncio.wait_for(anext(iterator), timeout=remaining)
                except StopAsyncIteration:
                    for item in buffered:
                        yield item
                    return
                buffered.append(chunk)
                if cls._chunk_has_output(chunk):
                    break
        except asyncio.CancelledError:
            if hasattr(iterator, "aclose"):
                await iterator.aclose()
            raise
        except TimeoutError as exc:
            if hasattr(iterator, "aclose"):
                await iterator.aclose()
            raise FirstTokenTimeoutError(
                f"Google produced no first output within {timeout_secs:.2f}s"
            ) from exc

        for item in buffered:
            yield item
        async for chunk in iterator:
            yield chunk

    @staticmethod
    def _text_chunk(text: str) -> GenerateContentResponse:
        return GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(role="model", parts=[Part.from_text(text=text)]),
                )
            ]
        )

    @classmethod
    async def _recovering_stream(
        cls,
        stream,
        timeout_secs: float,
        timeout_message: str,
        *,
        request_id: str = "unknown",
        provider_model: str = "google",
        started_at: float | None = None,
    ):
        started = time.monotonic() if started_at is None else started_at
        first_chunk_seen = False
        try:
            async for chunk in cls._first_output_stream(stream, timeout_secs):
                if not first_chunk_seen:
                    first_chunk_seen = True
                    logger.info(
                        "voice_llm request_id={} provider=google model={} status=first_output "
                        "latency_ms={} provider_response_id={}",
                        request_id,
                        provider_model,
                        round((time.monotonic() - started) * 1000, 1),
                        getattr(chunk, "response_id", None),
                    )
                yield chunk
        except FirstTokenTimeoutError:
            logger.warning(
                "voice_llm request_id={} provider=google model={} status=first_token_timeout "
                "latency_ms={} budget_ms={} action=spoken_recovery",
                request_id,
                provider_model,
                round((time.monotonic() - started) * 1000, 1),
                round(timeout_secs * 1000),
            )
            yield cls._text_chunk(timeout_message)

    @classmethod
    async def _recovery_stream(
        cls,
        timeout_message: str,
    ):
        """Return a valid stream when request creation itself times out."""
        yield cls._text_chunk(timeout_message)

    async def _stream_content(self, context):
        request_id = uuid.uuid4().hex
        provider_model = self._settings.model
        started = time.monotonic()
        logger.info(
            "voice_llm request_id={} provider=google model={} status=started "
            "deadline_ms={}",
            request_id,
            provider_model,
            round(self._first_token_timeout_secs * 1000),
        )
        try:
            stream = await asyncio.wait_for(
                super()._stream_content(context),
                timeout=self._first_token_timeout_secs,
            )
        except TimeoutError:
            logger.warning(
                "voice_llm request_id={} provider=google model={} status=stream_creation_timeout "
                "latency_ms={} budget_ms={} action=spoken_recovery",
                request_id,
                provider_model,
                round((time.monotonic() - started) * 1000, 1),
                round(self._first_token_timeout_secs * 1000),
            )
            return self._recovery_stream(self._timeout_message)

        elapsed = time.monotonic() - started
        remaining = max(0.001, self._first_token_timeout_secs - elapsed)
        return self._recovering_stream(
            stream,
            remaining,
            self._timeout_message,
            request_id=request_id,
            provider_model=provider_model,
            started_at=started,
        )


def get_google_llm():
    timeout_secs = first_token_timeout_seconds()
    return LatencyBoundGoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        first_token_timeout_secs=timeout_secs,
        timeout_message=timeout_recovery_text(),
        settings=GoogleLLMService.Settings(
            model=os.getenv("GOOGLE_MODEL", "gemini-3.1-flash-lite"),
            system_instruction=load_system_prompt(),
            thinking=GoogleLLMService.ThinkingConfig(
                thinking_level=os.getenv("GOOGLE_THINKING_LEVEL", "minimal"),
                include_thoughts=False
            ),
        ),
        function_call_timeout_secs=tool_timeout_seconds(),
        enable_async_tool_cancellation=True,
    )
