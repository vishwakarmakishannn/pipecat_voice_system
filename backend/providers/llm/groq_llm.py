import os
import asyncio
import time
import uuid
from loguru import logger
from openai import BadRequestError
from pipecat.services.groq.llm import GroqLLMService, GroqLLMSettings
from core.llm_config import first_token_timeout_seconds, total_timeout_seconds
from core.prompt_config import load_system_prompt
from core.tool_config import tool_timeout_seconds
from .stream_timeout import (
    LLMStreamDeadlineError,
    bounded_openai_stream,
    chunk_has_meaningful_output,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


def _groq_completion_settings(model: str) -> dict:
    """Return only parameters accepted by the selected Groq model.

    Pipecat merges ``GroqLLMSettings.extra`` into each chat-completion request.
    ``client_kwargs`` is intentionally not used for request parameters.
    """
    extra = {
        "parallel_tool_calls": _env_bool("GROQ_PARALLEL_TOOL_CALLS", False),
    }
    if model.startswith("openai/gpt-oss-"):
        effort = os.getenv("GROQ_REASONING_EFFORT", "low").strip().lower()
        if effort not in {"low", "medium", "high"}:
            raise ValueError(
                "GROQ_REASONING_EFFORT must be low, medium, or high, "
                f"got {effort!r}"
            )
        extra.update(
            {
                "reasoning_effort": effort,
                "extra_body": {
                    "include_reasoning": _env_bool(
                        "GROQ_INCLUDE_REASONING", False
                    ),
                },
            }
        )
    return extra


def _usage_value(value, *names):
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def _chunk_usage(chunk) -> dict:
    usage = getattr(chunk, "usage", None)
    if not usage:
        return {}
    prompt_details = _usage_value(usage, "prompt_tokens_details") or {}
    completion_details = _usage_value(usage, "completion_tokens_details") or {}
    values = {
        "prompt_tokens": _usage_value(usage, "prompt_tokens"),
        "completion_tokens": _usage_value(usage, "completion_tokens"),
        "total_tokens": _usage_value(usage, "total_tokens"),
        "cached_tokens": _usage_value(prompt_details, "cached_tokens"),
        "reasoning_tokens": _usage_value(completion_details, "reasoning_tokens"),
        "queue_time": _usage_value(usage, "queue_time"),
        "prompt_time": _usage_value(usage, "prompt_time"),
        "completion_time": _usage_value(usage, "completion_time"),
    }
    return {key: value for key, value in values.items() if value is not None}


class LatencyBoundGroqLLMService(GroqLLMService):
    # Groq's OpenAI-compatible endpoint accepts a broad message shape, but
    # Llama models do not reliably treat non-initial developer messages as
    # conversational evidence. Match Gemini's behavior for retrieved context:
    # keep the base system prompt as system, and send per-turn dynamic context
    # (web/RAG/memory developer messages) as user-visible context.
    supports_developer_role = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._connection_warmed = False
        self._warmup_attempted = False

    @property
    def connection_warmed(self) -> bool:
        return self._connection_warmed

    async def warm_connection(self, timeout_seconds: float | None = None) -> bool:
        """Boundedly warm this service's authenticated HTTP connection."""
        if self._connection_warmed:
            return True
        self._warmup_attempted = True
        raw_timeout = os.getenv("GROQ_WARMUP_TIMEOUT_SECONDS", "1.5")
        try:
            timeout = float(raw_timeout) if timeout_seconds is None else timeout_seconds
        except ValueError as exc:
            raise ValueError(
                f"GROQ_WARMUP_TIMEOUT_SECONDS must be a number, got {raw_timeout!r}"
            ) from exc
        if not 0.1 <= timeout <= 10.0:
            raise ValueError(
                "GROQ_WARMUP_TIMEOUT_SECONDS must be between 0.1 and 10.0, "
                f"got {timeout}"
            )

        started = time.monotonic()
        try:
            await asyncio.wait_for(self._client.models.list(), timeout=timeout)
        except Exception as exc:
            logger.warning(
                "voice_llm provider=groq model={} status=warmup_failed "
                "latency_ms={} error_type={}",
                self._settings.model,
                round((time.monotonic() - started) * 1000, 1),
                type(exc).__name__,
            )
            return False

        self._connection_warmed = True
        logger.info(
            "voice_llm provider=groq model={} status=warmed latency_ms={}",
            self._settings.model,
            round((time.monotonic() - started) * 1000, 1),
        )
        return True

    @staticmethod
    async def _instrumented_stream(
        stream,
        *,
        request_id: str,
        model: str,
        reasoning_effort: str | None,
        cold_start: bool,
        started_at: float,
    ):
        first_raw_seen = False
        first_output_seen = False
        usage = {}
        status = "completed"
        try:
            async for chunk in stream:
                elapsed_ms = round((time.monotonic() - started_at) * 1000, 1)
                if not first_raw_seen:
                    first_raw_seen = True
                    logger.info(
                        "voice_llm request_id={} provider=groq model={} "
                        "reasoning_effort={} cold_start={} status=first_raw_chunk "
                        "latency_ms={}",
                        request_id,
                        model,
                        reasoning_effort,
                        cold_start,
                        elapsed_ms,
                    )
                if not first_output_seen and chunk_has_meaningful_output(chunk):
                    first_output_seen = True
                    logger.info(
                        "voice_llm request_id={} provider=groq model={} "
                        "reasoning_effort={} cold_start={} status=first_output "
                        "latency_ms={}",
                        request_id,
                        model,
                        reasoning_effort,
                        cold_start,
                        elapsed_ms,
                    )
                usage.update(_chunk_usage(chunk))
                yield chunk
        except BaseException as exc:
            status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            raise
        finally:
            logger.info(
                "voice_llm request_id={} provider=groq model={} reasoning_effort={} "
                "cold_start={} status={} latency_ms={} first_raw_seen={} "
                "first_output_seen={} usage={}",
                request_id,
                model,
                reasoning_effort,
                cold_start,
                status,
                round((time.monotonic() - started_at) * 1000, 1),
                first_raw_seen,
                first_output_seen,
                usage,
            )

    async def get_chat_completions(self, context):
        """Apply deadlines at Pipecat's actual Groq request hook.

        Groq sometimes invents a tool name that was not included in the
        request. Its API rejects the entire completion before streaming. Retry
        that provider-specific validation failure once with tools disabled so
        the user still receives a spoken answer.
        """
        first_timeout = first_token_timeout_seconds()
        total_timeout = total_timeout_seconds()
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        model = self._settings.model
        reasoning_effort = (self._settings.extra or {}).get("reasoning_effort")
        cold_start = not self._connection_warmed
        logger.info(
            "voice_llm request_id={} provider=groq model={} reasoning_effort={} "
            "cold_start={} status=started first_output_deadline_ms={} "
            "total_deadline_ms={}",
            request_id,
            model,
            reasoning_effort,
            cold_start,
            round(first_timeout * 1000),
            round(total_timeout * 1000),
        )
        try:
            stream = await asyncio.wait_for(
                super().get_chat_completions(context), timeout=first_timeout
            )
        except BadRequestError as exc:
            if "tool call validation failed" not in str(exc).lower():
                raise
            configured_tools = context.tools
            logger.warning(
                "voice_llm provider=groq status=invalid_tool_call action=retry_without_tools"
            )
            context.set_tools()
            try:
                stream = await asyncio.wait_for(
                    super().get_chat_completions(context), timeout=first_timeout
                )
            finally:
                context.set_tools(configured_tools)
        except TimeoutError as exc:
            raise LLMStreamDeadlineError("Groq stream creation deadline exceeded") from exc
        self._connection_warmed = True
        elapsed = time.monotonic() - started
        bounded_stream = bounded_openai_stream(
            stream,
            max(0.001, first_timeout - elapsed),
            max(0.001, total_timeout - elapsed),
        )
        return self._instrumented_stream(
            bounded_stream,
            request_id=request_id,
            model=model,
            reasoning_effort=reasoning_effort,
            cold_start=cold_start,
            started_at=started,
        )

def get_groq_llm(*, system_instruction: str | None = None):
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
    return LatencyBoundGroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        settings=GroqLLMSettings(
            model=model,
            system_instruction=system_instruction or load_system_prompt(),
            extra=_groq_completion_settings(model),
        ),
        function_call_timeout_secs=tool_timeout_seconds(),
        enable_async_tool_cancellation=True,
    )
