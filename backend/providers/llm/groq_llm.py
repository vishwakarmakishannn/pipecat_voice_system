import os
import asyncio
import time
import uuid
from loguru import logger
from openai import APIConnectionError, APIError, APIStatusError, BadRequestError, RateLimitError
from pipecat.services.groq.llm import GroqLLMService, GroqLLMSettings
from core.llm_config import (
    first_token_timeout_seconds,
    groq_first_attempt_timeout_seconds,
    groq_live_max_attempts,
    llm_retry_reserve_seconds,
    timeout_recovery_text,
    total_timeout_seconds,
)
from .groq_runtime import (
    get_shared_groq_client,
    groq_runtime_warmed,
    mark_groq_runtime_unwarmed,
    warm_groq_runtime,
)
from core.prompt_config import load_system_prompt
from core.tool_config import tool_timeout_seconds
from .stream_timeout import (
    bounded_openai_stream,
    chunk_has_meaningful_output,
    openai_recovery_stream,
    prefetch_openai_first_output,
    recovering_openai_stream,
    report_llm_deadline,
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


def _is_tool_call_validation_error(exc: BaseException) -> bool:
    return "tool call validation failed" in str(exc).lower()


class LatencyBoundGroqLLMService(GroqLLMService):
    # Groq's OpenAI-compatible endpoint accepts a broad message shape, but
    # Llama models do not reliably treat non-initial developer messages as
    # conversational evidence. Match Gemini's behavior for retrieved context:
    # keep the base system prompt as system, and send per-turn dynamic context
    # (web/RAG/memory developer messages) as user-visible context.
    supports_developer_role = False

    def __init__(self, **kwargs):
        self._runtime_api_key = kwargs.get("api_key")
        self._runtime_base_url = kwargs.get("base_url")
        super().__init__(**kwargs)
        self._connection_warmed = groq_runtime_warmed(
            api_key=self._runtime_api_key,
            base_url=self._runtime_base_url,
        )
        self.diagnostic_callback = None

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Use the process pool with SDK retries disabled for live inference."""
        del kwargs
        return get_shared_groq_client(api_key=api_key, base_url=base_url)

    @property
    def connection_warmed(self) -> bool:
        return self._connection_warmed

    async def warm_connection(self, timeout_seconds: float | None = None) -> bool:
        """Warm the exact shared client used by live inference."""
        del timeout_seconds  # Runtime owns the single configured warmup deadline.
        self._connection_warmed = await warm_groq_runtime(
            api_key=self._runtime_api_key,
            base_url=self._runtime_base_url,
        )
        return self._connection_warmed

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
        request. Its API can reject the completion either as an HTTP error or
        as an error event inside the stream. Retry that provider-specific
        validation failure once with tools disabled so the user still receives
        a spoken answer.
        """
        first_timeout = first_token_timeout_seconds()
        total_timeout = total_timeout_seconds()
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        model = self._settings.model
        reasoning_effort = (self._settings.extra or {}).get("reasoning_effort")
        cold_start = not bool(getattr(self, "_connection_warmed", False))
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
        recovery_text = timeout_recovery_text()
        loop = asyncio.get_running_loop()
        first_deadline = loop.time() + first_timeout
        reserve = min(llm_retry_reserve_seconds(), first_timeout / 3)
        configured_tools = context.tools
        configured_max_attempts = groq_live_max_attempts()
        # Preserve one immediate provider-validation recovery for tool turns,
        # without splitting ordinary direct-turn transport deadlines.
        max_attempts = max(
            configured_max_attempts,
            2 if configured_tools else 1,
        )
        attempt_limit = (
            min(groq_first_attempt_timeout_seconds(), first_timeout)
            if configured_max_attempts > 1
            else first_timeout
        )
        attempt_history: list[dict] = []
        last_error: BaseException | None = None
        tools_disabled = False
        stream = None

        for attempt in range(1, max_attempts + 1):
            remaining = first_deadline - loop.time()
            if remaining <= reserve:
                break
            timeout = remaining if attempt == max_attempts else min(attempt_limit, remaining - reserve)
            attempt_started = time.monotonic()
            attempt_deadline = loop.time() + timeout
            phase = "stream_creation"
            try:
                stream = await asyncio.wait_for(
                    super().get_chat_completions(context), timeout=max(0.001, timeout)
                )
                phase = "first_output"
                first_output_remaining = min(first_deadline, attempt_deadline) - loop.time()
                if first_output_remaining <= 0:
                    raise TimeoutError
                stream = await prefetch_openai_first_output(
                    stream,
                    first_output_remaining,
                )
                attempt_history.append(
                    {
                        "attempt": attempt,
                        "phase": phase,
                        "outcome": "completed",
                        "duration_ms": round((time.monotonic() - attempt_started) * 1000, 1),
                    }
                )
                break
            except asyncio.CancelledError:
                if tools_disabled:
                    context.set_tools(configured_tools)
                raise
            except BadRequestError as exc:
                last_error = exc
                retry_without_tools = (
                    _is_tool_call_validation_error(exc)
                    and not tools_disabled
                    and attempt < max_attempts
                )
                attempt_history.append(
                    {
                        "attempt": attempt,
                        "phase": phase,
                        "outcome": "invalid_tool_call" if retry_without_tools else "failed",
                        "error_type": type(exc).__name__,
                        "http_status": getattr(exc, "status_code", None),
                        "duration_ms": round((time.monotonic() - attempt_started) * 1000, 1),
                    }
                )
                if not retry_without_tools:
                    if tools_disabled:
                        context.set_tools(configured_tools)
                    raise
                logger.warning(
                    "voice_llm request_id={} provider=groq attempt={} status=invalid_tool_call "
                    "action=retry_without_tools remaining_ms={}",
                    request_id,
                    attempt,
                    round(max(0, first_deadline - loop.time()) * 1000),
                )
                context.set_tools()
                tools_disabled = True
            except (TimeoutError, APIConnectionError, RateLimitError, APIStatusError) as exc:
                last_error = exc
                if isinstance(exc, (TimeoutError, APIConnectionError)):
                    self._connection_warmed = False
                    mark_groq_runtime_unwarmed(
                        api_key=getattr(self, "_runtime_api_key", None),
                        base_url=getattr(self, "_runtime_base_url", None),
                    )
                status_code = getattr(exc, "status_code", None)
                retryable = isinstance(exc, (TimeoutError, APIConnectionError, RateLimitError)) or (
                    isinstance(status_code, int) and status_code >= 500
                )
                attempt_history.append(
                    {
                        "attempt": attempt,
                        "phase": phase,
                        "outcome": "timeout" if isinstance(exc, TimeoutError) else "failed",
                        "error_type": type(exc).__name__,
                        "http_status": status_code,
                        "duration_ms": round((time.monotonic() - attempt_started) * 1000, 1),
                    }
                )
                remaining_after = first_deadline - loop.time()
                if not retryable or attempt >= max_attempts or remaining_after <= reserve:
                    break
                logger.warning(
                    "voice_llm request_id={} provider=groq attempt={} status=retrying "
                    "error_type={} remaining_ms={}",
                    request_id,
                    attempt,
                    type(exc).__name__,
                    round(max(0, remaining_after) * 1000),
                )
            except APIError as exc:
                # Groq can accept the HTTP request and then report tool-call
                # validation as an SSE error while the stream is being read.
                # The OpenAI SDK surfaces that path as APIError rather than the
                # BadRequestError used for an ordinary HTTP 400 response.
                last_error = exc
                retry_without_tools = (
                    _is_tool_call_validation_error(exc)
                    and not tools_disabled
                    and attempt < max_attempts
                )
                attempt_history.append(
                    {
                        "attempt": attempt,
                        "phase": phase,
                        "outcome": "invalid_tool_call" if retry_without_tools else "failed",
                        "error_type": type(exc).__name__,
                        "duration_ms": round((time.monotonic() - attempt_started) * 1000, 1),
                    }
                )
                if not retry_without_tools:
                    if tools_disabled:
                        context.set_tools(configured_tools)
                    raise
                logger.warning(
                    "voice_llm request_id={} provider=groq attempt={} status=invalid_tool_call "
                    "source=stream action=retry_without_tools remaining_ms={}",
                    request_id,
                    attempt,
                    round(max(0, first_deadline - loop.time()) * 1000),
                )
                context.set_tools()
                tools_disabled = True
            except Exception:
                if tools_disabled:
                    context.set_tools(configured_tools)
                raise

        if tools_disabled:
            context.set_tools(configured_tools)

        if stream is None:
            if isinstance(last_error, RateLimitError):
                terminal_code = "llm.rate_limited"
            elif isinstance(last_error, APIConnectionError):
                terminal_code = "llm.connection_failed"
            elif isinstance(last_error, APIStatusError):
                terminal_code = "llm.provider_unavailable"
            else:
                terminal_code = "llm.stream_creation_timeout"
            report_llm_deadline(
                self.diagnostic_callback,
                code=terminal_code,
                request_id=request_id,
                duration_ms=round((time.monotonic() - started) * 1000, 1),
                recovery_text=recovery_text,
                details={
                    "attempts": attempt_history,
                    "attempt_count": len(attempt_history),
                    "max_attempts": max_attempts,
                    "phase": attempt_history[-1]["phase"] if attempt_history else "stream_creation",
                    "last_error_type": type(last_error).__name__ if last_error else "TimeoutError",
                },
            )
            return openai_recovery_stream(recovery_text, model)
        self._connection_warmed = True
        elapsed = time.monotonic() - started
        bounded_stream = bounded_openai_stream(
            stream,
            max(0.001, first_timeout - elapsed),
            max(0.001, total_timeout - elapsed),
        )
        recovery_stream = recovering_openai_stream(
            bounded_stream,
            recovery_text=recovery_text,
            model=model,
            request_id=request_id,
            started_at=started,
            diagnostic_callback=self.diagnostic_callback,
        )
        return self._instrumented_stream(
            recovery_stream,
            request_id=request_id,
            model=model,
            reasoning_effort=reasoning_effort,
            cold_start=cold_start,
            started_at=started,
        )

def get_groq_llm(*, system_instruction: str | None = None):
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
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
