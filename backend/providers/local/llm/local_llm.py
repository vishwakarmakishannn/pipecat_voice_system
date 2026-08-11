"""Pipecat LLM service backed by a local OpenAI-compatible llama.cpp server."""

import asyncio
import inspect
import time
import uuid

from loguru import logger
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService

from core.llm_config import first_token_timeout_seconds, total_timeout_seconds
from core.prompt_config import load_system_prompt
from core.tool_config import tool_timeout_seconds
from providers.llm.stream_timeout import (
    LLMStreamDeadlineError,
    bounded_openai_stream,
    chunk_has_meaningful_output,
)

from .config import LocalLLMConfig, load_local_llm_config
from .runtime import LocalLLMRuntime, get_local_llm_runtime


_MAX_AUTOMATIC_CONTINUATIONS = 2
_CONTINUATION_INSTRUCTION = (
    "Continue the assistant answer exactly where it stopped. Finish it concisely "
    "without repeating any text and without calling a tool."
)


def _value(value, *names):
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def _chunk_metrics(chunk) -> dict:
    values = {}
    usage = getattr(chunk, "usage", None)
    if usage:
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            candidate = _value(usage, name)
            if candidate is not None:
                values[name] = candidate
    model_extra = getattr(chunk, "model_extra", None) or {}
    timings = _value(model_extra, "timings")
    if timings:
        values["server_timings"] = timings
    finish_reasons = [
        getattr(choice, "finish_reason", None)
        for choice in (getattr(chunk, "choices", None) or [])
    ]
    finish_reasons = [reason for reason in finish_reasons if reason is not None]
    if finish_reasons:
        values["finish_reason"] = finish_reasons[-1]
    return values


def _chunk_content(chunk) -> str:
    parts = []
    for choice in getattr(chunk, "choices", None) or []:
        delta = getattr(choice, "delta", None)
        content = getattr(delta, "content", None) if delta else None
        if content:
            parts.append(content)
    return "".join(parts)


def _chunk_has_tool_call(chunk) -> bool:
    for choice in getattr(chunk, "choices", None) or []:
        delta = getattr(choice, "delta", None)
        if delta and (
            getattr(delta, "tool_calls", None)
            or getattr(delta, "function_call", None)
        ):
            return True
    return False


def _chunk_hit_length_limit(chunk) -> bool:
    return any(
        str(getattr(choice, "finish_reason", "")).lower() == "length"
        for choice in (getattr(chunk, "choices", None) or [])
    )


async def _close_stream(stream) -> None:
    close = getattr(stream, "close", None) or getattr(stream, "aclose", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


class LocalLLMService(OpenAILLMService):
    """Stream local completions with the same voice latency deadlines as cloud."""

    supports_developer_role = False

    def __init__(
        self,
        *,
        runtime: LocalLLMRuntime,
        config: LocalLLMConfig,
        **kwargs,
    ):
        self._runtime = runtime
        self._local_config = config
        super().__init__(
            api_key=config.api_key,
            base_url=config.base_url,
            **kwargs,
        )

    def create_client(self, **kwargs):
        """Reuse the runtime client instead of opening one pool per session."""
        return self._runtime.client

    @property
    def connection_warmed(self) -> bool:
        return self._runtime.warmed

    async def warm_connection(self, timeout_seconds: float | None = None) -> bool:
        # Startup performs a strict warmup. Retain the per-session hook for
        # direct run_bot tests and non-standard launch paths.
        if timeout_seconds is None:
            await self._runtime.warm()
        else:
            await asyncio.wait_for(self._runtime.warm(), timeout=timeout_seconds)
        return True

    async def _complete_truncated_response(self, initial_stream, context: LLMContext):
        """Stream a response and transparently finish provider-truncated text."""
        stream = initial_stream
        complete_text = ""
        continuation_count = 0

        while True:
            hit_length_limit = False
            emitted_tool_call = False
            try:
                async for chunk in stream:
                    complete_text += _chunk_content(chunk)
                    emitted_tool_call = emitted_tool_call or _chunk_has_tool_call(chunk)
                    hit_length_limit = hit_length_limit or _chunk_hit_length_limit(chunk)
                    yield chunk
            finally:
                await _close_stream(stream)

            if not hit_length_limit or emitted_tool_call or not complete_text.strip():
                return
            if continuation_count >= _MAX_AUTOMATIC_CONTINUATIONS:
                logger.error(
                    "voice_llm provider=local status=continuation_limit_reached "
                    "continuations={} partial_chars={}",
                    continuation_count,
                    len(complete_text),
                )
                return

            continuation_count += 1
            logger.warning(
                "voice_llm provider=local status=length_limited "
                "action=continue continuation={} partial_chars={}",
                continuation_count,
                len(complete_text),
            )
            continuation_context = LLMContext(
                messages=[
                    *list(context.messages),
                    {"role": "assistant", "content": complete_text},
                    {"role": "user", "content": _CONTINUATION_INSTRUCTION},
                ]
            )
            stream = await super().get_chat_completions(continuation_context)

    @staticmethod
    async def _instrumented_stream(
        stream,
        *,
        request_id: str,
        model: str,
        cold_start: bool,
        started_at: float,
    ):
        first_output_seen = False
        metrics = {}
        status = "completed"
        try:
            async for chunk in stream:
                if not first_output_seen and chunk_has_meaningful_output(chunk):
                    first_output_seen = True
                    logger.info(
                        "voice_llm request_id={} provider=local model={} "
                        "cold_start={} status=first_output latency_ms={}",
                        request_id,
                        model,
                        cold_start,
                        round((time.monotonic() - started_at) * 1000, 1),
                    )
                metrics.update(_chunk_metrics(chunk))
                yield chunk
        except BaseException as exc:
            status = (
                "cancelled"
                if isinstance(exc, asyncio.CancelledError)
                else "failed"
            )
            raise
        finally:
            logger.info(
                "voice_llm request_id={} provider=local model={} cold_start={} "
                "status={} latency_ms={} first_output_seen={} metrics={}",
                request_id,
                model,
                cold_start,
                status,
                round((time.monotonic() - started_at) * 1000, 1),
                first_output_seen,
                metrics,
            )

    async def get_chat_completions(self, context):
        first_timeout = first_token_timeout_seconds()
        total_timeout = total_timeout_seconds()
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        model = self._settings.model
        cold_start = not self._runtime.warmed
        logger.info(
            "voice_llm request_id={} provider=local model={} cold_start={} "
            "status=started first_output_deadline_ms={} total_deadline_ms={}",
            request_id,
            model,
            cold_start,
            round(first_timeout * 1000),
            round(total_timeout * 1000),
        )
        try:
            stream = await asyncio.wait_for(
                super().get_chat_completions(context),
                timeout=first_timeout,
            )
        except TimeoutError as exc:
            raise LLMStreamDeadlineError(
                "Local LLM stream creation deadline exceeded"
            ) from exc

        elapsed = time.monotonic() - started
        completion_safe_stream = self._complete_truncated_response(stream, context)
        bounded_stream = bounded_openai_stream(
            completion_safe_stream,
            max(0.001, first_timeout - elapsed),
            max(0.001, total_timeout - elapsed),
        )
        return self._instrumented_stream(
            bounded_stream,
            request_id=request_id,
            model=model,
            cold_start=cold_start,
            started_at=started,
        )


def get_local_llm(*, system_instruction: str | None = None) -> LocalLLMService:
    from core.admission import current_voice_slot_id

    config = load_local_llm_config()
    runtime = get_local_llm_runtime(config)
    extra_body = dict(config.extra_body)
    slot_id = current_voice_slot_id.get()
    if slot_id is not None:
        # llama.cpp prompt caches are slot-local. Keep every turn in this
        # voice session on its reserved slot so the growing prefix is reused.
        extra_body["id_slot"] = slot_id
    return LocalLLMService(
        runtime=runtime,
        config=config,
        settings=LocalLLMService.Settings(
            model=config.model,
            system_instruction=system_instruction or load_system_prompt(),
            temperature=config.temperature,
            top_p=config.top_p,
            presence_penalty=config.presence_penalty,
            max_tokens=config.max_tokens,
            extra={
                "parallel_tool_calls": False,
                "extra_body": extra_body,
            },
        ),
        function_call_timeout_secs=tool_timeout_seconds(),
        enable_async_tool_cancellation=True,
        retry_on_timeout=False,
    )
