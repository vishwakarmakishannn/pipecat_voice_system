import asyncio
import time
import uuid

from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
)


class LLMStreamDeadlineError(TimeoutError):
    pass


def openai_text_chunk(text: str, model: str) -> ChatCompletionChunk:
    """Build a provider-neutral OpenAI-compatible recovery chunk."""
    return ChatCompletionChunk(
        id=f"recovery-{uuid.uuid4().hex}",
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(content=text, role="assistant"),
                finish_reason="stop",
            )
        ],
        created=int(time.time()),
        model=model,
        object="chat.completion.chunk",
    )


def chunk_is_recovery(chunk) -> bool:
    return str(getattr(chunk, "id", "")).startswith("recovery-")


def report_llm_deadline(
    diagnostic_callback,
    *,
    code: str,
    request_id: str,
    duration_ms: float,
    recovery_text: str,
    details: dict | None = None,
) -> None:
    if diagnostic_callback:
        diagnostic_callback(
            component="llm",
            code=code,
            severity="error",
            outcome="recovered",
            safe_message="The language model exceeded its response deadline.",
            request_id=request_id,
            duration_ms=duration_ms,
            retryable=True,
            recovered=True,
            fatal=False,
            details={"spoken_recovery": recovery_text, **(details or {})},
        )


async def openai_recovery_stream(text: str, model: str):
    yield openai_text_chunk(text, model)


async def recovering_openai_stream(
    stream,
    *,
    recovery_text: str,
    model: str,
    request_id: str,
    started_at: float,
    diagnostic_callback=None,
):
    """Convert a streaming deadline into one complete spoken recovery turn."""
    try:
        async for chunk in stream:
            yield chunk
    except LLMStreamDeadlineError as exc:
        code = (
            "llm.total_timeout"
            if "total" in str(exc).lower()
            else "llm.first_output_timeout"
        )
        report_llm_deadline(
            diagnostic_callback,
            code=code,
            request_id=request_id,
            duration_ms=round((time.monotonic() - started_at) * 1000, 1),
            recovery_text=recovery_text,
        )
        yield openai_text_chunk(recovery_text, model)


def chunk_has_meaningful_output(chunk) -> bool:
    for choice in getattr(chunk, "choices", None) or []:
        delta = getattr(choice, "delta", None)
        if delta and (
            getattr(delta, "content", None)
            or getattr(delta, "tool_calls", None)
            or getattr(delta, "function_call", None)
        ):
            return True
    return False


async def prefetch_openai_first_output(stream, timeout_seconds: float):
    """Buffer through the first meaningful chunk without leaking partial output.

    Returning only after text or a native tool delta is available lets a caller
    safely retry a stalled connection. Once this function succeeds, retries must
    stop because replaying could duplicate speech or a tool invocation.
    """
    iterator = stream.__aiter__()
    buffered = []

    async def close_underlying():
        if hasattr(iterator, "aclose"):
            await iterator.aclose()
        if iterator is not stream:
            if hasattr(stream, "close"):
                await stream.close()
            elif hasattr(stream, "aclose"):
                await stream.aclose()

    try:
        async with asyncio.timeout(timeout_seconds):
            while True:
                try:
                    chunk = await anext(iterator)
                except StopAsyncIteration as exc:
                    raise LLMStreamDeadlineError(
                        "LLM stream ended before meaningful output"
                    ) from exc
                buffered.append(chunk)
                if chunk_has_meaningful_output(chunk):
                    break
    except BaseException:
        await close_underlying()
        raise

    async def replay():
        try:
            for chunk in buffered:
                yield chunk
            async for chunk in iterator:
                yield chunk
        finally:
            await close_underlying()

    return replay()


async def bounded_openai_stream(stream, first_output_seconds: float, total_seconds: float):
    iterator = stream.__aiter__()
    loop = asyncio.get_running_loop()
    total_deadline = loop.time() + total_seconds
    first_deadline = min(total_deadline, loop.time() + first_output_seconds)
    first_seen = False
    try:
        while True:
            deadline = total_deadline if first_seen else first_deadline
            remaining = deadline - loop.time()
            if remaining <= 0:
                phase = "total" if first_seen else "first output"
                raise LLMStreamDeadlineError(f"LLM {phase} deadline exceeded")
            try:
                chunk = await asyncio.wait_for(anext(iterator), timeout=remaining)
            except StopAsyncIteration as exc:
                if not first_seen:
                    raise LLMStreamDeadlineError(
                        "LLM stream ended before meaningful output"
                    ) from exc
                return
            except TimeoutError as exc:
                phase = "total" if first_seen else "first output"
                raise LLMStreamDeadlineError(f"LLM {phase} deadline exceeded") from exc
            yield chunk
            if chunk_has_meaningful_output(chunk):
                first_seen = True
    finally:
        if hasattr(iterator, "aclose"):
            await iterator.aclose()
