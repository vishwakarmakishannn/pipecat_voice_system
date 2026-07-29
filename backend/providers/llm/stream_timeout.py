import asyncio


class LLMStreamDeadlineError(TimeoutError):
    pass


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
            except StopAsyncIteration:
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
