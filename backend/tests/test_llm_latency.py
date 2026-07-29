import asyncio
from types import SimpleNamespace

import pytest
from loguru import logger
from pipecat.services.google.llm import GoogleLLMService

from providers.llm.google_llm import FirstTokenTimeoutError, LatencyBoundGoogleLLMService
from providers.llm.stream_timeout import LLMStreamDeadlineError, bounded_openai_stream


def _chunk(text=None):
    part = SimpleNamespace(text=text, function_call=None, inline_data=None)
    content = SimpleNamespace(parts=[part])
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)])


@pytest.mark.anyio
async def test_google_stream_times_out_before_first_meaningful_output():
    async def stalled_stream():
        yield _chunk()
        await asyncio.Event().wait()

    with pytest.raises(FirstTokenTimeoutError):
        chunks = LatencyBoundGoogleLLMService._first_output_stream(stalled_stream(), 0.01)
        [chunk async for chunk in chunks]


@pytest.mark.anyio
async def test_google_stream_replays_metadata_then_first_output():
    metadata = _chunk()
    output = _chunk("hello")

    async def stream():
        yield metadata
        yield output

    chunks = [
        chunk
        async for chunk in LatencyBoundGoogleLLMService._first_output_stream(stream(), 0.1)
    ]
    assert chunks == [metadata, output]


@pytest.mark.anyio
async def test_google_timeout_becomes_spoken_recovery_chunk():
    async def stalled_stream():
        await asyncio.Event().wait()
        yield

    chunks = [
        chunk
        async for chunk in LatencyBoundGoogleLLMService._recovering_stream(
            stalled_stream(), 0.01, "Please try again."
        )
    ]
    assert len(chunks) == 1
    assert chunks[0].candidates[0].content.parts[0].text == "Please try again."


@pytest.mark.anyio
async def test_google_stream_creation_is_inside_first_token_deadline(monkeypatch):
    async def stalled_stream_creation(_self, _context):
        await asyncio.Event().wait()

    monkeypatch.setattr(GoogleLLMService, "_stream_content", stalled_stream_creation)
    service = object.__new__(LatencyBoundGoogleLLMService)
    service._first_token_timeout_secs = 0.01
    service._timeout_message = "Please try again."
    service._settings = SimpleNamespace(model="test-model")

    stream = await service._stream_content(None)
    chunks = [chunk async for chunk in stream]

    assert len(chunks) == 1
    assert chunks[0].candidates[0].content.parts[0].text == "Please try again."


@pytest.mark.anyio
async def test_openai_compatible_stream_has_first_output_deadline():
    async def stalled():
        await asyncio.Event().wait()
        yield None

    with pytest.raises(LLMStreamDeadlineError, match="first output"):
        async for _ in bounded_openai_stream(stalled(), 0.01, 1):
            pass


@pytest.mark.anyio
async def test_openai_compatible_stream_has_total_deadline():
    async def stream():
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))]
        )
        await asyncio.Event().wait()

    with pytest.raises(LLMStreamDeadlineError, match="total"):
        async for _ in bounded_openai_stream(stream(), 0.1, 0.02):
            pass


@pytest.mark.anyio
async def test_google_first_output_log_keeps_request_correlation_id():
    logs = []
    sink = logger.add(logs.append, format="{message}")
    try:
        async def stream():
            yield _chunk("answer")

        chunks = [
            chunk
            async for chunk in LatencyBoundGoogleLLMService._recovering_stream(
                stream(),
                0.2,
                "Please try again.",
                request_id="trace-123",
                provider_model="google-model",
            )
        ]
        assert chunks
    finally:
        logger.remove(sink)

    rendered = "".join(str(item) for item in logs)
    assert "request_id=trace-123" in rendered
    assert "provider=google" in rendered
    assert "status=first_output" in rendered
    assert "latency_ms=" in rendered
