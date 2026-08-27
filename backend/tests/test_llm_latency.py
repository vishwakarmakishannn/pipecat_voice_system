import asyncio
from types import SimpleNamespace

import pytest
import httpx
from loguru import logger
from openai import APIError
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.groq.llm import GroqLLMService

from providers.llm.google_llm import FirstTokenTimeoutError, LatencyBoundGoogleLLMService
from providers.llm.stream_timeout import (
    LLMStreamDeadlineError,
    bounded_openai_stream,
    chunk_is_recovery,
    recovering_openai_stream,
)


def _chunk(text=None):
    part = SimpleNamespace(text=text, function_call=None, inline_data=None)
    content = SimpleNamespace(parts=[part])
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)])


def _openai_chunk(text="hello"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text, tool_calls=None, function_call=None)
            )
        ],
        usage=None,
    )


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
async def test_google_empty_stream_is_a_first_output_failure():
    async def empty_stream():
        if False:
            yield None

    with pytest.raises(FirstTokenTimeoutError, match="before meaningful output"):
        async for _ in LatencyBoundGoogleLLMService._first_output_stream(
            empty_stream(), 0.1
        ):
            pass


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
async def test_google_silent_first_attempt_is_hedged_before_hard_deadline(monkeypatch):
    calls = 0

    async def fake_stream(_self, _context):
        nonlocal calls
        calls += 1
        if calls == 1:
            async def stalled():
                await asyncio.Event().wait()
                yield

            return stalled()

        async def answered():
            yield _chunk("hedged answer")

        return answered()

    monkeypatch.setattr(GoogleLLMService, "_stream_content", fake_stream)
    service = object.__new__(LatencyBoundGoogleLLMService)
    service._first_token_timeout_secs = 0.2
    service._total_timeout_secs = 1.0
    service._hedge_delay_secs = 0.02
    service._timeout_message = "Please try again."
    service._settings = SimpleNamespace(model="test-model")
    service.diagnostic_callback = None

    stream = await service._stream_content(None)
    chunks = [chunk async for chunk in stream]

    assert calls == 2
    assert chunks[0].candidates[0].content.parts[0].text == "hedged answer"


@pytest.mark.anyio
async def test_google_fast_first_output_never_launches_duplicate(monkeypatch):
    calls = 0

    async def fake_stream(_self, _context):
        nonlocal calls
        calls += 1

        async def answered():
            yield _chunk("first answer")

        return answered()

    monkeypatch.setattr(GoogleLLMService, "_stream_content", fake_stream)
    service = object.__new__(LatencyBoundGoogleLLMService)
    service._first_token_timeout_secs = 0.2
    service._total_timeout_secs = 1.0
    service._hedge_delay_secs = 0.05
    service._timeout_message = "Please try again."
    service._settings = SimpleNamespace(model="test-model")
    service.diagnostic_callback = None

    stream = await service._stream_content(None)
    chunks = [chunk async for chunk in stream]

    assert calls == 1
    assert chunks[0].candidates[0].content.parts[0].text == "first answer"


@pytest.mark.anyio
async def test_google_warmup_uses_existing_client_model_lookup():
    calls = []

    class Models:
        @staticmethod
        async def get(*, model):
            calls.append(model)
            return SimpleNamespace(name=model)

    service = object.__new__(LatencyBoundGoogleLLMService)
    service._client = SimpleNamespace(aio=SimpleNamespace(models=Models()))
    service._settings = SimpleNamespace(model="test-model")
    service._warmup_timeout_secs = 0.1
    service._connection_warmed = False

    assert await service.warm_connection() is True
    assert service.connection_warmed is True
    assert calls == ["test-model"]


@pytest.mark.anyio
async def test_openai_compatible_stream_has_first_output_deadline():
    async def stalled():
        await asyncio.Event().wait()
        yield None

    with pytest.raises(LLMStreamDeadlineError, match="first output"):
        async for _ in bounded_openai_stream(stalled(), 0.01, 1):
            pass


@pytest.mark.anyio
async def test_openai_compatible_empty_stream_is_a_first_output_failure():
    async def empty_stream():
        if False:
            yield None

    with pytest.raises(LLMStreamDeadlineError, match="before meaningful output"):
        async for _ in bounded_openai_stream(empty_stream(), 0.1, 1):
            pass


@pytest.mark.anyio
async def test_openai_compatible_metadata_only_stream_is_a_first_output_failure():
    async def metadata_only():
        yield _openai_chunk("")

    with pytest.raises(LLMStreamDeadlineError, match="before meaningful output"):
        async for _ in bounded_openai_stream(metadata_only(), 0.1, 1):
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
async def test_openai_compatible_timeout_becomes_spoken_recovery_and_diagnostic():
    diagnostics = []

    async def stalled():
        await asyncio.Event().wait()
        yield None

    bounded = bounded_openai_stream(stalled(), 0.01, 1)
    chunks = [
        chunk
        async for chunk in recovering_openai_stream(
            bounded,
            recovery_text="Please try that again.",
            model="test-model",
            request_id="request-1",
            started_at=asyncio.get_running_loop().time(),
            diagnostic_callback=lambda **payload: diagnostics.append(payload),
        )
    ]

    assert chunks[0].choices[0].delta.content == "Please try that again."
    assert chunk_is_recovery(chunks[0]) is True
    assert diagnostics[0]["code"] == "llm.first_output_timeout"
    assert diagnostics[0]["request_id"] == "request-1"
    assert diagnostics[0]["recovered"] is True


@pytest.mark.anyio
async def test_groq_transient_creation_failure_retries_within_one_deadline(monkeypatch):
    from providers.llm.groq_llm import LatencyBoundGroqLLMService

    calls = 0

    async def fake_create(_self, _context):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError

        async def stream():
            yield _openai_chunk("recovered answer")

        return stream()

    monkeypatch.setattr(GroqLLMService, "get_chat_completions", fake_create)
    monkeypatch.setenv("VOICE_LLM_FIRST_TOKEN_TIMEOUT_SECONDS", "0.5")
    monkeypatch.setenv("GROQ_LIVE_FIRST_ATTEMPT_TIMEOUT_SECONDS", "0.25")
    monkeypatch.setenv("VOICE_LLM_RETRY_RESERVE_SECONDS", "0.05")
    monkeypatch.setenv("GROQ_LIVE_MAX_ATTEMPTS", "2")
    service = object.__new__(LatencyBoundGroqLLMService)
    service._settings = SimpleNamespace(model="test-model", extra={})
    service._connection_warmed = True
    service.diagnostic_callback = None
    context = SimpleNamespace(tools=[], set_tools=lambda *_args: None)

    stream = await service.get_chat_completions(context)
    chunks = [chunk async for chunk in stream]

    assert calls == 2
    assert chunks[0].choices[0].delta.content == "recovered answer"


@pytest.mark.anyio
async def test_groq_terminal_creation_failure_emits_one_detailed_recovery(monkeypatch):
    from providers.llm.groq_llm import LatencyBoundGroqLLMService

    calls = 0
    diagnostics = []

    async def always_fail(_self, _context):
        nonlocal calls
        calls += 1
        raise TimeoutError

    monkeypatch.setattr(GroqLLMService, "get_chat_completions", always_fail)
    monkeypatch.setenv("VOICE_LLM_FIRST_TOKEN_TIMEOUT_SECONDS", "0.5")
    monkeypatch.setenv("GROQ_LIVE_FIRST_ATTEMPT_TIMEOUT_SECONDS", "0.25")
    monkeypatch.setenv("VOICE_LLM_RETRY_RESERVE_SECONDS", "0.05")
    monkeypatch.setenv("GROQ_LIVE_MAX_ATTEMPTS", "2")
    service = object.__new__(LatencyBoundGroqLLMService)
    service._settings = SimpleNamespace(model="test-model", extra={})
    service._connection_warmed = True
    service.diagnostic_callback = lambda **payload: diagnostics.append(payload)
    context = SimpleNamespace(tools=[], set_tools=lambda *_args: None)

    stream = await service.get_chat_completions(context)
    chunks = [chunk async for chunk in stream]

    assert calls == 2
    assert len(chunks) == 1
    assert len(diagnostics) == 1
    assert diagnostics[0]["code"] == "llm.stream_creation_timeout"
    assert diagnostics[0]["details"]["attempt_count"] == 2
    assert [item["attempt"] for item in diagnostics[0]["details"]["attempts"]] == [1, 2]


@pytest.mark.anyio
async def test_groq_retries_stream_that_stalls_before_meaningful_output(monkeypatch):
    from providers.llm.groq_llm import LatencyBoundGroqLLMService

    calls = 0

    async def fake_create(_self, _context):
        nonlocal calls
        calls += 1
        if calls == 1:
            async def stalled():
                yield SimpleNamespace(choices=[], usage=None)
                await asyncio.Event().wait()

            return stalled()

        async def answered():
            yield _openai_chunk("second attempt")

        return answered()

    monkeypatch.setattr(GroqLLMService, "get_chat_completions", fake_create)
    monkeypatch.setenv("VOICE_LLM_FIRST_TOKEN_TIMEOUT_SECONDS", "0.5")
    monkeypatch.setenv("GROQ_LIVE_FIRST_ATTEMPT_TIMEOUT_SECONDS", "0.25")
    monkeypatch.setenv("VOICE_LLM_RETRY_RESERVE_SECONDS", "0.05")
    monkeypatch.setenv("GROQ_LIVE_MAX_ATTEMPTS", "2")
    service = object.__new__(LatencyBoundGroqLLMService)
    service._settings = SimpleNamespace(model="test-model", extra={})
    service._connection_warmed = True
    service.diagnostic_callback = None
    context = SimpleNamespace(tools=[], set_tools=lambda *_args: None)

    stream = await service.get_chat_completions(context)
    chunks = [chunk async for chunk in stream]

    assert calls == 2
    assert chunks[0].choices[0].delta.content == "second attempt"


@pytest.mark.anyio
async def test_groq_retries_streamed_tool_validation_error_without_tools(monkeypatch):
    from providers.llm.groq_llm import LatencyBoundGroqLLMService

    calls = 0
    tool_updates = []

    async def fake_create(_self, _context):
        nonlocal calls
        calls += 1
        if calls == 1:
            async def rejected():
                raise APIError(
                    "Tool call validation failed: attempted to call tool 'raise_issue' "
                    "which was not in request.tools",
                    request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
                    body={"type": "invalid_request_error"},
                )
                yield

            return rejected()

        async def answered():
            yield _openai_chunk("Rohan reported a device transaction failure.")

        return answered()

    class Context:
        def __init__(self):
            self.tools = ["datetime", "search"]

        def set_tools(self, tools=None):
            self.tools = list(tools or [])
            tool_updates.append(list(self.tools))

    monkeypatch.setattr(GroqLLMService, "get_chat_completions", fake_create)
    monkeypatch.setenv("VOICE_LLM_FIRST_TOKEN_TIMEOUT_SECONDS", "0.5")
    monkeypatch.setenv("GROQ_LIVE_FIRST_ATTEMPT_TIMEOUT_SECONDS", "0.25")
    monkeypatch.setenv("VOICE_LLM_RETRY_RESERVE_SECONDS", "0.05")
    service = object.__new__(LatencyBoundGroqLLMService)
    service._settings = SimpleNamespace(model="test-model", extra={})
    service._connection_warmed = True
    service.diagnostic_callback = None
    context = Context()

    stream = await service.get_chat_completions(context)
    chunks = [chunk async for chunk in stream]

    assert calls == 2
    assert chunks[0].choices[0].delta.content.startswith("Rohan reported")
    assert tool_updates == [[], ["datetime", "search"]]
    assert context.tools == ["datetime", "search"]


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
