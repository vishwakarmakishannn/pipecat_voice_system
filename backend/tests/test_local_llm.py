import asyncio
from types import SimpleNamespace

import pytest
from openai import NOT_GIVEN
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.processors.aggregators.llm_context import LLMContext

from providers.local.llm.config import LocalLLMConfig, load_local_llm_config
from providers.local.llm.local_llm import LocalLLMService, _tool_schema_count
from providers.local.llm import local_llm as local_llm_module
from providers.local.llm.runtime import LocalLLMRuntime
from tools.registry import configured_openai_tool_schemas


async def iter_async(*items):
    for item in items:
        yield item


def _config(**updates) -> LocalLLMConfig:
    values = {
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "qwen3-4b-local",
        "api_key": "local-no-key",
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 512,
        "warmup_timeout_seconds": 2.0,
        "max_concurrent_sessions": 2,
        "validate_server_slots": False,
    }
    values.update(updates)
    return LocalLLMConfig(**values)


class FakeCompletions:
    def __init__(self, response=None):
        self.calls = []
        self.response = response or SimpleNamespace()

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, *, model="qwen3-4b-local", response=None):
        self.model_calls = 0
        self.closed = False
        self.completions = FakeCompletions(response)
        self.chat = SimpleNamespace(completions=self.completions)
        self._model = model
        self.models = SimpleNamespace(list=self.list_models)

    async def list_models(self):
        self.model_calls += 1
        await asyncio.sleep(0)
        return SimpleNamespace(data=[SimpleNamespace(id=self._model)])

    async def close(self):
        self.closed = True


def test_tool_schema_count_supports_pipecat_schema_and_legacy_lists():
    schema = SimpleNamespace(
        standard_tools=[object(), object()],
        custom_tools={"openai": [{"type": "web_search"}]},
    )

    assert _tool_schema_count(schema) == 3
    assert _tool_schema_count([object(), object()]) == 2
    assert _tool_schema_count(None) == 0


def test_local_config_defaults_match_tested_qwen_profile(monkeypatch):
    for name in (
        "LOCAL_LLM_BASE_URL",
        "LOCAL_LLM_MODEL",
        "LOCAL_LLM_API_KEY",
        "LOCAL_LLM_TEMPERATURE",
        "LOCAL_LLM_TOP_P",
        "LOCAL_LLM_TOP_K",
        "LOCAL_LLM_MIN_P",
        "LOCAL_LLM_PRESENCE_PENALTY",
        "LOCAL_LLM_MAX_TOKENS",
        "LOCAL_LLM_WARMUP_TIMEOUT_SECONDS",
        "LOCAL_LLM_MAX_CONCURRENT_SESSIONS",
        "LOCAL_LLM_VALIDATE_SERVER_SLOTS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VOICE_MAX_CONCURRENT_SESSIONS", "2")

    config = load_local_llm_config()

    assert config.base_url == "http://127.0.0.1:8080/v1"
    assert config.model == "qwen3-4b-local"
    assert config.max_tokens == 512
    assert config.max_concurrent_sessions == 2
    assert config.validate_server_slots is True
    assert config.extra_body == {
        "top_k": 20,
        "min_p": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none",
        "cache_prompt": True,
    }


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("LOCAL_LLM_MODEL", "  ", "must not be empty"),
        ("LOCAL_LLM_TEMPERATURE", "2.1", "must be between"),
        ("LOCAL_LLM_MAX_TOKENS", "0", "must be between"),
        ("LOCAL_LLM_TOP_K", "many", "must be an integer"),
    ],
)
def test_local_config_rejects_invalid_values(monkeypatch, name, value, match):
    monkeypatch.setenv("VOICE_MAX_CONCURRENT_SESSIONS", "2")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=match):
        load_local_llm_config()


@pytest.mark.anyio
async def test_runtime_warms_once_under_concurrent_calls_and_closes():
    client = FakeClient()
    runtime = LocalLLMRuntime(_config(), client_factory=lambda _config: client)

    await asyncio.gather(runtime.warm(), runtime.warm())

    assert runtime.warmed is True
    assert client.model_calls == 1
    assert len(client.completions.calls) == 2
    assert client.completions.calls[0]["extra_body"]["reasoning_effort"] == "none"
    assert client.completions.calls[0]["extra_body"]["cache_prompt"] is True
    assert sorted(
        call["extra_body"]["id_slot"]
        for call in client.completions.calls
    ) == [0, 1]
    warmup_tool_names = [
        tool["function"]["name"]
        for tool in client.completions.calls[0]["tools"]
    ]
    assert warmup_tool_names == [
        schema["function"]["name"]
        for schema in configured_openai_tool_schemas()
    ]
    assert "manage_issue_draft" in warmup_tool_names
    assert "get_current_datetime" in warmup_tool_names
    assert "tavily_search" not in warmup_tool_names
    assert client.completions.calls[0]["tool_choice"] == "auto"

    await runtime.close()
    await runtime.close()
    assert client.closed is True


@pytest.mark.anyio
async def test_runtime_warms_the_production_system_prompt(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(
        "providers.local.llm.runtime.load_system_prompt",
        lambda: "the complete production system prompt",
    )
    runtime = LocalLLMRuntime(_config(), client_factory=lambda _config: client)

    await runtime.warm()

    assert client.completions.calls[0]["messages"][0] == {
        "role": "system",
        "content": "the complete production system prompt",
    }
    assert client.completions.calls[0]["extra_body"]["cache_prompt"] is True
    assert len(client.completions.calls) == runtime.config.max_concurrent_sessions
    await runtime.close()


@pytest.mark.anyio
async def test_runtime_fails_startup_when_configured_model_is_missing():
    client = FakeClient(model="different-model")
    runtime = LocalLLMRuntime(_config(), client_factory=lambda _config: client)

    with pytest.raises(RuntimeError, match="is unavailable"):
        await runtime.warm()

    assert runtime.warmed is False
    await runtime.close()


@pytest.mark.anyio
async def test_runtime_rejects_fewer_llama_slots_than_admitted_sessions():
    client = FakeClient()

    async def one_slot(_config):
        return 1

    runtime = LocalLLMRuntime(
        _config(validate_server_slots=True),
        client_factory=lambda _config: client,
        slot_probe=one_slot,
    )

    with pytest.raises(RuntimeError, match="server=1, backend=2"):
        await runtime.warm()

    assert client.completions.calls == []
    await runtime.close()


@pytest.mark.anyio
async def test_local_service_streams_and_sends_voice_and_tool_settings(monkeypatch):
    captured = {}

    async def response_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Hello", tool_calls=None)
                )
            ],
            usage=None,
            model_extra=None,
        )

    class Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return response_stream()

    client = FakeClient()
    client.chat = SimpleNamespace(completions=Completions())
    runtime = LocalLLMRuntime(_config(), client_factory=lambda _config: client)
    runtime._warmed = True
    service = LocalLLMService(
        runtime=runtime,
        config=runtime.config,
        settings=LocalLLMService.Settings(
            model=runtime.config.model,
            system_instruction="system prompt",
            temperature=runtime.config.temperature,
            top_p=runtime.config.top_p,
            presence_penalty=runtime.config.presence_penalty,
            max_tokens=runtime.config.max_tokens,
            extra={
                "parallel_tool_calls": False,
                "extra_body": runtime.config.extra_body,
            },
        ),
        function_call_timeout_secs=3.0,
        enable_async_tool_cancellation=True,
        retry_on_timeout=False,
    )
    monkeypatch.setenv("VOICE_LLM_FIRST_TOKEN_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("VOICE_LLM_TOTAL_TIMEOUT_SECONDS", "2")
    context = LLMContext(
        messages=[
            {"role": "user", "content": "Use the retrieved fact."},
            {"role": "developer", "content": "Retrieved fact: blue."},
        ],
        tools=[
            FunctionSchema(
                name="lookup",
                description="Look something up.",
                properties={},
                required=[],
            )
        ],
        tool_choice={
            "type": "function",
            "function": {"name": "lookup"},
        },
    )

    stream = await service.get_chat_completions(context)
    chunks = [chunk async for chunk in stream]

    assert len(chunks) == 1
    assert captured["model"] == "qwen3-4b-local"
    assert captured["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "Use the retrieved fact."},
        {"role": "user", "content": "Retrieved fact: blue."},
    ]
    assert captured["parallel_tool_calls"] is False
    assert captured["tool_choice"] == {
        "type": "function",
        "function": {"name": "lookup"},
    }
    assert captured["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert captured["max_tokens"] == 512
    await runtime.close()


@pytest.mark.anyio
async def test_local_service_continues_a_length_limited_text_response(monkeypatch):
    calls = []

    def chunk(content=None, finish_reason=None):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=content, tool_calls=None),
                    finish_reason=finish_reason,
                )
            ],
            usage=None,
            model=None,
            model_extra=None,
        )

    streams = [
        iter_async(chunk("The complete answer starts"), chunk(finish_reason="length")),
        iter_async(chunk(" and finishes here."), chunk(finish_reason="stop")),
    ]

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return streams.pop(0)

    client = FakeClient()
    client.chat = SimpleNamespace(completions=Completions())
    runtime = LocalLLMRuntime(_config(), client_factory=lambda _config: client)
    runtime._warmed = True
    service = LocalLLMService(
        runtime=runtime,
        config=runtime.config,
        settings=LocalLLMService.Settings(
            model=runtime.config.model,
            system_instruction="system prompt",
            max_tokens=runtime.config.max_tokens,
        ),
        retry_on_timeout=False,
    )
    monkeypatch.setenv("VOICE_LLM_FIRST_TOKEN_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("VOICE_LLM_TOTAL_TIMEOUT_SECONDS", "2")
    context = LLMContext(messages=[{"role": "user", "content": "Explain it."}])

    stream = await service.get_chat_completions(context)
    chunks = [item async for item in stream]
    text = "".join(
        item.choices[0].delta.content or ""
        for item in chunks
        if item.choices and item.choices[0].delta
    )

    assert text == "The complete answer starts and finishes here."
    assert len(calls) == 2
    assert calls[1]["messages"][-2] == {
        "role": "assistant",
        "content": "The complete answer starts",
    }
    assert "Continue the assistant answer exactly where it stopped" in calls[1]["messages"][-1]["content"]
    assert calls[1].get("tools") is NOT_GIVEN
    await runtime.close()


@pytest.mark.anyio
@pytest.mark.parametrize("finish_reason", ["stop", "tool_calls"])
async def test_local_service_does_not_continue_complete_or_tool_responses(
    monkeypatch,
    finish_reason,
):
    calls = []
    tool_calls = [SimpleNamespace()] if finish_reason == "tool_calls" else None

    async def response_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None if tool_calls else "Finished.",
                        tool_calls=tool_calls,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=None,
            model=None,
            model_extra=None,
        )

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return response_stream()

    client = FakeClient()
    client.chat = SimpleNamespace(completions=Completions())
    runtime = LocalLLMRuntime(_config(), client_factory=lambda _config: client)
    runtime._warmed = True
    service = LocalLLMService(
        runtime=runtime,
        config=runtime.config,
        settings=LocalLLMService.Settings(
            model=runtime.config.model,
            system_instruction="system prompt",
            max_tokens=runtime.config.max_tokens,
        ),
        retry_on_timeout=False,
    )
    monkeypatch.setenv("VOICE_LLM_FIRST_TOKEN_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("VOICE_LLM_TOTAL_TIMEOUT_SECONDS", "2")

    stream = await service.get_chat_completions(
        LLMContext(messages=[{"role": "user", "content": "Answer."}])
    )
    _ = [item async for item in stream]

    assert len(calls) == 1
    await runtime.close()


def test_local_service_reuses_runtime_client():
    client = FakeClient()
    runtime = LocalLLMRuntime(_config(), client_factory=lambda _config: client)

    first = LocalLLMService(
        runtime=runtime,
        config=runtime.config,
        settings=LocalLLMService.Settings(model=runtime.config.model),
    )
    second = LocalLLMService(
        runtime=runtime,
        config=runtime.config,
        settings=LocalLLMService.Settings(model=runtime.config.model),
    )

    assert first._client is client
    assert second._client is client


def test_local_factory_pins_all_session_requests_to_admission_slot(monkeypatch):
    from core.admission import current_voice_slot_id

    runtime = LocalLLMRuntime(_config(), client_factory=lambda _config: FakeClient())
    monkeypatch.setattr(local_llm_module, "load_local_llm_config", lambda: runtime.config)
    monkeypatch.setattr(local_llm_module, "get_local_llm_runtime", lambda _config: runtime)
    token = current_voice_slot_id.set(1)
    try:
        service = local_llm_module.get_local_llm()
    finally:
        current_voice_slot_id.reset(token)

    assert service._settings.extra["extra_body"]["id_slot"] == 1
    assert service._settings.extra["extra_body"]["cache_prompt"] is True
