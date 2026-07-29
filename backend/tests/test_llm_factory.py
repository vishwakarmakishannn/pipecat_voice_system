import sys
from types import SimpleNamespace

import pytest

from providers.llm.factory import get_llm


@pytest.mark.parametrize(
    ("provider", "selected_module", "selected_factory"),
    [
        ("google", "providers.llm.google_llm", "get_google_llm"),
        ("groq", "providers.llm.groq_llm", "get_groq_llm"),
        ("openai", "providers.llm.openai_llm", "get_openai_llm"),
    ],
)
def test_factory_constructs_exactly_one_selected_provider(
    monkeypatch,
    provider,
    selected_module,
    selected_factory,
):
    calls = []
    marker = object()

    def factory(name, result=None):
        def build():
            calls.append(name)
            return result

        return build

    modules = {
        "providers.llm.google_llm": SimpleNamespace(
            get_google_llm=factory("google", marker if provider == "google" else None)
        ),
        "providers.llm.groq_llm": SimpleNamespace(
            get_groq_llm=factory("groq", marker if provider == "groq" else None)
        ),
        "providers.llm.openai_llm": SimpleNamespace(
            get_openai_llm=factory("openai", marker if provider == "openai" else None)
        ),
    }
    for module_name, module in modules.items():
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setenv("LLM_PROVIDER", f" {provider.upper()} ")
    monkeypatch.setenv("GROQ_API_KEY", "must-not-create-an-unselected-provider")

    assert get_llm() is marker
    assert calls == [provider]
    assert selected_module in sys.modules
    assert hasattr(sys.modules[selected_module], selected_factory)


def test_factory_rejects_unsupported_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="Expected google, groq, or openai"):
        get_llm()


def test_google_builder_has_no_cross_provider_fallback(monkeypatch):
    from providers.llm import google_llm

    captured = {}

    class FakeGoogleService:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(google_llm, "LatencyBoundGoogleLLMService", FakeGoogleService)
    monkeypatch.setattr(google_llm, "load_system_prompt", lambda: "system prompt")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("GROQ_API_KEY", "must-be-ignored")

    google_llm.get_google_llm()

    assert captured["api_key"] == "google-key"
    assert "fallback_llm" not in captured
    assert "hedge_delay_secs" not in captured


def test_openai_builder_uses_openai_env_and_tool_configuration(monkeypatch):
    from providers.llm import openai_llm

    captured = {}

    class FakeSettings:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeOpenAIService:
        Settings = FakeSettings

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(openai_llm, "LatencyBoundOpenAILLMService", FakeOpenAIService)
    monkeypatch.setattr(openai_llm, "load_system_prompt", lambda: "system prompt")
    monkeypatch.setattr(openai_llm, "tool_timeout_seconds", lambda: 3.5)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "configured-openai-model")

    openai_llm.get_openai_llm()

    assert captured["api_key"] == "openai-key"
    assert captured["settings"].model == "configured-openai-model"
    assert captured["settings"].system_instruction == "system prompt"
    assert captured["function_call_timeout_secs"] == 3.5
    assert captured["enable_async_tool_cancellation"] is True
    assert captured["retry_on_timeout"] is False


def test_groq_builder_does_not_treat_temperature_as_latency_control(monkeypatch):
    from providers.llm import groq_llm

    captured = {}

    class FakeSettings:
        def __init__(self, **kwargs):
            self.values = kwargs

    class FakeGroqService:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(groq_llm, "GroqLLMSettings", FakeSettings)
    monkeypatch.setattr(groq_llm, "LatencyBoundGroqLLMService", FakeGroqService)
    monkeypatch.setattr(groq_llm, "load_system_prompt", lambda: "system prompt")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")

    groq_llm.get_groq_llm()

    assert "temperature" not in captured["settings"].values


def test_groq_builder_sends_completion_controls_through_settings_extra(monkeypatch):
    from providers.llm import groq_llm

    captured = {}

    class FakeSettings:
        def __init__(self, **kwargs):
            self.values = kwargs

    class FakeGroqService:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(groq_llm, "GroqLLMSettings", FakeSettings)
    monkeypatch.setattr(groq_llm, "LatencyBoundGroqLLMService", FakeGroqService)
    monkeypatch.setattr(groq_llm, "load_system_prompt", lambda: "system prompt")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("GROQ_REASONING_EFFORT", "low")
    monkeypatch.setenv("GROQ_INCLUDE_REASONING", "false")
    monkeypatch.setenv("GROQ_PARALLEL_TOOL_CALLS", "false")

    groq_llm.get_groq_llm()

    assert captured["settings"].values["extra"] == {
        "parallel_tool_calls": False,
        "reasoning_effort": "low",
        "include_reasoning": False,
    }
    assert "client_kwargs" not in captured


def test_groq_builder_omits_reasoning_controls_for_non_reasoning_model(monkeypatch):
    from providers.llm import groq_llm

    captured = {}

    class FakeSettings:
        def __init__(self, **kwargs):
            self.values = kwargs

    class FakeGroqService:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(groq_llm, "GroqLLMSettings", FakeSettings)
    monkeypatch.setattr(groq_llm, "LatencyBoundGroqLLMService", FakeGroqService)
    monkeypatch.setattr(groq_llm, "load_system_prompt", lambda: "system prompt")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
    monkeypatch.setenv("GROQ_REASONING_EFFORT", "invalid-for-this-model")

    groq_llm.get_groq_llm()

    assert captured["settings"].values["extra"] == {
        "parallel_tool_calls": False,
    }


def test_groq_converts_dynamic_developer_context_to_user_visible_context():
    from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
    from pipecat.processors.aggregators.llm_context import LLMContext
    from providers.llm.groq_llm import LatencyBoundGroqLLMService

    context = LLMContext(
        messages=[
            {"role": "user", "content": "What is Taylor Swift's latest album?"},
            {
                "role": "developer",
                "content": (
                    "Live web search result for the current user request. "
                    "Use this result directly:\n"
                    '{"answer": "The Tortured Poets Department"}'
                ),
            },
        ]
    )

    params = OpenAILLMAdapter().get_llm_invocation_params(
        context,
        system_instruction="system prompt",
        convert_developer_to_user=not LatencyBoundGroqLLMService.supports_developer_role,
    )

    assert params["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "What is Taylor Swift's latest album?"},
        {
            "role": "user",
            "content": (
                "Live web search result for the current user request. "
                "Use this result directly:\n"
                '{"answer": "The Tortured Poets Department"}'
            ),
        },
    ]


def test_groq_builder_validates_gpt_oss_reasoning_effort(monkeypatch):
    from providers.llm import groq_llm

    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("GROQ_REASONING_EFFORT", "minimal")

    with pytest.raises(ValueError, match="GROQ_REASONING_EFFORT"):
        groq_llm.get_groq_llm()


@pytest.mark.anyio
async def test_groq_connection_warmup_uses_existing_client_and_fails_open():
    from providers.llm.groq_llm import LatencyBoundGroqLLMService

    class Models:
        def __init__(self):
            self.calls = 0

        async def list(self):
            self.calls += 1
            return []

    models = Models()
    service = object.__new__(LatencyBoundGroqLLMService)
    service._client = SimpleNamespace(models=models)
    service._settings = SimpleNamespace(model="llama-3.1-8b-instant")
    service._connection_warmed = False
    service._warmup_attempted = False

    assert await service.warm_connection(timeout_seconds=0.2) is True
    assert await service.warm_connection(timeout_seconds=0.2) is True
    assert models.calls == 1

    async def fail():
        raise OSError("network unavailable")

    cold = object.__new__(LatencyBoundGroqLLMService)
    cold._client = SimpleNamespace(models=SimpleNamespace(list=fail))
    cold._settings = SimpleNamespace(model="llama-3.1-8b-instant")
    cold._connection_warmed = False
    cold._warmup_attempted = False

    assert await cold.warm_connection(timeout_seconds=0.2) is False
    assert cold.connection_warmed is False
