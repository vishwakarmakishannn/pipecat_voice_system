from types import SimpleNamespace
import asyncio

import pytest
from pipecat.processors.aggregators.llm_context import LLMContext

from core.context_summary import (
    ContextMutationEpoch,
    LIVE_CONVERSATION_SUMMARY_MARKER,
    QUERY_SCOPED_CONTEXT_MARKER,
    SUMMARY_SYSTEM_PROMPT,
    build_assistant_summary_params,
    extract_live_conversation_summary,
)
from core.context_summary_config import VoiceContextSummaryConfig
from providers.llm.context_summary import (
    ContextSummaryCooldownError,
    SafeGroqContextSummaryService,
    StaleContextSummaryError,
    sanitize_summary_messages,
)


def summary_config(**overrides):
    values = {
        "enabled": True,
        "max_tokens": 3000,
        "max_messages": 20,
        "target_tokens": 900,
        "keep_messages": 8,
        "timeout_seconds": 6.0,
        "retry_cooldown_seconds": 30.0,
        "emergency_max_messages": 40,
        "emergency_max_chars": 24000,
        "model": "llama-3.1-8b-instant",
    }
    values.update(overrides)
    return VoiceContextSummaryConfig(**values)


def test_pipecat_summary_params_use_dedicated_llm_and_marker():
    llm = object()
    params = build_assistant_summary_params(summary_config(), llm)
    auto = params.auto_context_summarization_config

    assert params.enable_auto_context_summarization is True
    assert auto.max_context_tokens == 3000
    assert auto.max_unsummarized_messages == 20
    assert auto.summary_config.llm is llm
    assert auto.summary_config.min_messages_after_summary == 8
    assert auto.summary_config.summarization_prompt == SUMMARY_SYSTEM_PROMPT
    assert auto.summary_config.summary_message_template.startswith(
        LIVE_CONVERSATION_SUMMARY_MARKER
    )


def test_dedicated_groq_summary_inherits_reasoning_model_controls(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_REASONING_EFFORT", "low")
    monkeypatch.setenv("GROQ_INCLUDE_REASONING", "false")
    service = SafeGroqContextSummaryService(
        config=summary_config(model="openai/gpt-oss-20b"),
        mutation_epoch_getter=lambda: 0,
    )

    assert service._settings.extra == {
        "parallel_tool_calls": False,
        "reasoning_effort": "low",
        "extra_body": {"include_reasoning": False},
    }


def test_disabled_summary_params_do_not_create_auto_config():
    params = build_assistant_summary_params(
        summary_config(enabled=False), None
    )

    assert params.enable_auto_context_summarization is False
    assert params.auto_context_summarization_config is None


def test_summary_sanitizer_excludes_query_scoped_and_tool_payloads():
    messages = [
        {"role": "user", "content": "What did the report say?"},
        {
            "role": "developer",
            "content": f"{QUERY_SCOPED_CONTEXT_MARKER}\nRAG_GROUNDED_TURN: raw",
        },
        {"role": "tool", "content": '{"large": "search payload"}'},
        {"role": "assistant", "content": "The report says the deadline is Friday."},
    ]

    assert sanitize_summary_messages(messages) == [messages[0], messages[-1]]


def test_summary_sanitizer_unwraps_prior_live_summary_marker():
    assert sanitize_summary_messages(
        [
            {
                "role": "user",
                "content": f"{LIVE_CONVERSATION_SUMMARY_MARKER}\nEarlier context.",
            }
        ]
    ) == [{"role": "user", "content": "Earlier context."}]


def test_extract_applied_summary_ignores_ordinary_messages():
    messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "user",
            "content": f"{LIVE_CONVERSATION_SUMMARY_MARKER}\nThe user is planning a trip.",
        },
    ]

    assert extract_live_conversation_summary(messages) == (
        "The user is planning a trip."
    )


@pytest.mark.anyio
async def test_dedicated_summary_rejects_result_after_destructive_mutation(
    monkeypatch,
):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    epoch = ContextMutationEpoch()
    service = SafeGroqContextSummaryService(
        config=summary_config(), mutation_epoch_getter=lambda: epoch.value
    )

    async def inference(*_args, **_kwargs):
        epoch.bump("test_trim")
        return "summary"

    service.run_inference = inference
    context = LLMContext(
        messages=[
            {"role": "user", "content": f"question {index}"}
            for index in range(12)
        ]
    )
    frame = SimpleNamespace(
        context=context,
        min_messages_to_keep=2,
        target_context_tokens=100,
        summarization_prompt="ignored",
    )

    with pytest.raises(StaleContextSummaryError):
        await service._generate_summary(frame)
    await service.close()


@pytest.mark.anyio
async def test_ordinary_append_does_not_invalidate_summary(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    epoch = ContextMutationEpoch()
    service = SafeGroqContextSummaryService(
        config=summary_config(), mutation_epoch_getter=lambda: epoch.value
    )
    context = LLMContext(
        messages=[
            {"role": "user", "content": f"question {index}"}
            for index in range(12)
        ]
    )

    async def inference(*_args, **_kwargs):
        context.add_message({"role": "assistant", "content": "new append"})
        return "safe summary"

    service.run_inference = inference
    frame = SimpleNamespace(
        context=context,
        min_messages_to_keep=2,
        target_context_tokens=100,
        summarization_prompt="ignored",
    )

    summary, last_index = await service._generate_summary(frame)

    assert summary == "safe summary"
    assert last_index == 9
    assert epoch.value == 0
    await service.close()


@pytest.mark.anyio
async def test_cancelled_provider_attempt_activates_retry_cooldown(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    service = SafeGroqContextSummaryService(
        config=summary_config(), mutation_epoch_getter=lambda: 0
    )
    started = asyncio.Event()

    async def inference(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    service.run_inference = inference
    frame = SimpleNamespace(
        context=LLMContext(
            messages=[
                {"role": "user", "content": f"question {index}"}
                for index in range(12)
            ]
        ),
        min_messages_to_keep=2,
        target_context_tokens=100,
        summarization_prompt="ignored",
    )
    task = asyncio.create_task(service._generate_summary(frame))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with pytest.raises(ContextSummaryCooldownError):
        await service._generate_summary(frame)
    await service.close()
