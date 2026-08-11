from pipecat.processors.aggregators.llm_context import LLMContext

from core.processors import BoundedContextProcessor, immutable_context_messages
from core.context_summary import ContextMutationEpoch


def test_context_window_preserves_prefix_and_complete_recent_turns():
    prefix = {"role": "developer", "content": "stable memory"}
    messages = [
        prefix,
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
        {"role": "tool", "content": "recent tool result"},
    ]
    context = LLMContext(messages=messages)
    processor = BoundedContextProcessor(
        context,
        protected_messages=[prefix],
        max_messages=4,
        max_chars=1000,
    )

    assert processor.trim() == 2
    assert context.messages == [prefix, *messages[-3:]]


def test_context_window_always_keeps_latest_turn():
    latest = {"role": "user", "content": "x" * 2000}
    context = LLMContext(messages=[{"role": "user", "content": "old"}, latest])
    processor = BoundedContextProcessor(context, max_messages=2, max_chars=1000)

    processor.trim()

    assert context.messages == [latest]


def test_only_instruction_prefix_is_protected_from_memory_history():
    developer = {"role": "developer", "content": "durable user facts"}
    old_user = {"role": "user", "content": "old question"}
    old_assistant = {"role": "assistant", "content": "old answer"}
    latest_user = {"role": "user", "content": "latest question"}
    latest_assistant = {"role": "assistant", "content": "latest answer"}
    messages = [developer, old_user, old_assistant, latest_user, latest_assistant]
    context = LLMContext(messages=messages)
    processor = BoundedContextProcessor(
        context,
        protected_messages=immutable_context_messages(messages),
        max_messages=3,
        max_chars=1000,
    )

    assert processor.trim() == 2
    assert context.messages == [developer, latest_user, latest_assistant]


def test_destructive_emergency_trim_increments_mutation_epoch():
    epoch = ContextMutationEpoch()
    context = LLMContext(
        messages=[
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "latest"},
        ]
    )
    processor = BoundedContextProcessor(
        context,
        max_messages=2,
        max_chars=1000,
        mutation_epoch=epoch,
        trim_status="emergency_trimmed",
    )

    assert processor.trim() == 2
    assert epoch.value == 1


def test_noop_trim_does_not_increment_mutation_epoch():
    epoch = ContextMutationEpoch()
    context = LLMContext(messages=[{"role": "user", "content": "latest"}])
    processor = BoundedContextProcessor(
        context, max_messages=40, max_chars=24000, mutation_epoch=epoch
    )

    assert processor.trim() == 0
    assert epoch.value == 0
