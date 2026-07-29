from pipecat.processors.aggregators.llm_context import LLMContext

from core.processors import BoundedContextProcessor


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
