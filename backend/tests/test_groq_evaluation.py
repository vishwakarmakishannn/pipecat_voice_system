from types import SimpleNamespace

from scripts.evaluate_groq_voice_model import CASES, evaluate_message


def _message(*, content="", tool=None, arguments="{}"):
    tool_calls = []
    if tool:
        tool_calls.append(
            SimpleNamespace(
                function=SimpleNamespace(name=tool, arguments=arguments)
            )
        )
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def test_model_gate_accepts_contextual_native_tool_query():
    case = next(item for item in CASES if item.name == "correction_builds_contextual_query")
    passed, result = evaluate_message(
        case,
        _message(
            tool="tavily_search",
            arguments='{"query":"Samsung Galaxy A30s camera specifications"}',
        ),
    )

    assert passed is True
    assert result["reasons"] == []


def test_model_gate_rejects_simulated_tool_markup():
    case = next(item for item in CASES if item.name == "current_search")
    passed, result = evaluate_message(
        case,
        _message(
            content='<function=tavily_search>{"query":"Dell G15 price"}</function>'
        ),
    )

    assert passed is False
    assert any("simulated tool markup" in reason for reason in result["reasons"])
