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
    case = next(
        item for item in CASES
        if item.name == "mswipe_followup_builds_contextual_query"
    )
    passed, result = evaluate_message(
        case,
        _message(
            tool="search_mswipe_knowledge",
            arguments='{"query":"How to install Mswipe Soundbox"}',
        ),
    )

    assert passed is True
    assert result["reasons"] == []


def test_model_gate_rejects_simulated_tool_markup():
    case = next(item for item in CASES if item.name == "mswipe_product_knowledge")
    passed, result = evaluate_message(
        case,
        _message(
            content=(
                '<function=search_mswipe_knowledge>'
                '{"query":"Mswipe Soundbox"}</function>'
            )
        ),
    )

    assert passed is False
    assert any("simulated tool markup" in reason for reason in result["reasons"])
