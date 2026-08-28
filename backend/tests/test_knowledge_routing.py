import pytest

from services.knowledge.routing import route_mswipe_turn


@pytest.mark.parametrize(
    ("query", "route"),
    [
        ("Hello", "conversation"),
        ("How do I activate my Mswipe POS?", "knowledge"),
        ("What does error 51 mean?", "knowledge"),
        ("Check my transaction status", "live_lookup"),
        ("Raise a complaint ticket", "action"),
        ("How can I fix this and then raise a ticket?", "mixed"),
        ("Why is settlement delayed, and can you check mine?", "mixed"),
        ("I want to speak to a human agent", "human_handoff"),
        ("", "clarification"),
    ],
)
def test_mswipe_turn_routes_before_retrieval(query, route):
    assert route_mswipe_turn(query).name == route


def test_customer_specific_routes_require_authentication():
    assert route_mswipe_turn("Check my settlement status").requires_auth is True
    assert route_mswipe_turn("Raise a ticket").requires_auth is True
    assert route_mswipe_turn("What is settlement?").requires_auth is False
