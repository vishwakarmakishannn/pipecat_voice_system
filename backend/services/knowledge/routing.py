"""Deterministic first-stage routing for latency-sensitive voice turns."""

import re

from services.knowledge.types import TurnRoute


_CONVERSATION = re.compile(
    r"^(?:hi|hello|hey|thanks|thank you|okay|ok|yes|no|good morning|good evening|bye)[.! ]*$",
    re.IGNORECASE,
)
_ACTION = re.compile(
    r"\b(?:raise|create|open|log|cancel|close)\s+(?:a\s+)?(?:ticket|complaint|case)\b",
    re.IGNORECASE,
)
_LIVE_LOOKUP = re.compile(
    r"\b(?:my|mine|our)\b.*\b(?:status|settlement|transaction|device|machine|ticket|case|refund|payment)\b|"
    r"\b(?:check|track|verify|lookup|look up)\b.*\b(?:status|transaction|settlement|ticket|device)\b|"
    r"\b(?:check|track|verify)\s+(?:my|mine|ours)\b",
    re.IGNORECASE,
)
_KNOWLEDGE = re.compile(
    r"\b(?:how|what|why|when|where|which|can|does|is|are|meaning|means|fix|solve|"
    r"error|problem|issue|charges|fees|limit|activate|install|connect|settlement|refund|"
    r"mswipe|wise\s*pos|ypos|boombox|sound\s*box|neo\s*2|pos|nfc)\b",
    re.IGNORECASE,
)
_CLARIFICATION = re.compile(
    r"^(?:what|why|how|which|that|this|it|same|again|tell me more)[? ]*$",
    re.IGNORECASE,
)
_HUMAN = re.compile(
    r"\b(?:human|person|agent|supervisor|manager|representative|talk to someone|speak to someone)\b",
    re.IGNORECASE,
)


def route_mswipe_turn(query: str) -> TurnRoute:
    """Classify a completed caller turn without an extra LLM round trip.

    This is intentionally conservative: the serving layer may retrieve on a
    knowledge-like turn, but it never treats static text as authorization for a
    live lookup or state-changing action.
    """
    text = " ".join((query or "").split()).strip()
    if not text:
        return TurnRoute("clarification", 1.0, ("empty_turn",))
    if _HUMAN.search(text):
        return TurnRoute("human_handoff", 0.98, ("explicit_handoff",))
    action = bool(_ACTION.search(text))
    live = bool(_LIVE_LOOKUP.search(text))
    knowledge = bool(_KNOWLEDGE.search(text))
    if action and (live or knowledge):
        return TurnRoute("mixed", 0.92, ("action", "information"), requires_auth=True)
    if action:
        return TurnRoute("action", 0.97, ("state_change",), requires_auth=True)
    if live and knowledge:
        return TurnRoute("mixed", 0.88, ("static_explanation", "account_lookup"), requires_auth=True)
    if live:
        return TurnRoute("live_lookup", 0.94, ("customer_specific",), requires_auth=True)
    if _CONVERSATION.fullmatch(text):
        return TurnRoute("conversation", 0.99, ("social_turn",))
    if _CLARIFICATION.fullmatch(text):
        return TurnRoute("clarification", 0.82, ("underspecified",))
    # This is a dedicated Mswipe support system, so any substantive unknown
    # turn receives safe knowledge retrieval rather than requiring document
    # keywords. Confidence remains lower so no-answer behaviour stays active.
    return TurnRoute(
        "knowledge",
        0.9 if knowledge else 0.64,
        ("domain_question",) if knowledge else ("substantive_support_turn",),
    )
