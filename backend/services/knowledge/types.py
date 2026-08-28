from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID


RouteName = Literal[
    "conversation",
    "knowledge",
    "live_lookup",
    "action",
    "mixed",
    "clarification",
    "human_handoff",
]


@dataclass(frozen=True)
class TurnRoute:
    name: RouteName
    confidence: float
    reasons: tuple[str, ...] = ()
    requires_auth: bool = False


@dataclass(frozen=True)
class KnowledgeHit:
    unit_id: UUID
    stable_key: str
    unit_type: str
    title: str
    answer: str
    voice_answer: str | None
    source_uri: str
    source_label: str
    product: str | None
    topic: str | None
    requires_live_api: bool
    escalation_required: bool
    ticket_candidates: list[dict]
    score: float
    lexical_rank: float | None = None
    vector_similarity: float | None = None
    matched_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class KnowledgeResponse:
    status: Literal["ok", "no_answer", "unavailable"]
    query: str
    normalized_query: str
    route: TurnRoute
    release_id: UUID | None = None
    release_version: str | None = None
    confidence: float = 0.0
    hits: list[KnowledgeHit] = field(default_factory=list)
    reason: str | None = None
