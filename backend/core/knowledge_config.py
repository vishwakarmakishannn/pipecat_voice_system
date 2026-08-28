"""Validated configuration for the production Mswipe knowledge system.

The knowledge subsystem is intentionally configured independently from the
legacy, per-user RAG implementation.  It is disabled by default so a migrated
database cannot change live call behaviour until an approved release exists.
"""

import os

from core.memory_config import MEMORY_EMBEDDING_DIMENSION


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean; got {value!r}")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


KNOWLEDGE_ENABLED = _bool_env("MSWIPE_KNOWLEDGE_ENABLED", False)
KNOWLEDGE_STORAGE_DIR = os.getenv(
    "MSWIPE_KNOWLEDGE_STORAGE_DIR", "uploads/mswipe-knowledge"
)
KNOWLEDGE_ALLOWED_DOMAINS = tuple(
    domain.strip().lower()
    for domain in os.getenv("MSWIPE_KNOWLEDGE_ALLOWED_DOMAINS", "mswipe.com,www.mswipe.com").split(",")
    if domain.strip()
)
KNOWLEDGE_ADMIN_USER_IDS = frozenset(
    int(value.strip())
    for value in os.getenv("MSWIPE_KNOWLEDGE_ADMIN_USER_IDS", "").split(",")
    if value.strip()
)
KNOWLEDGE_USER_AGENT = os.getenv(
    "MSWIPE_KNOWLEDGE_USER_AGENT", "MswipeKnowledgeBot/1.0"
).strip()
KNOWLEDGE_RESPECT_ROBOTS = _bool_env("MSWIPE_KNOWLEDGE_RESPECT_ROBOTS", True)
KNOWLEDGE_FETCH_TIMEOUT_SECONDS = _float_env(
    "MSWIPE_KNOWLEDGE_FETCH_TIMEOUT_SECONDS", 15.0
)
KNOWLEDGE_MAX_RESPONSE_BYTES = _int_env(
    "MSWIPE_KNOWLEDGE_MAX_RESPONSE_BYTES", 5_000_000
)
KNOWLEDGE_MAX_CRAWL_PAGES = _int_env("MSWIPE_KNOWLEDGE_MAX_CRAWL_PAGES", 500)
KNOWLEDGE_MAX_CRAWL_DEPTH = _int_env("MSWIPE_KNOWLEDGE_MAX_CRAWL_DEPTH", 6)
KNOWLEDGE_WORKER_POLL_SECONDS = _float_env(
    "MSWIPE_KNOWLEDGE_WORKER_POLL_SECONDS", 1.0
)
KNOWLEDGE_WORKER_STALE_SECONDS = _float_env(
    "MSWIPE_KNOWLEDGE_WORKER_STALE_SECONDS", 1800.0
)
KNOWLEDGE_CRAWL_DELAY_SECONDS = _float_env(
    "MSWIPE_KNOWLEDGE_CRAWL_DELAY_SECONDS", 0.25
)
KNOWLEDGE_TOP_K = _int_env("MSWIPE_KNOWLEDGE_TOP_K", 4)
KNOWLEDGE_TEXT_CANDIDATES = _int_env("MSWIPE_KNOWLEDGE_TEXT_CANDIDATES", 30)
KNOWLEDGE_VECTOR_CANDIDATES = _int_env("MSWIPE_KNOWLEDGE_VECTOR_CANDIDATES", 30)
KNOWLEDGE_RRF_K = _int_env("MSWIPE_KNOWLEDGE_RRF_K", 60)
KNOWLEDGE_MIN_CONFIDENCE = _float_env("MSWIPE_KNOWLEDGE_MIN_CONFIDENCE", 0.42)
KNOWLEDGE_VOICE_TIMEOUT_SECONDS = _float_env(
    "MSWIPE_KNOWLEDGE_VOICE_TIMEOUT_SECONDS", 0.8
)
KNOWLEDGE_VOICE_CONTEXT_MAX_CHARS = _int_env(
    "MSWIPE_KNOWLEDGE_VOICE_CONTEXT_MAX_CHARS", 2400
)
KNOWLEDGE_VOICE_UNIT_MAX_CHARS = _int_env(
    "MSWIPE_KNOWLEDGE_VOICE_UNIT_MAX_CHARS", 900
)
KNOWLEDGE_EMBEDDING_DIMENSION = _int_env(
    "MSWIPE_KNOWLEDGE_EMBEDDING_DIMENSION", MEMORY_EMBEDDING_DIMENSION
)
KNOWLEDGE_EMBEDDING_PROVIDER = os.getenv(
    "MSWIPE_KNOWLEDGE_EMBEDDING_PROVIDER",
    os.getenv("MEMORY_EMBEDDING_PROVIDER", "disabled"),
).strip().lower()
if KNOWLEDGE_EMBEDDING_PROVIDER not in {"google", "openai", "disabled"}:
    raise ValueError(
        "MSWIPE_KNOWLEDGE_EMBEDDING_PROVIDER must be google, openai, or disabled"
    )
_default_embedding_model = {
    "google": "gemini-embedding-001",
    "openai": "text-embedding-3-small",
    "disabled": "none",
}[KNOWLEDGE_EMBEDDING_PROVIDER]
KNOWLEDGE_EMBEDDING_MODEL = os.getenv(
    "MSWIPE_KNOWLEDGE_EMBEDDING_MODEL", _default_embedding_model
).strip()
KNOWLEDGE_EMBEDDING_BATCH_SIZE = _int_env(
    "MSWIPE_KNOWLEDGE_EMBEDDING_BATCH_SIZE", 64
)

if not KNOWLEDGE_ALLOWED_DOMAINS:
    raise ValueError("MSWIPE_KNOWLEDGE_ALLOWED_DOMAINS must not be empty")
if not 1 <= KNOWLEDGE_MAX_CRAWL_PAGES <= 10_000:
    raise ValueError("MSWIPE_KNOWLEDGE_MAX_CRAWL_PAGES must be between 1 and 10000")
if not 0 <= KNOWLEDGE_MAX_CRAWL_DEPTH <= 20:
    raise ValueError("MSWIPE_KNOWLEDGE_MAX_CRAWL_DEPTH must be between 0 and 20")
if not 100_000 <= KNOWLEDGE_MAX_RESPONSE_BYTES <= 50_000_000:
    raise ValueError("MSWIPE_KNOWLEDGE_MAX_RESPONSE_BYTES is outside the safe range")
if not 0.1 <= KNOWLEDGE_FETCH_TIMEOUT_SECONDS <= 60:
    raise ValueError("MSWIPE_KNOWLEDGE_FETCH_TIMEOUT_SECONDS must be between 0.1 and 60")
if not 1 <= KNOWLEDGE_TOP_K <= 10:
    raise ValueError("MSWIPE_KNOWLEDGE_TOP_K must be between 1 and 10")
if not 0 <= KNOWLEDGE_MIN_CONFIDENCE <= 1:
    raise ValueError("MSWIPE_KNOWLEDGE_MIN_CONFIDENCE must be between 0 and 1")
if not 0.1 <= KNOWLEDGE_VOICE_TIMEOUT_SECONDS <= 3:
    raise ValueError("MSWIPE_KNOWLEDGE_VOICE_TIMEOUT_SECONDS must be between 0.1 and 3")
if KNOWLEDGE_EMBEDDING_DIMENSION != 768:
    raise ValueError(
        "MSWIPE_KNOWLEDGE_EMBEDDING_DIMENSION must remain 768 for schema v1; "
        "a dimension change requires a migration and a new release"
    )
if not 1 <= KNOWLEDGE_EMBEDDING_BATCH_SIZE <= 256:
    raise ValueError("MSWIPE_KNOWLEDGE_EMBEDDING_BATCH_SIZE must be between 1 and 256")
if not 0 <= KNOWLEDGE_CRAWL_DELAY_SECONDS <= 10:
    raise ValueError("MSWIPE_KNOWLEDGE_CRAWL_DELAY_SECONDS must be between 0 and 10")
