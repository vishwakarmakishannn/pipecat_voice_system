import os


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except TypeError, ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except TypeError, ValueError:
        return default


RAG_UPLOAD_DIR = os.getenv("RAG_UPLOAD_DIR", "uploads/rag")
RAG_MAX_UPLOAD_MB = _int_env("RAG_MAX_UPLOAD_MB", 25)
RAG_RETRIEVAL_TOP_K = _int_env("RAG_RETRIEVAL_TOP_K", 5)
RAG_VECTOR_CANDIDATES = _int_env("RAG_VECTOR_CANDIDATES", 20)
RAG_TEXT_CANDIDATES = _int_env("RAG_TEXT_CANDIDATES", 20)
RAG_RRF_K = _int_env("RAG_RRF_K", 60)
_rag_reranker_raw = os.getenv("RAG_RERANKER", "lightweight").strip().lower()
if _rag_reranker_raw in {"1", "true", "yes", "on"}:
    RAG_RERANKER = "lightweight"
elif _rag_reranker_raw in {"0", "false", "no", "off"}:
    RAG_RERANKER = "none"
elif _rag_reranker_raw in {"none", "lightweight"}:
    RAG_RERANKER = _rag_reranker_raw
else:
    raise ValueError("RAG_RERANKER must be 'lightweight' or 'none'")
RAG_SMART_ROUTER = os.getenv("RAG_SMART_ROUTER", "explicit").strip().lower()
if RAG_SMART_ROUTER not in {"explicit", "hybrid", "always"}:
    raise ValueError("RAG_SMART_ROUTER must be explicit, hybrid, or always")
RAG_READY_CORPUS_CACHE_TTL_SECONDS = _float_env(
    "RAG_READY_CORPUS_CACHE_TTL_SECONDS", 30.0
)
if not 1.0 <= RAG_READY_CORPUS_CACHE_TTL_SECONDS <= 300.0:
    raise ValueError("RAG_READY_CORPUS_CACHE_TTL_SECONDS must be between 1 and 300")
RAG_MIN_CONTENT_CHARS = _int_env("RAG_MIN_CONTENT_CHARS", 30)
RAG_MAX_CONTEXT_CHARS = _int_env("RAG_MAX_CONTEXT_CHARS", 6000)
RAG_CONTEXT_CHUNK_CHARS = _int_env("RAG_CONTEXT_CHUNK_CHARS", 1200)
RAG_MAX_CONTEXT_TOKENS = _int_env("RAG_MAX_CONTEXT_TOKENS", 1600)
RAG_CONTEXT_CHUNK_TOKENS = _int_env("RAG_CONTEXT_CHUNK_TOKENS", 450)
RAG_VOICE_CONTEXT_MAX_TOKENS = _int_env(
    "RAG_VOICE_CONTEXT_MAX_TOKENS", 420
)
RAG_VOICE_CONTEXT_CHUNK_TOKENS = _int_env(
    "RAG_VOICE_CONTEXT_CHUNK_TOKENS", 240
)
RAG_VOICE_CONTEXT_MAX_CHUNKS = _int_env("RAG_VOICE_CONTEXT_MAX_CHUNKS", 2)
if not 64 <= RAG_VOICE_CONTEXT_MAX_TOKENS <= RAG_MAX_CONTEXT_TOKENS:
    raise ValueError(
        "RAG_VOICE_CONTEXT_MAX_TOKENS must be between 64 and "
        "RAG_MAX_CONTEXT_TOKENS"
    )
if not 32 <= RAG_VOICE_CONTEXT_CHUNK_TOKENS <= RAG_VOICE_CONTEXT_MAX_TOKENS:
    raise ValueError(
        "RAG_VOICE_CONTEXT_CHUNK_TOKENS must be between 32 and "
        "RAG_VOICE_CONTEXT_MAX_TOKENS"
    )
if not 1 <= RAG_VOICE_CONTEXT_MAX_CHUNKS <= 5:
    raise ValueError("RAG_VOICE_CONTEXT_MAX_CHUNKS must be between 1 and 5")
RAG_TEXT_MATCH_MIN_RANK = _float_env("RAG_TEXT_MATCH_MIN_RANK", 0.0)
RAG_MIN_VECTOR_SIMILARITY = _float_env("RAG_MIN_VECTOR_SIMILARITY", 0.62)
RAG_MIN_TEXT_RANK = _float_env("RAG_MIN_TEXT_RANK", 0.15)
RAG_MIN_FINAL_SCORE = _float_env("RAG_MIN_FINAL_SCORE", 0.45)
RAG_RERANK_VECTOR_WEIGHT = _float_env("RAG_RERANK_VECTOR_WEIGHT", 0.55)
RAG_RERANK_TEXT_WEIGHT = _float_env("RAG_RERANK_TEXT_WEIGHT", 0.20)
RAG_RERANK_HEADING_WEIGHT = _float_env("RAG_RERANK_HEADING_WEIGHT", 0.25)
RAG_RERANK_EXACT_METADATA_BOOST = _float_env("RAG_RERANK_EXACT_METADATA_BOOST", 0.30)
RAG_LINK_MAX_DENSE_LINKS = _int_env("RAG_LINK_MAX_DENSE_LINKS", 12)
RAG_VOICE_QUERY_WINDOW_SECONDS = _float_env("RAG_VOICE_QUERY_WINDOW_SECONDS", 8.0)
RAG_VOICE_RETRIEVAL_TIMEOUT_SECONDS = _float_env(
    "RAG_VOICE_RETRIEVAL_TIMEOUT_SECONDS", 4.2
)
RAG_VOICE_MEMORY_TIMEOUT_SECONDS = _float_env("RAG_VOICE_MEMORY_TIMEOUT_SECONDS", 0.35)
RAG_VOICE_RAG_SOFT_TIMEOUT_SECONDS = _float_env(
    "RAG_VOICE_RAG_SOFT_TIMEOUT_SECONDS", 0.9
)
RAG_VOICE_RAG_TIMEOUT_SECONDS = _float_env("RAG_VOICE_RAG_TIMEOUT_SECONDS", 2.5)


def _validate_voice_rag_timeouts(soft_timeout: float, hard_timeout: float) -> None:
    """Require a real recovery window after the spoken slow-path filler."""
    if 0.05 <= soft_timeout < hard_timeout:
        return
    raise ValueError(
        "RAG_VOICE_RAG_SOFT_TIMEOUT_SECONDS must be at least 0.05 and "
        "strictly less than RAG_VOICE_RAG_TIMEOUT_SECONDS"
    )


_validate_voice_rag_timeouts(
    RAG_VOICE_RAG_SOFT_TIMEOUT_SECONDS,
    RAG_VOICE_RAG_TIMEOUT_SECONDS,
)
RAG_VECTOR_FUSION_TIMEOUT_SECONDS = _float_env(
    "RAG_VECTOR_FUSION_TIMEOUT_SECONDS", 0.7
)
if not 0.05 <= RAG_VECTOR_FUSION_TIMEOUT_SECONDS <= RAG_VOICE_RAG_TIMEOUT_SECONDS:
    raise ValueError(
        "RAG_VECTOR_FUSION_TIMEOUT_SECONDS must be between 0.05 and "
        "RAG_VOICE_RAG_TIMEOUT_SECONDS"
    )
RAG_VECTOR_DB_GRACE_SECONDS = _float_env("RAG_VECTOR_DB_GRACE_SECONDS", 0.15)
if not 0.0 <= RAG_VECTOR_DB_GRACE_SECONDS <= 0.5:
    raise ValueError("RAG_VECTOR_DB_GRACE_SECONDS must be between 0 and 0.5")
if (
    RAG_VECTOR_FUSION_TIMEOUT_SECONDS + RAG_VECTOR_DB_GRACE_SECONDS
    > RAG_VOICE_RAG_TIMEOUT_SECONDS
):
    raise ValueError(
        "RAG_VECTOR_FUSION_TIMEOUT_SECONDS plus RAG_VECTOR_DB_GRACE_SECONDS "
        "must not exceed RAG_VOICE_RAG_TIMEOUT_SECONDS"
    )
RAG_FOLLOWUP_FOCUS_MAX_TURNS = _int_env("RAG_FOLLOWUP_FOCUS_MAX_TURNS", 4)
if not 1 <= RAG_FOLLOWUP_FOCUS_MAX_TURNS <= 12:
    raise ValueError("RAG_FOLLOWUP_FOCUS_MAX_TURNS must be between 1 and 12")
RAG_MIN_STRONG_MATCHES = _int_env("RAG_MIN_STRONG_MATCHES", 1)
RAG_LINK_EXTRACTOR = os.getenv("RAG_LINK_EXTRACTOR", "trafilatura").lower()
RAG_LINK_FALLBACK_EXTRACTOR = os.getenv(
    "RAG_LINK_FALLBACK_EXTRACTOR", "trafilatura"
).lower()
RAG_ALLOW_BROWSER_EXTRACTOR = os.getenv(
    "RAG_ALLOW_BROWSER_EXTRACTOR", "false"
).lower() in {"1", "true", "yes", "on"}
RAG_LINK_MAX_BYTES = _int_env("RAG_LINK_MAX_BYTES", 5_000_000)
RAG_LINK_TIMEOUT_SECONDS = _float_env("RAG_LINK_TIMEOUT_SECONDS", 20.0)
RAG_LINK_RESPECT_ROBOTS = os.getenv("RAG_LINK_RESPECT_ROBOTS", "true").lower() not in {
    "0",
    "false",
    "no",
}
RAG_LINK_MIN_CHARS = _int_env("RAG_LINK_MIN_CHARS", 300)
RAG_LINK_CHUNK_CHARS = _int_env("RAG_LINK_CHUNK_CHARS", 1600)
RAG_LINK_CHUNK_OVERLAP = _int_env("RAG_LINK_CHUNK_OVERLAP", 200)
RAG_LINK_CHUNK_TOKENS = _int_env("RAG_LINK_CHUNK_TOKENS", 400)
RAG_LINK_CHUNK_OVERLAP_TOKENS = _int_env("RAG_LINK_CHUNK_OVERLAP_TOKENS", 40)
RAG_DOCUMENT_CHUNK_TOKENS = _int_env("RAG_DOCUMENT_CHUNK_TOKENS", 400)
RAG_DOCUMENT_CONTEXT_RESERVE_TOKENS = _int_env(
    "RAG_DOCUMENT_CONTEXT_RESERVE_TOKENS", 64
)
RAG_DOCUMENT_CHUNK_TOKENIZER = os.getenv(
    "RAG_DOCUMENT_CHUNK_TOKENIZER",
    "sentence-transformers/all-MiniLM-L6-v2",
)
RAG_LINK_MIN_QUALITY_SCORE = _float_env("RAG_LINK_MIN_QUALITY_SCORE", 0.55)
RAG_LINK_USER_AGENT = os.getenv("RAG_LINK_USER_AGENT", "AuraVoiceRAG/1.0")
RAG_INGEST_EMBED_CONCURRENCY = _int_env("RAG_INGEST_EMBED_CONCURRENCY", 4)
