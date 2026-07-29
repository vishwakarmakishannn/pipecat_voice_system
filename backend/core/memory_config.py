import os


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def memory_embedding_provider() -> str:
    provider = os.getenv("MEMORY_EMBEDDING_PROVIDER", "disabled").strip().lower()
    if provider not in {"google", "openai", "disabled"}:
        raise ValueError(
            "MEMORY_EMBEDDING_PROVIDER must be google, openai, or disabled; "
            f"got {provider!r}"
        )
    return provider


RECENT_MESSAGE_LIMIT = env_int("RECENT_MESSAGE_LIMIT", 20)
PRIOR_CONVERSATION_MESSAGE_LIMIT = env_int("PRIOR_CONVERSATION_MESSAGE_LIMIT", 10)
SUMMARY_MESSAGE_THRESHOLD = env_int("SUMMARY_MESSAGE_THRESHOLD", 20)
SUMMARY_CHAR_THRESHOLD = env_int("SUMMARY_CHAR_THRESHOLD", 4000)
MEMORY_LLM_TIMEOUT_SECONDS = env_float("MEMORY_LLM_TIMEOUT_SECONDS", 4.0)
MEMORY_FACT_CONFIDENCE_MIN = env_float("MEMORY_FACT_CONFIDENCE_MIN", 0.85)
MEMORY_RECALL_TOP_K = env_int("MEMORY_RECALL_TOP_K", 5)
MEMORY_RECALL_MIN_SCORE = env_float("MEMORY_RECALL_MIN_SCORE", 0.72)
MEMORY_VECTOR_DB = os.getenv("MEMORY_VECTOR_DB", "pgvector").lower()
MEMORY_EMBEDDING_PROVIDER = memory_embedding_provider()
MEMORY_EMBEDDING_CACHE_SIZE = env_int("MEMORY_EMBEDDING_CACHE_SIZE", 256)
MEMORY_EMBEDDING_CACHE_TTL_SECONDS = env_float("MEMORY_EMBEDDING_CACHE_TTL_SECONDS", 300.0)
MEMORY_FACTS_MAX_CHARS = env_int("MEMORY_FACTS_MAX_CHARS", 2000)
MEMORY_SUMMARY_MAX_CHARS = env_int("MEMORY_SUMMARY_MAX_CHARS", 3000)
MEMORY_RECENT_MAX_CHARS = env_int("MEMORY_RECENT_MAX_CHARS", 6000)
MEMORY_PRIOR_MAX_CHARS = env_int("MEMORY_PRIOR_MAX_CHARS", 3000)
MEMORY_PROMPT_MAX_TOKENS = env_int("MEMORY_PROMPT_MAX_TOKENS", 1500)

# Keep the DB schema provider-neutral by storing one fixed vector size.
# Both Google gemini-embedding-001 and OpenAI text-embedding-3-small can be used
# with this dimension; change it once before storing production memory if needed.
MEMORY_EMBEDDING_DIMENSION = env_int("MEMORY_EMBEDDING_DIMENSION", 768)
