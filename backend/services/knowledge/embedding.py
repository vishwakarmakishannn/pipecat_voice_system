"""Provider-versioned embeddings for the Mswipe corpus."""

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import os
import time

from loguru import logger

from core.knowledge_config import (
    KNOWLEDGE_EMBEDDING_BATCH_SIZE,
    KNOWLEDGE_EMBEDDING_DIMENSION,
    KNOWLEDGE_EMBEDDING_MODEL,
    KNOWLEDGE_EMBEDDING_PROVIDER,
    KNOWLEDGE_EMBEDDING_QUERY_TIMEOUT_SECONDS,
    KNOWLEDGE_EMBEDDING_CIRCUIT_COOLDOWN_SECONDS,
    KNOWLEDGE_EMBEDDING_CIRCUIT_FAILURES,
    KNOWLEDGE_QUERY_CACHE_SIZE,
    KNOWLEDGE_QUERY_CACHE_TTL_SECONDS,
    KNOWLEDGE_QUERY_INFLIGHT_MAX,
)

_google_client = None
_openai_client = None
_query_cache: OrderedDict[tuple[str, str, str], tuple[float, list[float]]] = (
    OrderedDict()
)
_query_inflight: dict[tuple[str, str, str], asyncio.Task] = {}
_query_lock = asyncio.Lock()
_circuit_lock = asyncio.Lock()
_circuit_failures = 0
_circuit_state = "closed"
_circuit_opened_at: float | None = None
_recovery_probe_task: asyncio.Task | None = None


@dataclass(frozen=True)
class QueryEmbeddingResult:
    vector: list[float] | None
    failure_class: str | None
    circuit_state: str
    cache_outcome: str
    duration_ms: float


def embedding_identity() -> tuple[str, str, int]:
    return (
        KNOWLEDGE_EMBEDDING_PROVIDER,
        KNOWLEDGE_EMBEDDING_MODEL,
        KNOWLEDGE_EMBEDDING_DIMENSION,
    )


async def embed_knowledge_texts(values: list[str]) -> list[list[float] | None]:
    """Embed a batch without silently crossing provider/model vector spaces."""
    normalized = [" ".join((value or "").split()) for value in values]
    if KNOWLEDGE_EMBEDDING_PROVIDER == "disabled":
        return [None] * len(normalized)
    nonempty = [(index, value) for index, value in enumerate(normalized) if value]
    output: list[list[float] | None] = [None] * len(values)
    if not nonempty:
        return output

    payload = [value for _, value in nonempty]
    vectors: list[list[float]] = []
    for start in range(0, len(payload), KNOWLEDGE_EMBEDDING_BATCH_SIZE):
        batch = payload[start : start + KNOWLEDGE_EMBEDDING_BATCH_SIZE]
        vectors.extend(await _embed_batch(batch))

    if len(vectors) != len(nonempty):
        raise RuntimeError("Embedding provider returned an incomplete batch")
    for (index, _), vector in zip(nonempty, vectors, strict=True):
        if len(vector) != KNOWLEDGE_EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Embedding dimension {len(vector)} does not match configured "
                f"dimension {KNOWLEDGE_EMBEDDING_DIMENSION}"
            )
        output[index] = list(vector)
    return output


async def _embed_batch(payload: list[str]) -> list[list[float]]:
    if KNOWLEDGE_EMBEDDING_PROVIDER == "google":
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY is required for knowledge embeddings")
        global _google_client
        if _google_client is None:
            from google import genai

            _google_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        from google.genai import types

        response = await _google_client.aio.models.embed_content(
            model=KNOWLEDGE_EMBEDDING_MODEL,
            contents=payload,
            config=types.EmbedContentConfig(
                output_dimensionality=KNOWLEDGE_EMBEDDING_DIMENSION
            ),
        )
        return [list(item.values) for item in response.embeddings]
    elif KNOWLEDGE_EMBEDDING_PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for knowledge embeddings")
        global _openai_client
        if _openai_client is None:
            from openai import AsyncOpenAI

            _openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = await _openai_client.embeddings.create(
            input=payload,
            model=KNOWLEDGE_EMBEDDING_MODEL,
            dimensions=KNOWLEDGE_EMBEDDING_DIMENSION,
        )
        return [
            list(item.embedding)
            for item in sorted(response.data, key=lambda item: item.index)
        ]
    else:  # guarded by configuration validation
        raise RuntimeError("Unsupported knowledge embedding provider")



def _validate_query_vector(vectors: list[list[float]]) -> list[float]:
    if len(vectors) != 1:
        raise RuntimeError("Embedding provider returned an incomplete query result")
    vector = list(vectors[0])
    if len(vector) != KNOWLEDGE_EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Embedding dimension {len(vector)} does not match configured "
            f"dimension {KNOWLEDGE_EMBEDDING_DIMENSION}"
        )
    return vector


async def _run_query_embedding(
    key: tuple[str, str, str],
    normalized: str,
) -> list[float]:
    try:
        vector = _validate_query_vector(await _embed_batch([normalized]))
        if (
            vector
            and KNOWLEDGE_QUERY_CACHE_SIZE > 0
            and KNOWLEDGE_QUERY_CACHE_TTL_SECONDS > 0
        ):
            async with _query_lock:
                _query_cache[key] = (time.monotonic(), list(vector))
                _query_cache.move_to_end(key)
                while len(_query_cache) > KNOWLEDGE_QUERY_CACHE_SIZE:
                    _query_cache.popitem(last=False)
        return vector
    finally:
        async with _query_lock:
            if _query_inflight.get(key) is asyncio.current_task():
                _query_inflight.pop(key, None)


async def _query_vector(value: str) -> tuple[list[float] | None, str]:
    normalized = " ".join((value or "").split())
    if not normalized or KNOWLEDGE_EMBEDDING_PROVIDER == "disabled":
        return None, "disabled"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    key = (KNOWLEDGE_EMBEDDING_PROVIDER, KNOWLEDGE_EMBEDDING_MODEL, digest)
    now = time.monotonic()

    async with _query_lock:
        cached = _query_cache.get(key)
        if cached and now - cached[0] <= KNOWLEDGE_QUERY_CACHE_TTL_SECONDS:
            _query_cache.move_to_end(key)
            return list(cached[1]), "hit"
        if cached:
            _query_cache.pop(key, None)
        task = _query_inflight.get(key)
        cache_outcome = "shared"
        if task is None:
            for inflight_key, inflight_task in list(_query_inflight.items()):
                if inflight_task.done():
                    _query_inflight.pop(inflight_key, None)
            if len(_query_inflight) >= KNOWLEDGE_QUERY_INFLIGHT_MAX:
                raise RuntimeError("Knowledge embedding in-flight capacity exceeded")
            task = asyncio.create_task(_run_query_embedding(key, normalized))
            _query_inflight[key] = task
            cache_outcome = "miss"
    return list(await asyncio.shield(task)), cache_outcome


async def embed_knowledge_text(value: str) -> list[float] | None:
    vector, _cache_outcome = await _query_vector(value)
    return vector


async def _record_circuit_success() -> None:
    global _circuit_failures, _circuit_state, _circuit_opened_at
    async with _circuit_lock:
        previous = _circuit_state
        _circuit_failures = 0
        _circuit_state = "closed"
        _circuit_opened_at = None
    if previous != "closed":
        logger.info("knowledge_embedding circuit_transition={}=>closed", previous)


async def _record_circuit_failure(failure_class: str) -> str:
    global _circuit_failures, _circuit_state, _circuit_opened_at
    async with _circuit_lock:
        _circuit_failures += 1
        if _circuit_failures >= KNOWLEDGE_EMBEDDING_CIRCUIT_FAILURES:
            previous = _circuit_state
            _circuit_state = "open"
            _circuit_opened_at = time.monotonic()
        else:
            previous = _circuit_state
        state = _circuit_state
    if previous != state:
        logger.warning(
            "knowledge_embedding circuit_transition={}=>{} failure_class={}",
            previous,
            state,
            failure_class,
        )
    return state


async def _recovery_probe() -> None:
    global _recovery_probe_task
    try:
        async with asyncio.timeout(KNOWLEDGE_EMBEDDING_QUERY_TIMEOUT_SECONDS):
            vector = _validate_query_vector(
                await _embed_batch(["knowledge embedding recovery probe"])
            )
        if vector:
            await _record_circuit_success()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await _record_circuit_failure(type(exc).__name__)
    finally:
        _recovery_probe_task = None


async def _circuit_allows_request() -> tuple[bool, str]:
    global _circuit_state, _recovery_probe_task
    async with _circuit_lock:
        state = _circuit_state
        opened_at = _circuit_opened_at
        cooldown_elapsed = bool(
            opened_at is not None
            and time.monotonic() - opened_at
            >= KNOWLEDGE_EMBEDDING_CIRCUIT_COOLDOWN_SECONDS
        )
        if state == "open" and cooldown_elapsed:
            _circuit_state = "half_open"
            state = "half_open"
            if _recovery_probe_task is None or _recovery_probe_task.done():
                _recovery_probe_task = asyncio.create_task(_recovery_probe())
        return state == "closed", state


async def query_knowledge_embedding(value: str) -> QueryEmbeddingResult:
    """Return bounded dense retrieval metadata without hiding degraded mode."""
    started = time.monotonic()
    if KNOWLEDGE_EMBEDDING_PROVIDER == "disabled":
        return QueryEmbeddingResult(None, "disabled", "disabled", "disabled", 0.0)
    allowed, state = await _circuit_allows_request()
    if not allowed:
        return QueryEmbeddingResult(
            None,
            "circuit_open",
            state,
            "bypassed",
            round((time.monotonic() - started) * 1000, 1),
        )
    try:
        async with asyncio.timeout(KNOWLEDGE_EMBEDDING_QUERY_TIMEOUT_SECONDS):
            vector, cache_outcome = await _query_vector(value)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        failure_class = "timeout" if isinstance(exc, TimeoutError) else type(exc).__name__
        state = await _record_circuit_failure(failure_class)
        return QueryEmbeddingResult(
            None,
            failure_class,
            state,
            "miss",
            round((time.monotonic() - started) * 1000, 1),
        )
    await _record_circuit_success()
    return QueryEmbeddingResult(
        list(vector) if vector is not None else None,
        None,
        "closed",
        cache_outcome,
        round((time.monotonic() - started) * 1000, 1),
    )


async def warm_knowledge_embedding() -> bool:
    """Warm dense retrieval without making it a voice-service dependency."""
    if KNOWLEDGE_EMBEDDING_PROVIDER == "disabled":
        return False
    try:
        async with asyncio.timeout(KNOWLEDGE_EMBEDDING_QUERY_TIMEOUT_SECONDS):
            vectors = await _embed_batch(["knowledge service readiness"])
        if not vectors or len(vectors[0]) != KNOWLEDGE_EMBEDDING_DIMENSION:
            raise RuntimeError("Knowledge embedding warmup returned an invalid vector")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            "knowledge_embedding status=degraded stage=warmup error_type={}",
            type(exc).__name__,
        )
        return False
    logger.info("knowledge_embedding status=ready stage=warmup")
    return True


def reset_knowledge_embedding_cache_for_tests() -> None:
    global _circuit_failures, _circuit_state, _circuit_opened_at, _recovery_probe_task
    _query_cache.clear()
    _query_inflight.clear()
    _circuit_failures = 0
    _circuit_state = "closed"
    _circuit_opened_at = None
    if _recovery_probe_task and not _recovery_probe_task.done():
        _recovery_probe_task.cancel()
    _recovery_probe_task = None
