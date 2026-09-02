import asyncio
import json
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import jwt
from loguru import logger
from sqlalchemy import and_, func, or_, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from api.auth import ALGORITHM, SECRET_KEY
from core.database import AsyncSessionLocal, VoiceSessionLocal
from core.memory_config import (
    MEMORY_EMBEDDING_BATCH_SIZE,
    MEMORY_EMBEDDING_DIMENSION,
    MEMORY_EMBEDDING_CACHE_SIZE,
    MEMORY_EMBEDDING_CACHE_TTL_SECONDS,
    MEMORY_EMBEDDING_RETRY_ATTEMPTS,
    MEMORY_FACTS_MAX_CHARS,
    MEMORY_SUMMARY_MAX_CHARS,
    MEMORY_RECENT_MAX_CHARS,
    MEMORY_PRIOR_MAX_CHARS,
    MEMORY_PROMPT_MAX_TOKENS,
    MEMORY_EMBEDDING_PROVIDER,
    MEMORY_FACT_CONFIDENCE_MIN,
    MEMORY_LLM_TIMEOUT_SECONDS,
    MEMORY_RECALL_MIN_SCORE,
    MEMORY_RECALL_TOP_K,
    MEMORY_VECTOR_DB,
    memory_embedding_provider,
    memory_llm_provider,
    PRIOR_CONVERSATION_MESSAGE_LIMIT,
    RECENT_MESSAGE_LIMIT,
)
from core.assistant_output import (
    contains_reserved_tool_markup,
    is_memory_safe_assistant_text,
)
from core.models import Call, MemoryChunk, TranscriptEntry, User, UserMemory
from core.prompt_config import load_memory_prompt


SINGLE_VALUE_KEYS = {
    "real_name",
    "preferred_name",
    "location",
    "role",
    "preferred_language",
}
MULTI_VALUE_KEYS = {"likes", "dislikes", "interests", "goals"}
VALID_DURABILITY = {"stable", "temporary"}
VALID_STATUSES = {"active", "inactive"}
INVALID_NAME_VALUES = {
    "a",
    "an",
    "the",
    "not",
    "fine",
    "good",
    "great",
    "ok",
    "okay",
    "well",
    "from",
    "going",
    "working",
}
_memory_llm_backoff_until = 0.0
_google_client = None
_openai_client = None
_groq_client = None
_embedding_cache: OrderedDict[tuple[str, str, str, int], tuple[float, list[float]]] = (
    OrderedDict()
)
_embedding_inflight: dict[tuple[str, str, str, int], asyncio.Task] = {}
_embedding_lock = asyncio.Lock()


def _get_google_client():
    global _google_client
    if _google_client is None:
        from google import genai

        _google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _google_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI

        _openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from openai import AsyncOpenAI

        _groq_client = AsyncOpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
    return _groq_client


def _get_local_memory_runtime():
    """Reuse the process-wide llama.cpp client warmed during app startup."""
    from providers.local.llm.runtime import get_local_llm_runtime

    return get_local_llm_runtime()


def is_memory_fact_candidate(text_value: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text_value or "").strip().lower())
    if not normalized or normalized.endswith("?"):
        return False
    patterns = (
        r"\b(my name is|call me|i am from|i live in|i work as|i work at)\b",
        r"\b(i like|i love|i prefer|i dislike|i hate|my goal is|i want to)\b",
        r"\b(my preferred language is|i speak)\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


@dataclass
class MemoryBundle:
    user: User
    call: Call
    facts: list[UserMemory]
    prior_call: Call | None = None
    prior_recent_transcripts: list[TranscriptEntry] | None = None


def normalize_runner_body(body: Any) -> dict[str, Any]:
    return body if isinstance(body, dict) else {}


async def authenticate_token(token: str | None, db: AsyncSession) -> User | None:
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except Exception as exc:
        logger.warning(f"Memory auth failed: {exc}")
        return None

    if not username:
        return None

    result = await db.execute(select(User).where(User.username == username))
    return result.scalars().first()


async def _load_recent_transcripts(
    db: AsyncSession,
    call_id,
    limit: int,
) -> list[TranscriptEntry]:
    result = await db.execute(
        select(TranscriptEntry)
        .where(
            TranscriptEntry.call_id == call_id,
            TranscriptEntry.speaker.in_(["You", "Aura"]),
        )
        .order_by(TranscriptEntry.created_at.desc(), TranscriptEntry.id.desc())
        .limit(max(limit * 3, limit))
    )
    entries = [
        entry
        for entry in result.scalars().all()
        if entry.speaker == "You"
        or is_memory_safe_assistant_text(entry.text, entry.source)
    ]
    return list(reversed(entries[:limit]))


async def _load_most_recent_prior_call(
    db: AsyncSession,
    user_id: int,
    current_call_id,
) -> Call | None:
    result = await db.execute(
        select(Call)
        .where(
            Call.user_id == user_id,
            Call.id != current_call_id,
            Call.deleted_at.is_(None),
            Call.status.in_(("completed", "failed", "cancelled", "abandoned")),
        )
        .order_by(Call.updated_at.desc(), Call.created_at.desc())
        .limit(1)
    )
    return result.scalars().first()


async def _load_active_facts(user_id: int) -> list[UserMemory]:
    async with VoiceSessionLocal() as db:
        result = await db.execute(
            select(UserMemory)
            .outerjoin(
                TranscriptEntry,
                TranscriptEntry.id == UserMemory.source_transcript_id,
            )
            .outerjoin(Call, Call.id == TranscriptEntry.call_id)
            .where(
                UserMemory.user_id == user_id,
                UserMemory.status == "active",
                or_(
                    UserMemory.source_transcript_id.is_(None),
                    and_(
                        Call.id.is_not(None),
                        Call.deleted_at.is_(None),
                        Call.status.in_(
                            ("completed", "failed", "cancelled", "abandoned")
                        ),
                    ),
                ),
            )
            .order_by(
                UserMemory.fact_type.asc(),
                UserMemory.key.asc(),
                UserMemory.updated_at.desc(),
            )
        )
        return list(result.scalars().all())


async def load_session_bundle(body: Any) -> MemoryBundle | None:
    """Authenticate and create one new immutable call for this voice session."""
    request_body = normalize_runner_body(body)
    token = request_body.get("token")
    if not token:
        return None

    async with VoiceSessionLocal() as db:
        user = await authenticate_token(token, db)
        if not user:
            logger.warning("Call startup rejected: invalid token")
            return None
        call = Call(
            user_id=user.id,
            title="New call",
            runner_session_id=str(request_body.get("_runner_session_id") or "") or None,
            transport=str(request_body.get("_transport") or "webrtc")[:32],
            direction="web",
            status="initializing",
        )
        db.add(call)
        await db.commit()
        await db.refresh(call)

    return MemoryBundle(
        user=user,
        call=call,
        facts=[],
        prior_call=None,
        prior_recent_transcripts=None,
    )


async def hydrate_memory_bundle(
    bundle: MemoryBundle,
    recent_limit: int = RECENT_MESSAGE_LIMIT,
) -> MemoryBundle:
    del recent_limit
    facts = await _load_active_facts(bundle.user.id)
    return MemoryBundle(
        user=bundle.user,
        call=bundle.call,
        facts=facts,
        prior_call=None,
        prior_recent_transcripts=None,
    )


def transcript_to_llm(entry: TranscriptEntry) -> dict[str, str] | None:
    role_map = {"You": "user", "Aura": "assistant"}
    role = role_map.get(entry.speaker)
    if not role:
        return None
    if entry.speaker == "Aura" and not is_memory_safe_assistant_text(
        entry.text, entry.source
    ):
        return None
    return {"role": role, "content": entry.text}


def _safe_prior_summary(summary: str | None) -> str:
    value = (summary or "").strip()
    return "" if contains_reserved_tool_markup(value) else value


def _clean_fact_value(value: str) -> str:
    value = re.split(r"[.!?\n]", value.strip(), maxsplit=1)[0]
    value = re.sub(r"\s+", " ", value).strip(" ,;:\"'")
    words = value.split()
    return " ".join(words[:16])


def _normalize_key(key: str) -> str:
    key = re.sub(r"[^a-z0-9_]+", "_", key.lower().strip())
    key = re.sub(r"_+", "_", key).strip("_")
    return key[:64]


def _normalize_value(value: str, key: str) -> str:
    value = _clean_fact_value(value)
    if key in {"real_name", "preferred_name"} and value:
        return value.split()[0].capitalize()
    return value


def is_valid_memory_fact(fact: UserMemory) -> bool:
    if fact.status != "active" or not fact.value:
        return False
    if fact.key in {"real_name", "preferred_name", "name"}:
        return fact.value.strip().lower() not in INVALID_NAME_VALUES
    return True


def _format_facts(facts: list[UserMemory]) -> str:
    lines = []
    for fact in facts:
        if not is_valid_memory_fact(fact):
            continue
        label = fact.key
        if fact.fact_type and fact.fact_type != "profile":
            label = f"{fact.fact_type}.{fact.key}"
        lines.append(f"- {label}: {fact.value}")
    return "\n".join(lines)[:MEMORY_FACTS_MAX_CHARS]


def _recent_llm_messages(messages: list[TranscriptEntry]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    total_chars = 0
    for message in reversed(messages):
        llm_message = transcript_to_llm(message)
        if llm_message is None:
            continue
        content = llm_message["content"]
        remaining = MEMORY_RECENT_MAX_CHARS - total_chars
        if remaining <= 0:
            break
        selected.append({**llm_message, "content": content[:remaining]})
        total_chars += min(len(content), remaining)
    return list(reversed(selected))


def _budget_memory_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Deduplicate and fit memory under one approximate token budget."""
    max_chars = max(4, MEMORY_PROMPT_MAX_TOKENS * 4)
    developers = [message for message in messages if message.get("role") == "developer"]
    conversation = [
        message for message in messages if message.get("role") != "developer"
    ]
    selected_developers: list[dict[str, str]] = []
    seen: set[str] = set()
    used = 0

    for message in developers:
        content = message.get("content", "")
        normalized = re.sub(r"\s+", " ", content).strip().lower()
        if not normalized or normalized in seen or used >= max_chars:
            continue
        remaining = max_chars - used
        selected = {**message, "content": content[:remaining]}
        selected_developers.append(selected)
        used += len(selected["content"])
        seen.add(normalized)

    developer_text = " ".join(
        re.sub(r"\s+", " ", message["content"]).lower()
        for message in selected_developers
    )
    selected_conversation: list[dict[str, str]] = []
    for message in reversed(conversation):
        content = message.get("content", "")
        normalized = re.sub(r"\s+", " ", content).strip().lower()
        if not normalized or normalized in seen:
            continue
        if len(normalized) >= 24 and normalized in developer_text:
            continue
        remaining = max_chars - used
        if remaining <= 0:
            break
        selected = {**message, "content": content[:remaining]}
        selected_conversation.append(selected)
        used += len(selected["content"])
        seen.add(normalized)

    return selected_developers + list(reversed(selected_conversation))


def build_memory_messages(bundle: MemoryBundle | None) -> list[dict[str, str]]:
    """Build the mutable context seed for one fresh call."""
    return build_live_context_messages(bundle)


def build_session_memory_context(bundle: MemoryBundle | None) -> str:
    """Return stable authenticated memory for the durable system instruction."""
    if not bundle:
        return ""
    sections: list[str] = []
    facts = _format_facts(bundle.facts)
    if facts:
        sections.append(f"Stable user facts:\n{facts}")
    return "\n\n".join(sections)


def build_live_context_messages(bundle: MemoryBundle | None) -> list[dict[str, str]]:
    """A new call never inherits another call's mutable transcript."""
    return []


def _extract_json_object(text_value: str) -> dict[str, Any]:
    if not text_value:
        return {}
    cleaned = text_value.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


async def _generate_text_with_memory_llm(prompt: str) -> str | None:
    global _memory_llm_backoff_until
    if asyncio.get_running_loop().time() < _memory_llm_backoff_until:
        return None
    provider = memory_llm_provider()
    if provider == "disabled":
        return None

    async def generate_google() -> str | None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None
        client = _get_google_client()
        response = await client.aio.models.generate_content(
            model=os.getenv(
                "GOOGLE_MEMORY_MODEL", os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
            ),
            contents=prompt,
        )
        return getattr(response, "text", None)

    async def generate_openai() -> str | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        client = _get_openai_client()
        response = await client.chat.completions.create(
            model=os.getenv(
                "OPENAI_MEMORY_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def generate_groq() -> str | None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        client = _get_groq_client()
        response = await client.chat.completions.create(
            model=os.getenv(
                "GROQ_MEMORY_MODEL", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def generate_local() -> str | None:
        runtime = _get_local_memory_runtime()
        config = runtime.config
        response = await runtime.client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=0.0,
            max_tokens=config.max_tokens,
            extra_body=config.extra_body,
        )
        return response.choices[0].message.content

    generator = {
        "google": generate_google,
        "groq": generate_groq,
        "local": generate_local,
        "openai": generate_openai,
    }.get(provider)
    if generator is None:
        logger.error("Unsupported memory LLM provider: {}", provider)
        return None

    try:
        return await asyncio.wait_for(generator(), timeout=MEMORY_LLM_TIMEOUT_SECONDS)
    except Exception as exc:
        logger.warning("Memory {} LLM call failed: {}", provider, exc)
        if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
            _memory_llm_backoff_until = asyncio.get_running_loop().time() + 60.0
        return None


def _embedding_model(provider: str) -> str:
    if provider == "google":
        return os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")
    if provider == "openai":
        return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    raise ValueError(f"Unsupported memory embedding provider: {provider!r}")


def _normalize_embedding(embedding: list[float] | None, provider: str) -> list[float]:
    if not embedding:
        raise RuntimeError(f"{provider} returned an empty embedding")
    if len(embedding) != MEMORY_EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"{provider} embedding dimension {len(embedding)} does not match "
            f"MEMORY_EMBEDDING_DIMENSION={MEMORY_EMBEDDING_DIMENSION}"
        )
    return list(embedding)


async def _embed_batch_uncached(values: list[str], provider: str) -> list[list[float]]:
    """Embed one batch in a single provider request, preserving input order."""
    if not values:
        return []
    model = _embedding_model(provider)

    if provider == "google":
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY is required for Google embeddings")
        from google.genai import types

        response = await _get_google_client().aio.models.embed_content(
            model=model,
            contents=values,
            config=types.EmbedContentConfig(
                output_dimensionality=MEMORY_EMBEDDING_DIMENSION
            ),
        )
        embeddings = [item.values for item in response.embeddings]
    elif provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings")
        response = await _get_openai_client().embeddings.create(
            input=values,
            model=model,
            dimensions=MEMORY_EMBEDDING_DIMENSION,
        )
        embeddings = [
            item.embedding
            for item in sorted(response.data, key=lambda item: item.index)
        ]
    else:
        raise ValueError(f"Unsupported memory embedding provider: {provider!r}")

    if len(embeddings) != len(values):
        raise RuntimeError(
            f"{provider} returned {len(embeddings)} embeddings for {len(values)} inputs"
        )
    return [_normalize_embedding(embedding, provider) for embedding in embeddings]


def _embedding_retry_delay(exc: Exception, attempt: int) -> float | None:
    message = str(exc)
    retryable = any(
        marker in message.lower()
        for marker in (
            "429",
            "resource_exhausted",
            "rate limit",
            "timeout",
            "timed out",
            "500",
            "502",
            "503",
            "504",
            "unavailable",
            "connection",
        )
    )
    if not retryable:
        return None
    matches = re.findall(
        r"(?:retry(?:delay)?[^0-9]{0,20})([0-9]+(?:\.[0-9]+)?)\s*s",
        message,
        flags=re.IGNORECASE,
    )
    requested = max((float(value) for value in matches), default=0.0)
    return min(60.0, max(requested, float(2 ** max(0, attempt - 1))))


async def _embed_uncached(value: str, provider: str) -> list[float] | None:
    try:
        result = await asyncio.wait_for(
            _embed_batch_uncached([value], provider),
            timeout=MEMORY_LLM_TIMEOUT_SECONDS * 5,
        )
        return result[0]
    except Exception as exc:
        # Never cross-fallback to another provider: embeddings from different
        # providers/models do not share a compatible vector space.
        logger.warning("{} embedding call failed: {}", provider, exc)
        return None


def _cache_embedding(key: tuple[str, str, str, int], embedding: list[float]) -> None:
    _embedding_cache[key] = (time.monotonic(), list(embedding))
    _embedding_cache.move_to_end(key)
    while len(_embedding_cache) > max(1, MEMORY_EMBEDDING_CACHE_SIZE):
        _embedding_cache.popitem(last=False)


async def embed_texts(
    values: list[str],
    *,
    require_all: bool = False,
) -> list[list[float] | None]:
    """Embed many texts efficiently, optionally enforcing atomic completeness."""
    normalized = [re.sub(r"\s+", " ", (value or "").strip()) for value in values]
    results: list[list[float] | None] = [None] * len(normalized)
    if not normalized or MEMORY_VECTOR_DB != "pgvector":
        return results

    provider = memory_embedding_provider()
    if provider == "disabled":
        return results
    model = _embedding_model(provider)
    positions: dict[str, list[int]] = {}
    for index, value in enumerate(normalized):
        if value:
            positions.setdefault(value, []).append(index)

    missing: list[str] = []
    now = time.monotonic()
    async with _embedding_lock:
        for value, indexes in positions.items():
            key = (provider, model, value, MEMORY_EMBEDDING_DIMENSION)
            cached = _embedding_cache.get(key)
            if cached and now - cached[0] <= MEMORY_EMBEDDING_CACHE_TTL_SECONDS:
                _embedding_cache.move_to_end(key)
                for index in indexes:
                    results[index] = list(cached[1])
            else:
                if cached:
                    _embedding_cache.pop(key, None)
                missing.append(value)

    completed: dict[str, list[float]] = {}
    batch_size = max(1, MEMORY_EMBEDDING_BATCH_SIZE)
    try:
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            last_error: Exception | None = None
            for attempt in range(1, max(1, MEMORY_EMBEDDING_RETRY_ATTEMPTS) + 1):
                try:
                    embedded = await asyncio.wait_for(
                        _embed_batch_uncached(batch, provider),
                        timeout=MEMORY_LLM_TIMEOUT_SECONDS * 5,
                    )
                    completed.update(zip(batch, embedded, strict=True))
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    delay = _embedding_retry_delay(exc, attempt)
                    if delay is None or attempt >= max(
                        1, MEMORY_EMBEDDING_RETRY_ATTEMPTS
                    ):
                        break
                    logger.warning(
                        "{} embedding batch attempt {}/{} failed; retrying in {:.1f}s: {}",
                        provider,
                        attempt,
                        max(1, MEMORY_EMBEDDING_RETRY_ATTEMPTS),
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
            if last_error is not None:
                raise last_error
    except Exception as exc:
        logger.warning("{} embedding batch failed: {}", provider, exc)
        if require_all:
            raise RuntimeError(
                f"Could not create a complete {provider} embedding index; "
                "the previous RAG index was preserved"
            ) from exc
        return [None] * len(normalized)

    async with _embedding_lock:
        for value, embedding in completed.items():
            key = (provider, model, value, MEMORY_EMBEDDING_DIMENSION)
            _cache_embedding(key, embedding)
            for index in positions[value]:
                results[index] = list(embedding)

    if require_all and any(
        value and results[index] is None for index, value in enumerate(normalized)
    ):
        raise RuntimeError(
            f"Could not create a complete {provider} embedding index; "
            "the previous RAG index was preserved"
        )
    return results


async def embed_text(value: str) -> list[float] | None:
    value = re.sub(r"\s+", " ", (value or "").strip())
    if not value or MEMORY_VECTOR_DB != "pgvector":
        return None

    provider = memory_embedding_provider()
    if provider == "disabled":
        return None
    model = _embedding_model(provider)
    key = (provider, model, value, MEMORY_EMBEDDING_DIMENSION)
    now = time.monotonic()

    async with _embedding_lock:
        cached = _embedding_cache.get(key)
        if cached and now - cached[0] <= MEMORY_EMBEDDING_CACHE_TTL_SECONDS:
            _embedding_cache.move_to_end(key)
            return list(cached[1])
        if cached:
            _embedding_cache.pop(key, None)
        task = _embedding_inflight.get(key)
        if task is None:
            task = asyncio.create_task(_embed_uncached(value, provider))
            _embedding_inflight[key] = task

    try:
        embedding = await asyncio.shield(task)
    finally:
        async with _embedding_lock:
            if _embedding_inflight.get(key) is task and task.done():
                _embedding_inflight.pop(key, None)

    if embedding:
        async with _embedding_lock:
            _cache_embedding(key, embedding)
        return list(embedding)
    return None


def _transcript_lines(messages: list[TranscriptEntry], max_messages: int = 40) -> str:
    lines = []
    for message in messages[-max_messages:]:
        if message.speaker == "Aura" and not is_memory_safe_assistant_text(
            message.text, message.source
        ):
            continue
        speaker = "User" if message.speaker == "You" else "Aura"
        content = re.sub(r"\s+", " ", message.text or "").strip()
        if len(content) > 500:
            content = content[:497].rstrip() + "..."
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def _fallback_summary(messages: list[TranscriptEntry]) -> str:
    lines = []
    for message in messages[-20:]:
        if message.speaker == "Aura" and not is_memory_safe_assistant_text(
            message.text, message.source
        ):
            continue
        speaker = "User" if message.speaker == "You" else "Aura"
        content = re.sub(r"\s+", " ", message.text or "").strip()
        if len(content) > 180:
            content = content[:177].rstrip() + "..."
        lines.append(f"- {speaker}: {content}")
    return "Recent voice call notes:\n" + "\n".join(lines)


def _valid_fact_event(event: dict[str, Any]) -> dict[str, Any] | None:
    action = str(event.get("action", "ignore")).lower()
    key = _normalize_key(str(event.get("key", "")))
    value = _normalize_value(str(event.get("value", "")), key)
    fact_type = str(
        event.get("fact_type")
        or ("preference" if key in MULTI_VALUE_KEYS else "profile")
    ).lower()
    durability = str(event.get("durability", "stable")).lower()
    status = str(event.get("status", "active")).lower()
    try:
        confidence = float(event.get("confidence", 0))
    except TypeError, ValueError:
        confidence = 0

    if action not in {"upsert", "deactivate", "ignore"}:
        return None
    if action == "ignore":
        return None
    if confidence < MEMORY_FACT_CONFIDENCE_MIN:
        return None
    if not key or not value:
        return None
    if (
        key in {"real_name", "preferred_name", "name"}
        and value.lower() in INVALID_NAME_VALUES
    ):
        return None
    if durability not in VALID_DURABILITY:
        durability = "stable"
    if status not in VALID_STATUSES:
        status = "active"
    if durability == "temporary":
        return None

    if key == "name":
        key = "real_name"

    return {
        "action": action,
        "fact_type": fact_type,
        "key": key,
        "value": value,
        "confidence": confidence,
        "durability": durability,
        "status": status,
    }


async def classify_memory_events(
    user_text: str, assistant_text: str | None = None
) -> list[dict[str, Any]]:
    if not is_memory_fact_candidate(user_text):
        return []
    base_prompt = load_memory_prompt()
    prompt = f"{base_prompt}\n\nUser: {user_text}\nAssistant: {assistant_text or ''}"
    response = await _generate_text_with_memory_llm(prompt)
    data = _extract_json_object(response or "")
    events = []
    for item in data.get("events", []):
        if isinstance(item, dict) and (event := _valid_fact_event(item)):
            events.append(event)
    return events


async def apply_fact_events(
    db: AsyncSession,
    user_id: int,
    events: list[dict[str, Any]],
    source_message_id: int | None = None,
) -> None:
    if not events:
        return

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for event in events:
        key = event["key"]
        value = event["value"]
        fact_type = event["fact_type"]

        if event["action"] == "deactivate":
            await db.execute(
                text(
                    """
                    UPDATE user_memories
                    SET status = 'inactive', updated_at = :updated_at
                    WHERE user_id = :user_id
                      AND fact_type = :fact_type
                      AND key = :key
                      AND lower(value) = lower(:value)
                    """
                ),
                {
                    "updated_at": now,
                    "user_id": user_id,
                    "fact_type": fact_type,
                    "key": key,
                    "value": value,
                },
            )
            continue

        if key in SINGLE_VALUE_KEYS:
            await db.execute(
                text(
                    """
                    UPDATE user_memories
                    SET status = 'inactive', updated_at = :updated_at
                    WHERE user_id = :user_id
                      AND fact_type = :fact_type
                      AND key = :key
                      AND status = 'active'
                    """
                ),
                {
                    "updated_at": now,
                    "user_id": user_id,
                    "fact_type": fact_type,
                    "key": key,
                },
            )

        stmt = insert(UserMemory).values(
            user_id=user_id,
            fact_type=fact_type,
            key=key,
            value=value,
            confidence=event["confidence"],
            durability=event["durability"],
            status="active",
            source_transcript_id=source_message_id,
            created_at=now,
            updated_at=now,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_user_memory_fact_value",
            set_={
                "confidence": event["confidence"],
                "durability": event["durability"],
                "status": "active",
                "source_transcript_id": source_message_id,
                "updated_at": now,
            },
        )
        await db.execute(stmt)


def build_memory_chunk(
    call_id, transcripts: list[TranscriptEntry]
) -> dict[str, Any] | None:
    valid_messages = [
        entry
        for entry in transcripts
        if entry.speaker in {"You", "Aura"}
        and entry.text
        and (
            entry.speaker == "You"
            or is_memory_safe_assistant_text(entry.text, entry.source)
        )
    ]
    if not valid_messages:
        return None

    return {
        "call_id": call_id,
        "transcript_start_id": valid_messages[0].id,
        "transcript_end_id": valid_messages[-1].id,
        "chunk_text": _transcript_lines(
            valid_messages, max_messages=len(valid_messages)
        ),
        "summary": _fallback_summary(valid_messages),
    }


async def store_memory_chunk(
    db: AsyncSession,
    call: Call,
    transcripts: list[TranscriptEntry],
) -> MemoryChunk | None:
    chunk = build_memory_chunk(call.id, transcripts)
    if not chunk:
        return None

    embedding = await embed_text(chunk["chunk_text"])
    if not embedding:
        return None

    now = datetime.now(timezone.utc)
    stmt = insert(MemoryChunk).values(
        user_id=call.user_id,
        call_id=call.id,
        transcript_start_id=chunk["transcript_start_id"],
        transcript_end_id=chunk["transcript_end_id"],
        chunk_text=chunk["chunk_text"],
        summary=chunk["summary"],
        embedding=embedding,
        created_at=now,
        updated_at=now,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_memory_chunk_transcript_window",
        set_={
            "chunk_text": chunk["chunk_text"],
            "summary": chunk["summary"],
            "embedding": embedding,
            "updated_at": now,
        },
    )
    try:
        async with db.begin_nested():
            result = await db.execute(stmt.returning(MemoryChunk.id))
            chunk_id = result.scalar_one_or_none()
    except Exception as exc:
        logger.warning(f"Skipping vector memory chunk write: {exc}")
        return None

    if not chunk_id:
        return None
    stored = await db.execute(select(MemoryChunk).where(MemoryChunk.id == chunk_id))
    return stored.scalars().first()


_CROSS_CALL_RECALL_RE = re.compile(
    r"\b(?:last time|previously|another (?:call|conversation|chat|session)|"
    r"(?:previous|prior|earlier|past) (?:call|conversation|chat|session)|"
    r"(?:call|conversation|chat|session) (?:before|from before)|"
    r"past conversations?)\b",
    re.IGNORECASE,
)


def is_recall_query(query: str) -> bool:
    """Return whether a turn explicitly requests cross-call episodic recall.

    Current-call dialogue is already in the live LLM context. Broad wording
    such as "I said" or "what did we discuss" must not trigger database and
    embedding work unless the user clearly scopes it to an older call.
    """
    return bool(_CROSS_CALL_RECALL_RE.search(query or ""))


async def retrieve_semantic_memories(
    user_id: int,
    query: str,
    top_k: int = MEMORY_RECALL_TOP_K,
    query_embedding=None,
    current_call_id=None,
) -> list[tuple[MemoryChunk, float]]:
    if not is_recall_query(query):
        return []

    if isinstance(query_embedding, asyncio.Future) or asyncio.iscoroutine(
        query_embedding
    ):
        embedding = await query_embedding
    elif query_embedding is not None:
        embedding = query_embedding
    else:
        embedding = await embed_text(query)
    if not embedding:
        return []

    try:
        async with VoiceSessionLocal() as db:
            distance = MemoryChunk.embedding.cosine_distance(embedding).label(
                "distance"
            )
            conditions = [
                MemoryChunk.user_id == user_id,
                MemoryChunk.embedding.is_not(None),
                Call.deleted_at.is_(None),
                Call.status.in_(("completed", "failed", "cancelled", "abandoned")),
            ]
            if current_call_id is not None:
                conditions.append(Call.id != current_call_id)
            result = await db.execute(
                select(MemoryChunk, distance)
                .join(Call, Call.id == MemoryChunk.call_id)
                .where(*conditions)
                .order_by(distance)
                .limit(top_k)
            )
            memories = []
            for chunk, dist in result.all():
                if contains_reserved_tool_markup(chunk.summary or chunk.chunk_text):
                    continue
                score = 1 - float(dist)
                if score >= MEMORY_RECALL_MIN_SCORE:
                    memories.append((chunk, score))
            return memories
    except Exception as exc:
        logger.warning(f"Skipping vector memory retrieval: {exc}")
        return []


async def build_turn_memory_context(
    user_id: int,
    query: str,
    query_embedding=None,
    current_call_id=None,
) -> str | None:
    # An explicit reference to the immediately preceding call has a fast,
    # deterministic local path. Broader recall queries use semantic chunks.
    immediate_recall = bool(
        re.search(
            r"\b(?:last time|previously|previous call|prior call|earlier call)\b",
            query,
            re.IGNORECASE,
        )
    )
    if immediate_recall and current_call_id is not None:
        async with VoiceSessionLocal() as db:
            prior_call = await _load_most_recent_prior_call(
                db,
                user_id,
                current_call_id,
            )
            prior_transcripts = (
                await _load_recent_transcripts(
                    db,
                    prior_call.id,
                    PRIOR_CONVERSATION_MESSAGE_LIMIT,
                )
                if prior_call
                else []
            )
        if prior_transcripts:
            lines = [
                "Relevant recent prior call retrieved on explicit recall. "
                "Use it only to answer the current recall question."
            ]
            if summary := _safe_prior_summary(prior_call.summary):
                lines.append(f"- Summary: {summary[:MEMORY_SUMMARY_MAX_CHARS]}")
            lines.extend(
                f"- {'User' if entry.speaker == 'You' else 'Aura'}: {entry.text}"
                for entry in prior_transcripts
            )
            return "\n".join(lines)[:MEMORY_PRIOR_MAX_CHARS]

    memories = await retrieve_semantic_memories(
        user_id,
        query,
        MEMORY_RECALL_TOP_K,
        query_embedding=query_embedding,
        current_call_id=current_call_id,
    )
    if not memories:
        if not is_recall_query(query):
            return None
        async with VoiceSessionLocal() as db:
            prior_call = await _load_most_recent_prior_call(
                db,
                user_id,
                current_call_id,
            )
            if not prior_call:
                return None
            prior_transcripts = await _load_recent_transcripts(
                db,
                prior_call.id,
                PRIOR_CONVERSATION_MESSAGE_LIMIT,
            )
        if not prior_transcripts:
            return None
        lines = [
            "Relevant recent prior call retrieved on explicit recall. "
            "Use it only to answer the current recall question."
        ]
        if summary := _safe_prior_summary(prior_call.summary):
            lines.append(f"- Summary: {summary[:MEMORY_SUMMARY_MAX_CHARS]}")
        lines.extend(
            f"- {'User' if entry.speaker == 'You' else 'Aura'}: {entry.text}"
            for entry in prior_transcripts
        )
        return "\n".join(lines)[:MEMORY_PRIOR_MAX_CHARS]

    lines = [
        "Relevant long-term episodic memories retrieved for this user. Use them only if relevant to the user's current question."
    ]
    for chunk, score in memories:
        content = chunk.summary or chunk.chunk_text
        if contains_reserved_tool_markup(content):
            continue
        lines.append(f"- score={score:.2f} call={chunk.call_id}: {content}")
    return "\n".join(lines)


async def maintain_memory_chunks_if_needed(db: AsyncSession, call: Call) -> None:
    """Maintain episodic chunks; live Pipecat summaries own canonical summary."""
    count_result = await db.execute(
        select(func.count(TranscriptEntry.id)).where(
            TranscriptEntry.call_id == call.id,
            TranscriptEntry.speaker.in_(["You", "Aura"]),
        )
    )
    message_count = count_result.scalar_one()

    if message_count % 8 != 0:
        return
    recent_result = await db.execute(
        select(TranscriptEntry)
        .where(
            TranscriptEntry.call_id == call.id,
            TranscriptEntry.speaker.in_(["You", "Aura"]),
        )
        .order_by(TranscriptEntry.created_at.desc(), TranscriptEntry.id.desc())
        .limit(8)
    )
    recent_transcripts = list(reversed(recent_result.scalars().all()))
    if len(recent_transcripts) >= 2 and recent_transcripts[-1].speaker == "Aura":
        await store_memory_chunk(db, call, recent_transcripts)


async def process_saved_transcript(call_id, transcript_id: int) -> None:
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Call).where(Call.id == call_id))
        call = result.scalars().first()
        transcript_result = await db.execute(
            select(TranscriptEntry).where(TranscriptEntry.id == transcript_id)
        )
        entry = transcript_result.scalars().first()

        if call and entry:
            if entry.speaker == "You":
                events = await classify_memory_events(entry.text)
                await apply_fact_events(db, call.user_id, events, entry.id)

            if entry.speaker == "Aura":
                await maintain_memory_chunks_if_needed(db, call)

            await db.commit()
