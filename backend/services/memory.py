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
from sqlalchemy import func, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from api.auth import ALGORITHM, SECRET_KEY
from core.database import AsyncSessionLocal, VoiceSessionLocal
from core.memory_config import (
    MEMORY_EMBEDDING_DIMENSION,
    MEMORY_EMBEDDING_CACHE_SIZE,
    MEMORY_EMBEDDING_CACHE_TTL_SECONDS,
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
    PRIOR_CONVERSATION_MESSAGE_LIMIT,
    RECENT_MESSAGE_LIMIT,
    SUMMARY_MESSAGE_THRESHOLD,
)
from core.models import Conversation, MemoryChunk, Message, RagFile, User, UserMemory
from core.prompt_config import load_memory_prompt


SINGLE_VALUE_KEYS = {"real_name", "preferred_name", "location", "role", "preferred_language"}
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
_embedding_cache: OrderedDict[tuple[str, str, str, int], tuple[float, list[float]]] = OrderedDict()
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
    primary_conversation: Conversation
    facts: list[UserMemory]
    primary_summary: str
    primary_recent_messages: list[Message]
    prior_conversation: Conversation | None = None
    prior_recent_messages: list[Message] | None = None
    has_ready_rag_corpus: bool = False

    @property
    def conversation(self) -> Conversation:
        return self.primary_conversation

    @property
    def summary(self) -> str:
        return self.primary_summary

    @property
    def recent_messages(self) -> list[Message]:
        return self.primary_recent_messages


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


async def authenticate_conversation(
    token: str | None,
    conversation_id: int,
    db: AsyncSession,
) -> tuple[User, Conversation] | None:
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

    result = await db.execute(
        select(User, Conversation)
        .join(Conversation, Conversation.user_id == User.id)
        .where(
            User.username == username,
            Conversation.id == conversation_id,
        )
    )
    row = result.first()
    return (row[0], row[1]) if row else None


async def _load_recent_messages(
    db: AsyncSession,
    conversation_id: int,
    limit: int,
) -> list[Message]:
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id, Message.role.in_(["You", "Aura"]))
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(limit)
    )
    return list(reversed(result.scalars().all()))


async def _load_most_recent_prior_conversation(
    db: AsyncSession,
    user_id: int,
    current_conversation_id: int,
) -> Conversation | None:
    result = await db.execute(
        select(Conversation)
        .where(
            Conversation.user_id == user_id,
            Conversation.id != current_conversation_id,
        )
        .order_by(Conversation.updated_at.desc(), Conversation.created_at.desc())
        .limit(1)
    )
    return result.scalars().first()


async def _load_active_facts(user_id: int) -> list[UserMemory]:
    async with VoiceSessionLocal() as db:
        result = await db.execute(
            select(UserMemory)
            .where(UserMemory.user_id == user_id, UserMemory.status == "active")
            .order_by(UserMemory.fact_type.asc(), UserMemory.key.asc(), UserMemory.updated_at.desc())
        )
        return list(result.scalars().all())


async def _load_recent_messages_in_session(conversation_id: int, limit: int) -> list[Message]:
    async with VoiceSessionLocal() as db:
        return await _load_recent_messages(db, conversation_id, limit)


async def load_session_bundle(body: Any) -> MemoryBundle | None:
    """Authenticate and resolve the conversation without optional history I/O."""
    request_body = normalize_runner_body(body)
    token = request_body.get("token")
    conversation_id = request_body.get("conversation_id")
    if not token:
        return None

    if conversation_id is not None:
        try:
            conversation_id = int(conversation_id)
        except (TypeError, ValueError):
            logger.warning("Memory hydration skipped: invalid conversation_id")
            return None

    async with VoiceSessionLocal() as db:
        if conversation_id is None:
            user = await authenticate_token(token, db)
            if not user:
                logger.warning("Memory hydration skipped: invalid token")
                return None
            conversation = Conversation(user_id=user.id, title="New conversation")
            db.add(conversation)
            await db.commit()
            await db.refresh(conversation)
            conversation_id = conversation.id
        else:
            authenticated = await authenticate_conversation(token, conversation_id, db)
            if not authenticated:
                logger.warning("Memory hydration skipped: invalid token or conversation ownership")
                return None
            user, conversation = authenticated

        user_id = user.id
        ready_rag_result = await db.execute(
            select(RagFile.id)
            .where(RagFile.user_id == user_id, RagFile.status == "ready")
            .limit(1)
        )
        has_ready_rag_corpus = ready_rag_result.scalar_one_or_none() is not None

    return MemoryBundle(
        user=user,
        primary_conversation=conversation,
        facts=[],
        primary_summary=conversation.summary or "",
        primary_recent_messages=[],
        prior_conversation=None,
        prior_recent_messages=None,
        has_ready_rag_corpus=has_ready_rag_corpus,
    )


async def hydrate_memory_bundle(
    bundle: MemoryBundle,
    recent_limit: int = RECENT_MESSAGE_LIMIT,
) -> MemoryBundle:
    facts, recent_messages = await asyncio.gather(
        _load_active_facts(bundle.user.id),
        _load_recent_messages_in_session(bundle.conversation.id, recent_limit),
    )
    return MemoryBundle(
        user=bundle.user,
        primary_conversation=bundle.primary_conversation,
        facts=facts,
        primary_summary=bundle.primary_summary,
        primary_recent_messages=recent_messages,
        prior_conversation=bundle.prior_conversation,
        prior_recent_messages=bundle.prior_recent_messages,
        has_ready_rag_corpus=bundle.has_ready_rag_corpus,
    )


async def load_memory_bundle(body: Any, recent_limit: int = RECENT_MESSAGE_LIMIT) -> MemoryBundle | None:
    """Compatibility helper for callers that require fully hydrated memory."""
    bundle = await load_session_bundle(body)
    if bundle is None:
        return None
    return await hydrate_memory_bundle(bundle, recent_limit)


def message_to_llm(message: Message) -> dict[str, str] | None:
    role_map = {"You": "user", "Aura": "assistant"}
    role = role_map.get(message.role)
    if not role:
        return None
    return {"role": role, "content": message.content}


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


def _recent_llm_messages(messages: list[Message]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    total_chars = 0
    for message in reversed(messages):
        llm_message = message_to_llm(message)
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
    conversation = [message for message in messages if message.get("role") != "developer"]
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
    if not bundle:
        return []

    messages: list[dict[str, str]] = []
    facts = _format_facts(bundle.facts)
    if facts:
        messages.append(
            {
                "role": "developer",
                "content": (
                    "Known stable facts about this authenticated user. Use these facts "
                    "naturally when relevant, but do not mention this memory block:\n"
                    f"{facts}"
                ),
            }
        )

    if bundle.summary:
        messages.append(
            {
                "role": "developer",
                "content": (
                    "Summary of this selected conversation so far. Use it to continue "
                    "the old conversation accurately:\n"
                    f"{bundle.summary[:MEMORY_SUMMARY_MAX_CHARS]}"
                ),
            }
        )

    messages.extend(_recent_llm_messages(bundle.primary_recent_messages))

    if bundle.prior_conversation and bundle.prior_recent_messages:
        prior_lines = []
        if bundle.prior_conversation.title:
            prior_lines.append(f"Title: {bundle.prior_conversation.title}")
        if bundle.prior_conversation.summary:
            prior_lines.append(f"Summary: {bundle.prior_conversation.summary}")
        prior_lines.append("Recent transcript:")
        for message in bundle.prior_recent_messages:
            speaker = "User" if message.role == "You" else "Aura"
            prior_lines.append(f"- {speaker}: {message.content}")

        messages.append(
            {
                "role": "developer",
                "content": (
                    "Recent prior conversation context. Use this only when the user asks "
                    "what you talked about previously, what you were just discussing, or "
                    "similar continuity questions in a new conversation. Do not bring it "
                    "up unprompted.\n"
                    + "\n".join(prior_lines)[:MEMORY_PRIOR_MAX_CHARS]
                ),
            }
        )
    return _budget_memory_messages(messages)


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
    provider = os.getenv("LLM_PROVIDER", "google").lower()

    async def generate_google() -> str | None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None
        client = _get_google_client()
        response = await client.aio.models.generate_content(
            model=os.getenv("GOOGLE_MEMORY_MODEL", os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")),
            contents=prompt,
        )
        return getattr(response, "text", None)

    async def generate_openai() -> str | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        client = _get_openai_client()
        response = await client.chat.completions.create(
            model=os.getenv("OPENAI_MEMORY_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    async def generate_groq() -> str | None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        client = _get_groq_client()
        response = await client.chat.completions.create(
            model=os.getenv("GROQ_MEMORY_MODEL", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    generator = {
        "google": generate_google,
        "groq": generate_groq,
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


async def _embed_uncached(value: str, provider: str) -> list[float] | None:
    def normalize_embedding(embedding: list[float] | None, provider: str) -> list[float] | None:
        if not embedding:
            return None
        if len(embedding) != MEMORY_EMBEDDING_DIMENSION:
            logger.warning(
                f"{provider} embedding dimension {len(embedding)} does not match "
                f"MEMORY_EMBEDDING_DIMENSION={MEMORY_EMBEDDING_DIMENSION}; skipping vector memory."
            )
            return None
        return embedding

    async def embed_google() -> list[float] | None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: return None
        from google.genai import types
        client = _get_google_client()
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")
        response = await client.aio.models.embed_content(
            model=model,
            contents=value,
            config=types.EmbedContentConfig(output_dimensionality=MEMORY_EMBEDDING_DIMENSION)
        )
        return normalize_embedding(response.embeddings[0].values, "Google")

    async def embed_openai() -> list[float] | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: return None
        client = _get_openai_client()
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        response = await client.embeddings.create(input=value, model=model)
        return normalize_embedding(response.data[0].embedding, "OpenAI")

    if provider == "disabled":
        return None
    generator = {
        "google": embed_google,
        "openai": embed_openai,
    }.get(provider)
    if generator is None:
        raise ValueError(f"Unsupported memory embedding provider: {provider!r}")
    try:
        return await asyncio.wait_for(
            generator(), timeout=MEMORY_LLM_TIMEOUT_SECONDS * 5
        )
    except Exception as exc:
        # Never cross-fallback to another paid provider. The selected provider
        # owns its availability, quota, and latency policy.
        logger.warning("{} embedding call failed: {}", provider, exc)
        return None


async def embed_text(value: str) -> list[float] | None:
    value = re.sub(r"\s+", " ", (value or "").strip())
    if not value or MEMORY_VECTOR_DB != "pgvector":
        return None

    provider = memory_embedding_provider()
    if provider == "disabled":
        return None
    model = (
        os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")
        if provider == "google"
        else os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
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
            _embedding_cache[key] = (time.monotonic(), list(embedding))
            _embedding_cache.move_to_end(key)
            while len(_embedding_cache) > max(1, MEMORY_EMBEDDING_CACHE_SIZE):
                _embedding_cache.popitem(last=False)
        return list(embedding)
    return None


def _transcript_lines(messages: list[Message], max_messages: int = 40) -> str:
    lines = []
    for message in messages[-max_messages:]:
        speaker = "User" if message.role == "You" else "Aura"
        content = re.sub(r"\s+", " ", message.content or "").strip()
        if len(content) > 500:
            content = content[:497].rstrip() + "..."
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def _fallback_summary(messages: list[Message]) -> str:
    lines = []
    for message in messages[-20:]:
        speaker = "User" if message.role == "You" else "Aura"
        content = re.sub(r"\s+", " ", message.content or "").strip()
        if len(content) > 180:
            content = content[:177].rstrip() + "..."
        lines.append(f"- {speaker}: {content}")
    return "Recent voice conversation notes:\n" + "\n".join(lines)


async def generate_conversation_summary(messages: list[Message]) -> str:
    transcript = _transcript_lines(messages)
    if not transcript:
        return ""

    prompt = (
        "Summarize this voice conversation for future continuity. Keep it concise, "
        "factual, and useful when the user later asks what they talked about. Capture "
        "topics discussed, user questions, answers given, decisions, unresolved items, "
        "and user preferences. Do not invent facts.\n\n"
        f"{transcript}\n\n"
        "Return only the summary text in 4-8 short bullet points."
    )
    summary = await _generate_text_with_memory_llm(prompt)
    return summary.strip() if summary else _fallback_summary(messages)


def _valid_fact_event(event: dict[str, Any]) -> dict[str, Any] | None:
    action = str(event.get("action", "ignore")).lower()
    key = _normalize_key(str(event.get("key", "")))
    value = _normalize_value(str(event.get("value", "")), key)
    fact_type = str(event.get("fact_type") or ("preference" if key in MULTI_VALUE_KEYS else "profile")).lower()
    durability = str(event.get("durability", "stable")).lower()
    status = str(event.get("status", "active")).lower()
    try:
        confidence = float(event.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0

    if action not in {"upsert", "deactivate", "ignore"}:
        return None
    if action == "ignore":
        return None
    if confidence < MEMORY_FACT_CONFIDENCE_MIN:
        return None
    if not key or not value:
        return None
    if key in {"real_name", "preferred_name", "name"} and value.lower() in INVALID_NAME_VALUES:
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


async def classify_memory_events(user_text: str, assistant_text: str | None = None) -> list[dict[str, Any]]:
    if not is_memory_fact_candidate(user_text):
        return []
    base_prompt = load_memory_prompt()
    prompt = (
        f"{base_prompt}\n\n"
        f"User: {user_text}\n"
        f"Assistant: {assistant_text or ''}"
    )
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
            source_message_id=source_message_id,
            created_at=now,
            updated_at=now,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_user_memory_fact_value",
            set_={
                "confidence": event["confidence"],
                "durability": event["durability"],
                "status": "active",
                "source_message_id": source_message_id,
                "updated_at": now,
            },
        )
        await db.execute(stmt)


def build_memory_chunk(conversation_id: int, messages: list[Message]) -> dict[str, Any] | None:
    valid_messages = [message for message in messages if message.role in {"You", "Aura"} and message.content]
    if not valid_messages:
        return None

    return {
        "conversation_id": conversation_id,
        "message_start_id": valid_messages[0].id,
        "message_end_id": valid_messages[-1].id,
        "chunk_text": _transcript_lines(valid_messages, max_messages=len(valid_messages)),
        "summary": _fallback_summary(valid_messages),
    }


async def store_memory_chunk(
    db: AsyncSession,
    conversation: Conversation,
    messages: list[Message],
) -> MemoryChunk | None:
    chunk = build_memory_chunk(conversation.id, messages)
    if not chunk:
        return None

    embedding = await embed_text(chunk["chunk_text"])
    if not embedding:
        return None

    now = datetime.now(timezone.utc)
    stmt = insert(MemoryChunk).values(
        user_id=conversation.user_id,
        conversation_id=conversation.id,
        message_start_id=chunk["message_start_id"],
        message_end_id=chunk["message_end_id"],
        chunk_text=chunk["chunk_text"],
        summary=chunk["summary"],
        embedding=embedding,
        created_at=now,
        updated_at=now,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_memory_chunk_message_window",
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


def is_recall_query(query: str) -> bool:
    lowered = query.lower()
    recall_terms = [
        "what did",
        "what was",
        "previously",
        "earlier",
        "last time",
        "remind me",
        "remember",
        "talk about",
        "discuss",
        "mentioned",
        "i said",
        "did i ask",
    ]
    return any(term in lowered for term in recall_terms)


async def retrieve_semantic_memories(
    user_id: int,
    query: str,
    top_k: int = MEMORY_RECALL_TOP_K,
    query_embedding=None,
) -> list[tuple[MemoryChunk, float]]:
    if not is_recall_query(query):
        return []

    if isinstance(query_embedding, asyncio.Future) or asyncio.iscoroutine(query_embedding):
        embedding = await query_embedding
    elif query_embedding is not None:
        embedding = query_embedding
    else:
        embedding = await embed_text(query)
    if not embedding:
        return []

    try:
        async with VoiceSessionLocal() as db:
            distance = MemoryChunk.embedding.cosine_distance(embedding).label("distance")
            result = await db.execute(
                select(MemoryChunk, distance)
                .where(MemoryChunk.user_id == user_id, MemoryChunk.embedding.is_not(None))
                .order_by(distance)
                .limit(top_k)
            )
            memories = []
            for chunk, dist in result.all():
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
    current_conversation_id: int | None = None,
) -> str | None:
    # Questions about the immediately preceding chat have a deterministic,
    # local answer. Prefer that path over a remote embedding so cross-chat
    # recall remains reliable inside the voice latency budget.
    if is_recall_query(query) and current_conversation_id is not None:
        async with VoiceSessionLocal() as db:
            prior_conversation = await _load_most_recent_prior_conversation(
                db,
                user_id,
                current_conversation_id,
            )
            prior_messages = (
                await _load_recent_messages(
                    db,
                    prior_conversation.id,
                    PRIOR_CONVERSATION_MESSAGE_LIMIT,
                )
                if prior_conversation
                else []
            )
        if prior_messages:
            lines = [
                "Relevant recent prior conversation retrieved on explicit recall. "
                "Use it only to answer the current recall question."
            ]
            lines.extend(
                f"- {'User' if message.role == 'You' else 'Aura'}: {message.content}"
                for message in prior_messages
            )
            return "\n".join(lines)[:MEMORY_PRIOR_MAX_CHARS]

    memories = await retrieve_semantic_memories(
        user_id,
        query,
        MEMORY_RECALL_TOP_K,
        query_embedding=query_embedding,
    )
    if not memories:
        if not is_recall_query(query):
            return None
        async with VoiceSessionLocal() as db:
            prior_conversation = await _load_most_recent_prior_conversation(
                db,
                user_id,
                current_conversation_id or -1,
            )
            if not prior_conversation:
                return None
            prior_messages = await _load_recent_messages(
                db,
                prior_conversation.id,
                PRIOR_CONVERSATION_MESSAGE_LIMIT,
            )
        if not prior_messages:
            return None
        lines = [
            "Relevant recent prior conversation retrieved on explicit recall. "
            "Use it only to answer the current recall question."
        ]
        lines.extend(
            f"- {'User' if message.role == 'You' else 'Aura'}: {message.content}"
            for message in prior_messages
        )
        return "\n".join(lines)[:MEMORY_PRIOR_MAX_CHARS]

    lines = [
        "Relevant long-term episodic memories retrieved for this user. Use them only if relevant to the user's current question."
    ]
    for chunk, score in memories:
        lines.append(f"- score={score:.2f} conversation={chunk.conversation_id}: {chunk.summary or chunk.chunk_text}")
    return "\n".join(lines)


async def update_conversation_summary_if_needed(db: AsyncSession, conversation: Conversation) -> None:
    count_result = await db.execute(
        select(func.count(Message.id)).where(
            Message.conversation_id == conversation.id,
            Message.role.in_(["You", "Aura"]),
        )
    )
    message_count = count_result.scalar_one()

    if message_count <= SUMMARY_MESSAGE_THRESHOLD:
        if message_count % 8 != 0:
            return
        recent_result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation.id, Message.role.in_(["You", "Aura"]))
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(8)
        )
        recent_messages = list(reversed(recent_result.scalars().all()))
        if len(recent_messages) >= 2:
            await store_memory_chunk(db, conversation, recent_messages)
        return

    messages_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id, Message.role.in_(["You", "Aura"]))
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(40)
    )
    messages = list(reversed(messages_result.scalars().all()))
    total_chars = sum(len(message.content or "") for message in messages)

    if message_count % SUMMARY_MESSAGE_THRESHOLD == 0:
        conversation.summary = await generate_conversation_summary(messages)

    if message_count % 8 == 0 and len(messages) >= 2 and messages[-1].role == "Aura":
        await store_memory_chunk(db, conversation, messages[-8:])


async def save_conversation_message(
    conversation_id: int | None,
    role: str,
    content: str,
) -> Message | None:
    content = (content or "").strip()
    if not conversation_id or not content or role not in {"You", "Aura", "ToolCall", "RagCall"}:
        return None

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Conversation).where(Conversation.id == conversation_id))
        conversation = result.scalars().first()
        if not conversation:
            logger.warning(f"Could not save memory message: conversation {conversation_id} not found")
            return None

        message = Message(conversation_id=conversation_id, role=role, content=content)
        db.add(message)
        
        # We need to update conversation title/updated_at here before commit
        conversation.updated_at = datetime.now(timezone.utc)
        if conversation.title == "New conversation" and message.role == "You":
            new_title = " ".join(message.content.split()[:4])
            conversation.title = new_title or conversation.title
            
        await db.commit()
        await db.refresh(message)
        
    from core.task_queue import task_queue
    task_queue.enqueue(
        _process_saved_message_background,
        conversation_id,
        message.id,
        key=conversation_id,
        enrichment=True,
    )
    return message


async def _process_saved_message_background(conversation_id: int, message_id: int):
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Conversation).where(Conversation.id == conversation_id))
        conversation = result.scalars().first()
        msg_result = await db.execute(select(Message).where(Message.id == message_id))
        message = msg_result.scalars().first()
        
        if conversation and message:
            if message.role == "You":
                events = await classify_memory_events(message.content)
                await apply_fact_events(db, conversation.user_id, events, message.id)
        
            if message.role == "Aura":
                await update_conversation_summary_if_needed(db, conversation)
            
            await db.commit()

async def process_saved_message(db: AsyncSession, conversation: Conversation, message: Message) -> None:
    from core.task_queue import task_queue
    task_queue.enqueue(
        _process_saved_message_background,
        conversation.id,
        message.id,
        key=conversation.id,
        enrichment=True,
    )
