"""Process-scoped Groq HTTP clients for latency-sensitive voice inference."""

from __future__ import annotations

import asyncio
import os
import threading
import time

import httpx
from loguru import logger
from openai import AsyncOpenAI, DefaultAsyncHttpxClient


GROQ_BASE_URL = "https://api.groq.com/openai/v1"

_clients: dict[tuple[str, str], AsyncOpenAI] = {}
_warm_state: dict[tuple[str, str], bool] = {}
_clients_lock = threading.Lock()


def _cache_key(api_key: str | None, base_url: str | None = None) -> tuple[str, str]:
    resolved_key = (api_key or "").strip()
    if not resolved_key:
        raise ValueError("GROQ_API_KEY is required when Groq is selected")
    return ((base_url or GROQ_BASE_URL).rstrip("/"), resolved_key)


def groq_runtime_warmed(
    *, api_key: str | None = None, base_url: str | None = None
) -> bool:
    """Return the observed connection state, never an optimistic default."""
    resolved_key = api_key if api_key is not None else os.getenv("GROQ_API_KEY")
    try:
        cache_key = _cache_key(resolved_key, base_url)
    except ValueError:
        return False
    with _clients_lock:
        return _warm_state.get(cache_key, False)


def mark_groq_runtime_unwarmed(
    *, api_key: str | None = None, base_url: str | None = None
) -> None:
    resolved_key = api_key if api_key is not None else os.getenv("GROQ_API_KEY")
    try:
        cache_key = _cache_key(resolved_key, base_url)
    except ValueError:
        return
    with _clients_lock:
        _warm_state[cache_key] = False


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}")
    return value


def groq_http_timeout() -> httpx.Timeout:
    """Return explicit transport limits; SDK-owned retries are disabled separately."""
    return httpx.Timeout(
        connect=_bounded_float("GROQ_CONNECT_TIMEOUT_SECONDS", 1.5, 0.1, 10.0),
        read=_bounded_float("GROQ_READ_TIMEOUT_SECONDS", 30.0, 1.0, 120.0),
        write=_bounded_float("GROQ_WRITE_TIMEOUT_SECONDS", 5.0, 0.5, 30.0),
        pool=_bounded_float("GROQ_POOL_TIMEOUT_SECONDS", 0.5, 0.1, 10.0),
    )


def get_shared_groq_client(
    *,
    api_key: str | None,
    base_url: str | None = None,
) -> AsyncOpenAI:
    """Return one concurrency-safe OpenAI-compatible client per Groq endpoint/key.

    A small keyed cache keeps tests and explicit credential rotation isolated while
    ensuring ordinary calls reuse the same HTTP/2 connection pool. The OpenAI SDK
    retry layer is intentionally disabled so the voice deadline owns all retries.
    """
    cache_key = _cache_key(api_key, base_url)
    resolved_url, resolved_key = cache_key
    with _clients_lock:
        existing = _clients.get(cache_key)
        if existing is not None and not existing.is_closed():
            return existing

        timeout = groq_http_timeout()
        http_client = DefaultAsyncHttpxClient(
            http2=True,
            timeout=timeout,
            limits=httpx.Limits(
                max_keepalive_connections=100,
                max_connections=1000,
                keepalive_expiry=_bounded_float(
                    "GROQ_KEEPALIVE_EXPIRY_SECONDS", 60.0, 5.0, 300.0
                ),
            ),
        )
        client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=resolved_url,
            max_retries=0,
            timeout=timeout,
            http_client=http_client,
        )
        _clients[cache_key] = client
        return client


async def warm_groq_runtime(
    *, api_key: str | None = None, base_url: str | None = None
) -> bool:
    """Warm DNS/TLS/authentication and retain the truthful observed state."""
    api_key = (api_key if api_key is not None else os.getenv("GROQ_API_KEY", "")).strip()
    if not api_key:
        return False
    cache_key = _cache_key(api_key, base_url)
    if groq_runtime_warmed(api_key=api_key, base_url=base_url):
        return True
    client = get_shared_groq_client(api_key=api_key, base_url=base_url)
    timeout = _bounded_float("GROQ_WARMUP_TIMEOUT_SECONDS", 5.0, 0.1, 15.0)
    started = time.monotonic()
    try:
        await asyncio.wait_for(client.models.list(), timeout=timeout)
    except Exception as exc:
        with _clients_lock:
            _warm_state[cache_key] = False
        logger.warning(
            "voice_llm provider=groq status=startup_warmup_failed latency_ms={} "
            "error_type={}",
            round((time.monotonic() - started) * 1000, 1),
            type(exc).__name__,
        )
        return False
    with _clients_lock:
        _warm_state[cache_key] = True
    logger.info(
        "voice_llm provider=groq status=startup_warmed latency_ms={}",
        round((time.monotonic() - started) * 1000, 1),
    )
    return True


async def shutdown_groq_runtime() -> None:
    """Close every process-scoped client exactly once during application shutdown."""
    with _clients_lock:
        clients = list(_clients.values())
        _clients.clear()
        _warm_state.clear()
    if clients:
        await asyncio.gather(*(client.close() for client in clients), return_exceptions=True)


def reset_groq_runtime_for_tests() -> list[AsyncOpenAI]:
    """Detach cached clients for unit tests; callers remain responsible for closing them."""
    with _clients_lock:
        clients = list(_clients.values())
        _clients.clear()
        _warm_state.clear()
    return clients
