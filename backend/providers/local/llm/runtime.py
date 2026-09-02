"""Process-wide client and startup warmup for a local llama.cpp server."""

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Any

from loguru import logger
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
import httpx

from core.prompt_config import load_system_prompt
from tools.registry import configured_openai_tool_schemas

from .config import LocalLLMConfig, load_local_llm_config


ClientFactory = Callable[[LocalLLMConfig], Any]
SlotProbe = Callable[[LocalLLMConfig], Any]


def _create_client(config: LocalLLMConfig) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        http_client=DefaultAsyncHttpxClient(
            limits=httpx.Limits(
                max_keepalive_connections=4,
                max_connections=8,
                keepalive_expiry=None,
            )
        ),
    )


async def _probe_server_slots(config: LocalLLMConfig) -> int:
    server_root = config.base_url.removesuffix("/v1")
    async with httpx.AsyncClient(timeout=config.warmup_timeout_seconds) as client:
        response = await client.get(f"{server_root}/slots")
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("llama.cpp /slots returned an invalid payload")
    return len(payload)


class LocalLLMRuntime:
    """Own the shared async client used by all local voice sessions."""

    def __init__(
        self,
        config: LocalLLMConfig,
        *,
        client_factory: ClientFactory = _create_client,
        slot_probe: SlotProbe = _probe_server_slots,
    ):
        self.config = config
        self.client = client_factory(config)
        self._slot_probe = slot_probe
        self._warm_lock = asyncio.Lock()
        self._warmed = False
        self._closed = False

    @property
    def warmed(self) -> bool:
        return self._warmed

    async def warm(self) -> None:
        """Verify the configured model and run one short generation."""
        if self._closed:
            raise RuntimeError("Local LLM runtime is closed")
        if self._warmed:
            return

        async with self._warm_lock:
            if self._warmed:
                return
            started = time.monotonic()

            async def perform_warmup() -> None:
                models = await self.client.models.list()
                model_ids = {
                    getattr(model, "id", None)
                    for model in getattr(models, "data", models)
                }
                if self.config.model not in model_ids:
                    available = sorted(
                        model_id for model_id in model_ids if model_id
                    )
                    raise RuntimeError(
                        f"Local LLM model {self.config.model!r} is unavailable; "
                        f"server reported {available!r}"
                    )
                if self.config.validate_server_slots:
                    slot_count = await self._slot_probe(self.config)
                    if slot_count < self.config.max_concurrent_sessions:
                        raise RuntimeError(
                            "llama.cpp has fewer parallel slots than the backend "
                            f"admits: server={slot_count}, "
                            f"backend={self.config.max_concurrent_sessions}"
                        )
                warmup_messages = [
                    {
                        "role": "system",
                        "content": load_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": "Reply with only OK to confirm readiness.",
                    },
                ]

                async def warm_slot(slot_id: int) -> None:
                    await self.client.chat.completions.create(
                        model=self.config.model,
                        messages=warmup_messages,
                        stream=False,
                        temperature=0.0,
                        max_tokens=4,
                        tools=configured_openai_tool_schemas(),
                        tool_choice="auto",
                        extra_body={
                            **self.config.extra_body,
                            "id_slot": slot_id,
                        },
                    )

                # Prompt caches are slot-local in llama.cpp. Keep requests in
                # flight together so every admitted parallel slot receives
                # the production prefix before voice traffic begins.
                await asyncio.gather(
                    *(
                        warm_slot(slot_id)
                        for slot_id in range(
                            self.config.max_concurrent_sessions
                        )
                    )
                )

            try:
                await asyncio.wait_for(
                    perform_warmup(),
                    timeout=self.config.warmup_timeout_seconds,
                )
            except BaseException as exc:
                logger.error(
                    "voice_startup stage=local_llm_warmup provider=local "
                    "model={} status=failed duration_ms={} error_type={}",
                    self.config.model,
                    round((time.monotonic() - started) * 1000, 1),
                    type(exc).__name__,
                )
                raise

            self._warmed = True
            logger.info(
                "voice_startup stage=local_llm_warmup provider=local model={} "
                "status=ready duration_ms={}",
                self.config.model,
                round((time.monotonic() - started) * 1000, 1),
            )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.client.close()


_runtime: LocalLLMRuntime | None = None
_runtime_lock = threading.Lock()


def get_local_llm_runtime(
    config: LocalLLMConfig | None = None,
) -> LocalLLMRuntime:
    """Return the one process-wide runtime for the active configuration."""
    global _runtime
    config = config or load_local_llm_config()
    with _runtime_lock:
        if _runtime is None or _runtime._closed:
            _runtime = LocalLLMRuntime(config)
        elif _runtime.config != config:
            raise RuntimeError(
                "Local LLM runtime is already initialized with different "
                "settings; restart the backend after changing LOCAL_LLM_*"
            )
        return _runtime


async def warm_local_llm_runtime() -> None:
    await get_local_llm_runtime().warm()


async def shutdown_local_llm_runtime() -> None:
    global _runtime
    with _runtime_lock:
        runtime = _runtime
        _runtime = None
    if runtime is not None:
        await runtime.close()
