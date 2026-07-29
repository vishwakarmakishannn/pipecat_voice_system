import asyncio
import time
from collections.abc import Callable
from typing import Any
from loguru import logger


async def _construct_voice_service(name: str, factory: Callable[[], Any]) -> Any:
    started = time.monotonic()
    service = await asyncio.to_thread(factory)
    logger.info(
        "voice_startup stage=service_constructed service={} duration_ms={}",
        name,
        round((time.monotonic() - started) * 1000, 1),
    )
    return service


async def initialize_voice_services(
    stt_factory: Callable[[], Any],
    tts_factory: Callable[[], Any],
    llm_factory: Callable[[], Any],
) -> tuple[Any, Any, Any]:
    """Construct independent, potentially blocking services concurrently."""
    constructors = [
        asyncio.create_task(_construct_voice_service(name, factory))
        for name, factory in zip(("stt", "tts", "llm"), (stt_factory, tts_factory, llm_factory), strict=True)
    ]
    stt, tts, llm = await asyncio.gather(*constructors)
    return stt, tts, llm


async def initialize_voice_runtime(
    stt_factory: Callable[[], Any],
    tts_factory: Callable[[], Any],
    llm_factory: Callable[[], Any],
    session_loader: Callable[[Any], Any],
    session_body: Any,
):
    """Overlap independent provider construction and session identity I/O."""
    started = time.monotonic()
    services_task = asyncio.create_task(
        initialize_voice_services(stt_factory, tts_factory, llm_factory)
    )
    session_task = asyncio.create_task(session_loader(session_body))
    services, session = await asyncio.gather(services_task, session_task)
    logger.info(
        "voice_startup stage=runtime_ready duration_ms={}",
        round((time.monotonic() - started) * 1000, 1),
    )
    return services, session
