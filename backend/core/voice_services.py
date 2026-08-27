import asyncio
import time
from collections.abc import Callable
from typing import Any
from loguru import logger


class VoiceSessionAuthenticationError(PermissionError):
    """Raised when a voice session cannot be bound to an authenticated user."""


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
    *,
    session_hydrator: Callable[[Any], Any] | None = None,
    session_llm_factory: Callable[[Any], Any] | None = None,
):
    """Authenticate, hydrate memory, and construct session-bound services safely."""
    started = time.monotonic()
    session = await session_loader(session_body)
    if session is None:
        raise VoiceSessionAuthenticationError(
            "A valid authenticated voice session token is required"
        )
    if not session_hydrator and not session_llm_factory:
        services = await initialize_voice_services(stt_factory, tts_factory, llm_factory)
    else:
        # Authentication remains the first side effect. Once accepted, overlap
        # optional DB hydration with independent STT/TTS construction, then
        # build the LLM with the resulting durable session instruction.
        stt_task = asyncio.create_task(_construct_voice_service("stt", stt_factory))
        tts_task = asyncio.create_task(_construct_voice_service("tts", tts_factory))
        hydrated_session = session
        if session_hydrator:
            try:
                hydrated_session = await session_hydrator(session)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Optional session memory hydration failed: {!r}", exc
                )
        selected_llm_factory = (
            (lambda: session_llm_factory(hydrated_session))
            if session_llm_factory
            else llm_factory
        )
        llm_task = asyncio.create_task(
            _construct_voice_service("llm", selected_llm_factory)
        )
        try:
            services = tuple(await asyncio.gather(stt_task, tts_task, llm_task))
        except BaseException as exc:
            # The authenticated call exists before provider construction. Keep
            # that identity attached so startup failures can be finalized.
            setattr(exc, "voice_session", hydrated_session)
            raise
        session = hydrated_session
    logger.info(
        "voice_startup stage=runtime_ready duration_ms={}",
        round((time.monotonic() - started) * 1000, 1),
    )
    return services, session
