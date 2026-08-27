import asyncio
import os
import time
import uuid

from loguru import logger
from google.genai.types import Candidate, Content, GenerateContentResponse, Part
from pipecat.services.google.llm import GoogleLLMService
from core.llm_config import (
    first_token_timeout_seconds,
    google_hedge_delay_seconds,
    google_warmup_timeout_seconds,
    timeout_recovery_text,
    total_timeout_seconds,
)
from core.prompt_config import load_system_prompt
from core.tool_config import tool_timeout_seconds


class FirstTokenTimeoutError(TimeoutError):
    """Raised when a live LLM stream produces no meaningful first chunk."""


class LatencyBoundGoogleLLMService(GoogleLLMService):
    def __init__(
        self,
        *,
        first_token_timeout_secs: float,
        total_timeout_secs: float,
        hedge_delay_secs: float,
        warmup_timeout_secs: float,
        timeout_message: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._first_token_timeout_secs = first_token_timeout_secs
        self._total_timeout_secs = total_timeout_secs
        self._hedge_delay_secs = min(
            hedge_delay_secs,
            max(0.1, first_token_timeout_secs - 0.1),
        )
        self._warmup_timeout_secs = warmup_timeout_secs
        self._timeout_message = timeout_message
        self._connection_warmed = False
        self.diagnostic_callback = None
        self.phase_callback = None

    def _emit_phase(self, phase: str, **details) -> None:
        callback = getattr(self, "phase_callback", None)
        if callback:
            try:
                callback(phase=phase, **details)
            except Exception as exc:
                logger.debug(
                    "voice_llm provider=google status=phase_callback_failed "
                    "phase={} error_type={}",
                    phase,
                    type(exc).__name__,
                )

    @property
    def connection_warmed(self) -> bool:
        return self._connection_warmed

    async def warm_connection(self, timeout_seconds: float | None = None) -> bool:
        """Warm DNS/TLS/auth/model lookup on the same client used for calls."""
        timeout = timeout_seconds or self._warmup_timeout_secs
        started = time.monotonic()
        try:
            await asyncio.wait_for(
                self._client.aio.models.get(model=self._settings.model),
                timeout=timeout,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "voice_llm provider=google model={} status=warmup_failed "
                "duration_ms={} error_type={}",
                self._settings.model,
                round((time.monotonic() - started) * 1000, 1),
                type(exc).__name__,
            )
            return False
        self._connection_warmed = True
        self._emit_phase("connection_warmed")
        logger.info(
            "voice_llm provider=google model={} status=warmed duration_ms={}",
            self._settings.model,
            round((time.monotonic() - started) * 1000, 1),
        )
        return True

    @staticmethod
    def _chunk_has_output(chunk) -> bool:
        for candidate in getattr(chunk, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                if (
                    getattr(part, "text", None)
                    or getattr(part, "function_call", None)
                    or getattr(part, "inline_data", None)
                ):
                    return True
        return False

    @classmethod
    async def _first_output_stream(cls, stream, timeout_secs: float):
        iterator = stream.__aiter__()
        buffered = []
        deadline = asyncio.get_running_loop().time() + timeout_secs
        try:
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError
                try:
                    chunk = await asyncio.wait_for(anext(iterator), timeout=remaining)
                except StopAsyncIteration as exc:
                    raise FirstTokenTimeoutError(
                        "Google stream ended before meaningful output"
                    ) from exc
                buffered.append(chunk)
                if cls._chunk_has_output(chunk):
                    break
        except asyncio.CancelledError:
            if hasattr(iterator, "aclose"):
                await iterator.aclose()
            raise
        except FirstTokenTimeoutError:
            if hasattr(iterator, "aclose"):
                await iterator.aclose()
            raise
        except TimeoutError as exc:
            if hasattr(iterator, "aclose"):
                await iterator.aclose()
            raise FirstTokenTimeoutError(
                f"Google produced no first output within {timeout_secs:.2f}s"
            ) from exc

        for item in buffered:
            yield item
        async for chunk in iterator:
            yield chunk

    @staticmethod
    def _text_chunk(text: str) -> GenerateContentResponse:
        return GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(role="model", parts=[Part.from_text(text=text)]),
                )
            ]
        )

    @classmethod
    async def _recovering_stream(
        cls,
        stream,
        timeout_secs: float,
        timeout_message: str,
        *,
        request_id: str = "unknown",
        provider_model: str = "google",
        started_at: float | None = None,
        diagnostic_callback=None,
        total_timeout_secs: float | None = None,
    ):
        started = time.monotonic() if started_at is None else started_at
        first_chunk_seen = False
        try:
            async def consume():
                nonlocal first_chunk_seen
                async for chunk in cls._first_output_stream(stream, timeout_secs):
                    if not first_chunk_seen:
                        first_chunk_seen = True
                        logger.info(
                            "voice_llm request_id={} provider=google model={} status=first_output "
                            "latency_ms={} provider_response_id={}",
                            request_id,
                            provider_model,
                            round((time.monotonic() - started) * 1000, 1),
                            getattr(chunk, "response_id", None),
                        )
                    yield chunk

            if total_timeout_secs is None:
                async for chunk in consume():
                    yield chunk
            else:
                async with asyncio.timeout(total_timeout_secs):
                    async for chunk in consume():
                        yield chunk
        except FirstTokenTimeoutError:
            duration_ms = round((time.monotonic() - started) * 1000, 1)
            logger.warning(
                "voice_llm request_id={} provider=google model={} status=first_token_timeout "
                "latency_ms={} budget_ms={} action=spoken_recovery",
                request_id,
                provider_model,
                duration_ms,
                round(timeout_secs * 1000),
            )
            if diagnostic_callback:
                diagnostic_callback(
                    component="llm",
                    code="llm.first_output_timeout",
                    severity="error",
                    outcome="recovered",
                    safe_message="The language model did not produce output before the first-output deadline.",
                    request_id=request_id,
                    duration_ms=duration_ms,
                    retryable=True,
                    recovered=True,
                    fatal=False,
                    details={"spoken_recovery": timeout_message},
                )
            yield cls._text_chunk(timeout_message)
        except TimeoutError:
            duration_ms = round((time.monotonic() - started) * 1000, 1)
            logger.warning(
                "voice_llm request_id={} provider=google model={} status=total_timeout "
                "latency_ms={} action=spoken_recovery",
                request_id,
                provider_model,
                duration_ms,
            )
            if diagnostic_callback:
                diagnostic_callback(
                    component="llm",
                    code="llm.total_timeout",
                    severity="error",
                    outcome="recovered",
                    safe_message="The language model exceeded its total response deadline.",
                    request_id=request_id,
                    duration_ms=duration_ms,
                    retryable=True,
                    recovered=True,
                    fatal=False,
                    details={"spoken_recovery": timeout_message},
                )
            yield cls._text_chunk(timeout_message)

    @classmethod
    async def _recovery_stream(
        cls,
        timeout_message: str,
    ):
        """Return a valid stream when request creation itself times out."""
        yield cls._text_chunk(timeout_message)

    async def _prefetched_attempt(
        self,
        context,
        *,
        attempt: int,
        request_id: str,
        deadline: float,
    ):
        """Create one request and return a replayable stream after first output."""
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise FirstTokenTimeoutError("Google first-output deadline elapsed")
        attempt_started = time.monotonic()
        logger.info(
            "voice_llm request_id={} provider=google model={} status=attempt_started "
            "attempt={} budget_ms={}",
            request_id,
            self._settings.model,
            attempt,
            round(remaining * 1000),
        )
        self._emit_phase("attempt_started", attempt=attempt)
        try:
            async with asyncio.timeout(remaining):
                stream = await super()._stream_content(context)
                guarded = self._first_output_stream(
                    stream,
                    max(0.001, deadline - asyncio.get_running_loop().time()),
                )
                first = await anext(guarded)
        except TimeoutError as exc:
            raise FirstTokenTimeoutError(
                f"Google attempt {attempt} produced no output before the deadline"
            ) from exc

        async def replay():
            try:
                yield first
                async for chunk in guarded:
                    yield chunk
            finally:
                if hasattr(guarded, "aclose"):
                    await guarded.aclose()

        logger.info(
            "voice_llm request_id={} provider=google model={} status=attempt_ready "
            "attempt={} latency_ms={}",
            request_id,
            self._settings.model,
            attempt,
            round((time.monotonic() - attempt_started) * 1000, 1),
        )
        self._emit_phase("attempt_ready", attempt=attempt)
        return replay()

    @staticmethod
    async def _close_attempt(task: asyncio.Task) -> None:
        if not task.done():
            task.cancel()
        results = await asyncio.gather(task, return_exceptions=True)
        if results and not isinstance(results[0], BaseException):
            stream = results[0]
            if hasattr(stream, "aclose"):
                await stream.aclose()

    async def _select_hedged_stream(self, context, request_id: str, deadline: float):
        """Race at most two pre-output attempts and return exactly one winner."""
        first = asyncio.create_task(
            self._prefetched_attempt(
                context,
                attempt=1,
                request_id=request_id,
                deadline=deadline,
            )
        )
        tasks: dict[asyncio.Task, int] = {first: 1}
        failures: list[dict] = []
        hedge_delay = min(
            getattr(self, "_hedge_delay_secs", 2.0),
            max(0.001, deadline - asyncio.get_running_loop().time()),
        )
        done, _ = await asyncio.wait(
            tasks,
            timeout=hedge_delay,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            second = asyncio.create_task(
                self._prefetched_attempt(
                    context,
                    attempt=2,
                    request_id=request_id,
                    deadline=deadline,
                )
            )
            tasks[second] = 2

        try:
            while tasks:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                done, _ = await asyncio.wait(
                    tasks,
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    break
                for completed in done:
                    attempt = tasks.pop(completed)
                    try:
                        winner = completed.result()
                    except Exception as exc:
                        failures.append({
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                        })
                        if attempt == 1 and not tasks:
                            second = asyncio.create_task(
                                self._prefetched_attempt(
                                    context,
                                    attempt=2,
                                    request_id=request_id,
                                    deadline=deadline,
                                )
                            )
                            tasks[second] = 2
                        continue
                    losers = list(tasks)
                    for loser in losers:
                        tasks.pop(loser, None)
                    await asyncio.gather(
                        *(self._close_attempt(loser) for loser in losers),
                        return_exceptions=True,
                    )
                    logger.info(
                        "voice_llm request_id={} provider=google model={} "
                        "status=attempt_selected attempt={} hedged={}",
                        request_id,
                        self._settings.model,
                        attempt,
                        attempt == 2 or bool(losers),
                    )
                    self._emit_phase(
                        "attempt_selected",
                        attempt=attempt,
                        hedged=attempt == 2 or bool(losers),
                    )
                    return winner, attempt, failures
        finally:
            if tasks:
                await asyncio.gather(
                    *(self._close_attempt(task) for task in list(tasks)),
                    return_exceptions=True,
                )
        raise FirstTokenTimeoutError("Google produced no meaningful first output")

    async def _hedged_recovering_stream(
        self,
        context,
        *,
        request_id: str,
        provider_model: str,
        started: float,
    ):
        first_deadline = asyncio.get_running_loop().time() + self._first_token_timeout_secs
        try:
            stream, winner_attempt, failures = await self._select_hedged_stream(
                context,
                request_id,
                first_deadline,
            )
            elapsed = time.monotonic() - started
            remaining_first = max(0.001, self._first_token_timeout_secs - elapsed)
            total_remaining = max(
                0.001,
                getattr(self, "_total_timeout_secs", 30.0) - elapsed,
            )
            async for chunk in self._recovering_stream(
                stream,
                remaining_first,
                self._timeout_message,
                request_id=request_id,
                provider_model=provider_model,
                started_at=started,
                diagnostic_callback=getattr(self, "diagnostic_callback", None),
                total_timeout_secs=total_remaining,
            ):
                yield chunk
        except FirstTokenTimeoutError:
            duration_ms = round((time.monotonic() - started) * 1000, 1)
            self._emit_phase("recovered", reason="first_output_timeout")
            callback = getattr(self, "diagnostic_callback", None)
            if callback:
                callback(
                    component="llm",
                    code="llm.first_output_timeout",
                    severity="error",
                    outcome="recovered",
                    safe_message="The language model did not produce output before the first-output deadline.",
                    request_id=request_id,
                    duration_ms=duration_ms,
                    retryable=True,
                    recovered=True,
                    fatal=False,
                    details={
                        "spoken_recovery": self._timeout_message,
                        "attempt_count": 2,
                        "hedged": True,
                    },
                )
            yield self._text_chunk(self._timeout_message)

    async def _stream_content(self, context):
        request_id = uuid.uuid4().hex
        provider_model = self._settings.model
        started = time.monotonic()
        logger.info(
            "voice_llm request_id={} provider=google model={} status=started "
            "deadline_ms={}",
            request_id,
            provider_model,
            round(self._first_token_timeout_secs * 1000),
        )
        return self._hedged_recovering_stream(
            context,
            request_id=request_id,
            provider_model=provider_model,
            started=started,
        )


def get_google_llm(*, system_instruction: str | None = None):
    timeout_secs = first_token_timeout_seconds()
    return LatencyBoundGoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        first_token_timeout_secs=timeout_secs,
        total_timeout_secs=total_timeout_seconds(),
        hedge_delay_secs=google_hedge_delay_seconds(),
        warmup_timeout_secs=google_warmup_timeout_seconds(),
        timeout_message=timeout_recovery_text(),
        settings=GoogleLLMService.Settings(
            model=os.getenv("GOOGLE_MODEL", "gemini-3.1-flash-lite"),
            system_instruction=system_instruction or load_system_prompt(),
            thinking=GoogleLLMService.ThinkingConfig(
                thinking_level=os.getenv("GOOGLE_THINKING_LEVEL", "minimal"),
                include_thoughts=False
            ),
        ),
        function_call_timeout_secs=tool_timeout_seconds(),
        enable_async_tool_cancellation=True,
    )
