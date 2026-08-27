"""Dedicated Groq service for safe, out-of-band context summarization."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable

from loguru import logger
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.groq.llm import GroqLLMService, GroqLLMSettings
from pipecat.utils.context.llm_context_summarization import (
    LLMContextSummarizationUtil,
)

from core.context_summary import (
    LIVE_CONVERSATION_SUMMARY_MARKER,
    QUERY_SCOPED_CONTEXT_MARKER,
    SUMMARY_SYSTEM_PROMPT,
)
from core.context_summary_config import VoiceContextSummaryConfig
from core.assistant_output import contains_reserved_tool_markup
from providers.llm.groq_llm import _groq_completion_settings
from providers.llm.groq_runtime import get_shared_groq_client


class StaleContextSummaryError(RuntimeError):
    """Raised when destructive context mutation makes a summary unsafe to apply."""


class ContextSummaryCooldownError(RuntimeError):
    """Raised when a recent provider failure suppresses an immediate retry."""


def sanitize_summary_messages(messages: list[dict]) -> list[dict[str, str]]:
    """Keep conversational meaning while excluding query/tool payloads."""
    sanitized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        content = content.strip()
        if role == "user" and content.startswith(LIVE_CONVERSATION_SUMMARY_MARKER):
            content = content.removeprefix(
                LIVE_CONVERSATION_SUMMARY_MARKER
            ).strip()
        if not content or QUERY_SCOPED_CONTEXT_MARKER in content:
            continue
        if role == "assistant" and contains_reserved_tool_markup(content):
            continue
        sanitized.append({"role": role, "content": content})
    return sanitized


class SafeGroqContextSummaryService(GroqLLMService):
    """Generate summaries on Groq and reject results after destructive edits."""

    supports_developer_role = False

    def create_client(self, api_key=None, base_url=None, **kwargs):
        del kwargs
        return get_shared_groq_client(api_key=api_key, base_url=base_url)

    def __init__(
        self,
        *,
        config: VoiceContextSummaryConfig,
        mutation_epoch_getter: Callable[[], int],
    ) -> None:
        super().__init__(
            api_key=os.getenv("GROQ_API_KEY", ""),
            settings=GroqLLMSettings(
                model=config.model,
                system_instruction=SUMMARY_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=config.target_tokens,
                extra=_groq_completion_settings(config.model),
            ),
        )
        self._summary_config = config
        self._mutation_epoch_getter = mutation_epoch_getter
        self._cooldown_until = 0.0
        self.diagnostic_callback = None

    async def _generate_summary(self, frame) -> tuple[str, int]:
        now = time.monotonic()
        if now < self._cooldown_until:
            raise ContextSummaryCooldownError("Context summarization retry cooldown is active")

        starting_epoch = self._mutation_epoch_getter()
        selection = LLMContextSummarizationUtil.get_messages_to_summarize(
            frame.context, frame.min_messages_to_keep
        )
        messages = sanitize_summary_messages(selection.messages)
        if not messages:
            raise RuntimeError("No safe conversation messages to summarize")

        transcript = LLMContextSummarizationUtil.format_messages_for_summary(messages)
        summary_context = LLMContext(
            messages=[{"role": "user", "content": f"Conversation history:\n{transcript}"}]
        )
        logger.info(
            "voice_context_summary status=started model={} messages={} epoch={}",
            self._summary_config.model,
            len(messages),
            starting_epoch,
        )
        started = time.monotonic()
        try:
            summary = await self.run_inference(
                summary_context,
                max_tokens=frame.target_context_tokens,
                system_instruction=SUMMARY_SYSTEM_PROMPT,
            )
        except asyncio.CancelledError:
            self._cooldown_until = (
                time.monotonic() + self._summary_config.retry_cooldown_seconds
            )
            logger.warning(
                "voice_context_summary status=cancelled model={} duration_ms={}",
                self._summary_config.model,
                round((time.monotonic() - started) * 1000, 1),
            )
            raise
        except Exception as exc:
            self._cooldown_until = (
                time.monotonic() + self._summary_config.retry_cooldown_seconds
            )
            logger.exception(
                "voice_context_summary status=failed model={} duration_ms={}",
                self._summary_config.model,
                round((time.monotonic() - started) * 1000, 1),
            )
            if self.diagnostic_callback:
                self.diagnostic_callback(
                    component="context_summary",
                    code="context_summary.inference_failed",
                    severity="warning",
                    outcome="degraded",
                    safe_message="Live context summarization failed; the active call continued with its existing context.",
                    operator_detail=exc,
                    retryable=True,
                )
            raise

        if self._mutation_epoch_getter() != starting_epoch:
            logger.warning(
                "voice_context_summary status=stale_discarded starting_epoch={} current_epoch={}",
                starting_epoch,
                self._mutation_epoch_getter(),
            )
            raise StaleContextSummaryError(
                "Context changed destructively while the summary was generated"
            )
        if not summary or not summary.strip():
            self._cooldown_until = (
                time.monotonic() + self._summary_config.retry_cooldown_seconds
            )
            raise RuntimeError("Groq returned an empty context summary")
        return summary.strip(), selection.last_summarized_index

    async def close(self) -> None:
        # The process-scoped Groq pool is owned by application lifespan, not a
        # single call's context-summary service.
        return None


def build_context_summary_service(
    config: VoiceContextSummaryConfig,
    mutation_epoch_getter: Callable[[], int],
) -> SafeGroqContextSummaryService | None:
    if not config.enabled:
        return None
    return SafeGroqContextSummaryService(
        config=config,
        mutation_epoch_getter=mutation_epoch_getter,
    )
