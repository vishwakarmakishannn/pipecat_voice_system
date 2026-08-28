"""Pipecat boundary that injects release-scoped Mswipe evidence per turn."""

import asyncio
import time

from loguru import logger
from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.llm_context import LLMContext

from core.context_summary import QUERY_SCOPED_CONTEXT_MARKER
from core.knowledge_config import KNOWLEDGE_VOICE_TIMEOUT_SECONDS
from core.log_safety import safe_text_metadata
from services.knowledge.retrieval import (
    format_voice_knowledge_context,
    retrieve_knowledge,
)


class MswipeKnowledgeProcessor(FrameProcessor):
    def __init__(
        self,
        context: LLMContext,
        latency_state=None,
        mutation_epoch=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._context = context
        self._latency_state = latency_state
        self._mutation_epoch = mutation_epoch
        self._dynamic_messages: list[dict] = []
        self._active_task: asyncio.Task | None = None
        self._generation = 0

    def clear_dynamic_context(self) -> None:
        if not self._dynamic_messages:
            return
        ids = {id(message) for message in self._dynamic_messages}
        previous = len(self._context.messages)
        self._context.messages[:] = [
            message for message in self._context.messages if id(message) not in ids
        ]
        self._dynamic_messages.clear()
        if self._mutation_epoch and len(self._context.messages) != previous:
            self._mutation_epoch.bump("mswipe_knowledge_context_cleared")

    def start_user_turn(self) -> None:
        self._generation += 1
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
        self._active_task = None
        self.clear_dynamic_context()

    def _latest_user_text(self) -> str:
        for message in reversed(self._context.messages):
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content
        return ""

    def _install(self, content: str) -> None:
        message = {
            "role": "developer",
            "content": f"{QUERY_SCOPED_CONTEXT_MARKER}\n{content}",
        }
        self._context.add_message(message)
        self._dynamic_messages.append(message)
        if self._mutation_epoch:
            self._mutation_epoch.bump("mswipe_knowledge_context_added")

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM or not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        self.clear_dynamic_context()
        query = self._latest_user_text().strip()
        if not query:
            await self.push_frame(frame, direction)
            return
        generation = self._generation
        started = time.monotonic()
        task = asyncio.create_task(retrieve_knowledge(query))
        self._active_task = task
        try:
            response = await asyncio.wait_for(
                task, timeout=KNOWLEDGE_VOICE_TIMEOUT_SECONDS
            )
            if generation != self._generation:
                return
            context = format_voice_knowledge_context(response)
            if context:
                self._install(context)
                if self._latency_state:
                    self._latency_state.rag_used = True
                    self._latency_state.rag_latency_ms = round(
                        (time.monotonic() - started) * 1000, 1
                    )
            elif response.route.name in {"knowledge", "mixed"}:
                self._install(
                    "MSWIPE_KNOWLEDGE_STATUS: No sufficiently reliable published "
                    "answer was found for this turn. Do not invent Mswipe facts. "
                    "Ask one concise clarifying question or offer escalation. "
                    f"status={response.status}; reason={response.reason}."
                )
            logger.info(
                "mswipe_knowledge route={} status={} confidence={} duration_ms={} query_meta={}",
                response.route.name,
                response.status,
                round(response.confidence, 3),
                round((time.monotonic() - started) * 1000, 1),
                safe_text_metadata(query),
            )
        except TimeoutError:
            if generation == self._generation:
                self._install(
                    "MSWIPE_KNOWLEDGE_STATUS: Retrieval timed out. Do not invent an "
                    "answer; briefly offer to retry or escalate."
                )
            logger.warning(
                "mswipe_knowledge status=timeout budget_ms={} query_meta={}",
                round(KNOWLEDGE_VOICE_TIMEOUT_SECONDS * 1000),
                safe_text_metadata(query),
            )
        except asyncio.CancelledError:
            return
        finally:
            if self._active_task is task:
                self._active_task = None
        if generation == self._generation:
            await self.push_frame(frame, direction)

    async def cleanup(self):
        self.start_user_turn()
        await super().cleanup()
