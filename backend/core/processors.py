import asyncio
from array import array
from collections.abc import Callable
import json
import re
import sys
import time
from dataclasses import dataclass, field
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMContextFrame,
    LLMTextFrame,
    OutputTransportMessageFrame,
    TranscriptionFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallCancelFrame,
    InterruptionFrame,
    InterimTranscriptionFrame,
    TTSSpeakFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.llm_context import LLMContext
from services.memory import (
    build_turn_memory_context,
    embed_text,
    is_recall_query,
    save_conversation_message,
)
from services.rag import build_rag_context_with_payload, is_rag_query
from services.rag import should_attempt_rag_retrieval
from core.rag_config import (
    RAG_VOICE_MEMORY_TIMEOUT_SECONDS,
    RAG_VOICE_RAG_TIMEOUT_SECONDS,
    RAG_VOICE_RETRIEVAL_TIMEOUT_SECONDS,
    RAG_VOICE_WEB_TIMEOUT_SECONDS,
)
from core.tool_config import tool_filler_delay_seconds, tool_filler_enabled
from core.audio_config import (
    trim_tts_leading_silence,
    tts_silence_preroll_ms,
    tts_silence_threshold,
)
from core.realtime_gate import realtime_turn_gate


def transport_server_message(message_type: str, payload: dict) -> OutputTransportMessageFrame:
    """Build the custom RTVI transport envelope consumed by the web client."""
    return OutputTransportMessageFrame({
        "label": "rtvi-ai",
        "type": "server-message",
        "data": {"type": message_type, "payload": payload},
    })


@dataclass
class TurnLatencyState:
    _gate_token: object = field(default_factory=object, repr=False)
    llm_warm_state_getter: Callable[[], bool] | None = field(
        default=None, repr=False
    )
    session_id: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    tts_provider: str | None = None
    tts_model: str | None = None
    llm_connection_warmed: bool = False
    turn_id: int = 0
    started_at: float | None = None
    first_llm_seen: bool = False
    first_speakable_text_seen: bool = False
    first_audio_seen: bool = False
    tool_used: bool = False
    rag_used: bool = False
    tool_filler_spoken: bool = False
    first_llm_ms: float | None = None
    first_speakable_text_ms: float | None = None
    active: bool = False
    speech_turn_open: bool = False
    turn_identity_open: bool = False
    speech_started_at: float | None = None
    speech_stopped_at: float | None = None
    audio_speech_stopped_at: float | None = None
    final_stt_at: float | None = None
    stage_times: dict[str, float] | None = None
    interim_stt_count: int = 0
    final_stt_fragment_count: int = 0
    rag_considered: bool = False
    rag_bypassed: bool = False
    response_finished: bool = False
    cancelled: bool = False

    def __post_init__(self):
        self.stage_times = {}

    @property
    def priority_key(self) -> tuple[object, int]:
        # Keep a strongly referenced token in the global gate. ``id(self)`` can
        # be reused after a state is collected and accidentally collide with a
        # leaked older turn.
        return (self._gate_token, self.turn_id)

    def mark_user_started(self):
        if self.speech_turn_open:
            return
        self.turn_id += 1
        # Response-relative timing is assigned at final STT. Clearing it here
        # prevents this speech event from inheriting the previous turn origin.
        self.started_at = None
        self.active = False
        self.speech_turn_open = True
        self.turn_identity_open = True
        self.speech_started_at = time.monotonic()
        self.speech_stopped_at = None
        self.audio_speech_stopped_at = None
        self.final_stt_at = None
        self.interim_stt_count = 0
        self.final_stt_fragment_count = 0
        self.stage_times = {"user_started": self.speech_started_at}
        self.emit("user_started")

    def mark_user_stopped(self):
        if not self.speech_turn_open and not self.active:
            self.mark_user_started()
        self.speech_stopped_at = time.monotonic()
        self.mark_stage("user_stopped", self.speech_stopped_at)
        self.emit("user_stopped")
        self.speech_turn_open = False

    def mark_vad_user_stopped(self, stop_secs: float) -> None:
        """Record the estimated last speech sample before VAD confirmation."""
        self.audio_speech_stopped_at = time.monotonic() - max(0.0, stop_secs)
        self.mark_stage("audio_speech_stopped", self.audio_speech_stopped_at)

    def mark_stage(self, stage: str, at: float | None = None):
        if self.stage_times is None:
            self.stage_times = {}
        self.stage_times.setdefault(stage, time.monotonic() if at is None else at)

    def start_turn(self):
        if self.active:
            # Text input can arrive after the previous answer has produced
            # audio even if its completion frame did not close this state.
            # Once audio was measured, a later context is a new user turn.
            if not self.first_audio_seen:
                return
            self.finish_turn()
        if not self.turn_identity_open:
            # Text-only/no-VAD transports still need a stable turn identity.
            self.turn_id += 1
            self.turn_identity_open = True
            # Unlike voice, text input has no TranscriptionFrame to replace
            # these values. Clear every response origin when opening its new
            # identity so timing cannot inherit the preceding text turn.
            self.started_at = None
            self.speech_started_at = None
            self.speech_stopped_at = None
            self.audio_speech_stopped_at = None
            self.final_stt_at = None
            self.stage_times = {}
        self.active = True
        self.started_at = self.final_stt_at or time.monotonic()
        if self.final_stt_at is None:
            self.final_stt_at = self.started_at
            self.mark_stage("final_stt", self.final_stt_at)
        self.first_llm_seen = False
        self.first_speakable_text_seen = False
        self.first_audio_seen = False
        self.tool_used = False
        self.rag_used = False
        self.tool_filler_spoken = False
        self.first_llm_ms = None
        self.first_speakable_text_ms = None
        if self.llm_warm_state_getter:
            self.llm_connection_warmed = bool(self.llm_warm_state_getter())
        self.rag_considered = False
        self.rag_bypassed = False
        self.response_finished = False
        self.cancelled = False
        self.mark_stage("turn_ready")
        realtime_turn_gate.begin(self.priority_key)
        self.emit("final_stt")

    def record_interim_stt(self):
        self.interim_stt_count += 1
        now = time.monotonic()
        if self.stage_times is None:
            self.stage_times = {}
        self.stage_times.setdefault("first_interim_stt", now)
        self.stage_times["latest_interim_stt"] = now
        self.emit("interim_stt")

    def record_final_stt_fragment(self):
        """Record the latest final fragment without opening response processing."""
        if not self.turn_identity_open:
            # Text-only/no-VAD transports still need a stable turn identity.
            self.turn_id += 1
            self.turn_identity_open = True
            self.stage_times = {}
        self.final_stt_at = time.monotonic()
        self.final_stt_fragment_count += 1
        if self.stage_times is None:
            self.stage_times = {}
        self.stage_times["final_stt"] = self.final_stt_at
        self.emit("final_stt_fragment")

    def finish_response(self):
        """Close LLM processing without releasing work ahead of first audio."""
        self.response_finished = True
        self.active = False
        self.speech_turn_open = False
        if self.first_audio_seen:
            realtime_turn_gate.end(self.priority_key)
            self.turn_identity_open = False

    def finish_turn(self):
        """Force-close a cancelled/no-audio turn and release its gate."""
        realtime_turn_gate.end(self.priority_key)
        self.active = False
        self.speech_turn_open = False
        self.turn_identity_open = False
        self.cancelled = True

    def telemetry_payload(self) -> dict:
        origin = self.speech_stopped_at or self.final_stt_at or self.started_at
        stages_ms = {}
        if origin is not None:
            stages_ms = {
                name: round((timestamp - origin) * 1000, 1)
                for name, timestamp in (self.stage_times or {}).items()
            }
        speech_ms = None
        if self.speech_started_at is not None and self.speech_stopped_at is not None:
            speech_ms = round((self.speech_stopped_at - self.speech_started_at) * 1000, 1)
        return {
            "session_id": self.session_id,
            "basis": "user_stopped" if self.speech_stopped_at is not None else "final_stt",
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "tts_provider": self.tts_provider,
            "tts_model": self.tts_model,
            "llm_connection_warmed": self.llm_connection_warmed,
            "speech_ms": speech_ms,
            "interim_stt_count": self.interim_stt_count,
            "final_stt_fragment_count": self.final_stt_fragment_count,
            "stages_ms": stages_ms,
            "server_emitted_unix_ms": round(time.time() * 1000),
        }

    def emit(self, stage: str):
        if stage in {"user_started", "user_stopped"}:
            origin = self.speech_started_at
        else:
            origin = self.started_at
        elapsed = None if origin is None else round((time.monotonic() - origin) * 1000, 1)
        logger.info(
            "voice_latency session={} turn={} stage={} elapsed_ms={}",
            self.session_id, self.turn_id, stage, elapsed,
        )


class LatencyBoundaryProcessor(FrameProcessor):
    def __init__(self, state: TurnLatencyState, boundary: str, **kwargs):
        super().__init__(**kwargs)
        self._state = state
        self._boundary = boundary

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        telemetry_frame = None
        if direction == FrameDirection.DOWNSTREAM:
            if self._boundary == "turn" and isinstance(frame, UserStartedSpeakingFrame):
                self._state.mark_user_started()
            elif self._boundary == "turn" and isinstance(frame, UserStoppedSpeakingFrame):
                self._state.mark_user_stopped()
            elif self._boundary == "vad" and isinstance(frame, VADUserStartedSpeakingFrame):
                self._state.mark_user_started()
            elif self._boundary == "vad" and isinstance(frame, VADUserStoppedSpeakingFrame):
                self._state.mark_vad_user_stopped(frame.stop_secs)
            elif self._boundary == "turn" and isinstance(frame, LLMContextFrame):
                self._state.start_turn()
            elif self._boundary == "turn" and isinstance(frame, InterruptionFrame):
                self._state.finish_turn()
            elif self._boundary == "stt" and isinstance(frame, TranscriptionFrame):
                self._state.record_final_stt_fragment()
            elif self._boundary == "stt" and isinstance(frame, InterimTranscriptionFrame):
                self._state.record_interim_stt()
            elif self._boundary == "llm" and isinstance(frame, LLMTextFrame):
                if not self._state.first_llm_seen:
                    self._state.first_llm_seen = True
                    self._state.first_llm_ms = round((time.monotonic() - self._state.started_at) * 1000, 1) if self._state.started_at else None
                    self._state.mark_stage("first_llm_text")
                    self._state.emit("first_llm_text")
                if (
                    not self._state.first_speakable_text_seen
                    and any(character.isalnum() for character in frame.text)
                ):
                    self._state.first_speakable_text_seen = True
                    self._state.first_speakable_text_ms = round(
                        (time.monotonic() - self._state.started_at) * 1000, 1
                    ) if self._state.started_at else None
                    self._state.mark_stage("first_speakable_text")
                    self._state.emit("first_speakable_text")
            elif (
                self._boundary == "tts"
                and isinstance(frame, TTSStartedFrame)
                and self._state.first_llm_seen
            ):
                self._state.mark_stage("tts_request_started")
                self._state.emit("tts_request_started")
            elif (
                self._boundary == "tts"
                and isinstance(frame, TTSAudioRawFrame)
                and self._state.first_llm_seen
                and not self._state.first_audio_seen
                and not self._state.cancelled
            ):
                self._state.first_audio_seen = True
                realtime_turn_gate.end(self._state.priority_key)
                self._state.mark_stage("first_tts_audio")
                self._state.emit("first_tts_audio")
                audio_at = time.monotonic()
                response_origin = (
                    self._state.audio_speech_stopped_at
                    or self._state.speech_stopped_at
                    or self._state.final_stt_at
                    or self._state.started_at
                )
                total_ms = round((audio_at - response_origin) * 1000, 1) if response_origin else None
                final_stt_to_audio_ms = round(
                    (audio_at - self._state.final_stt_at) * 1000, 1
                ) if self._state.final_stt_at else None
                stages = self._state.stage_times or {}
                speakable_at = stages.get("first_speakable_text")
                tts_request_at = stages.get("tts_request_started")

                def stage_delta(start, end):
                    return round((end - start) * 1000, 1) if start and end else None

                category = "tool" if self._state.tool_used else "rag" if self._state.rag_used else "direct"
                telemetry_frame = OutputTransportMessageFrame({
                        "label": "rtvi-ai",
                        "type": "server-message",
                        "data": {
                            "type": "latency_stats",
                            "payload": {
                                "turn_id": self._state.turn_id,
                                "category": category,
                                "with_tools": self._state.tool_used,
                                "rag_used": self._state.rag_used,
                                "rag_considered": self._state.rag_considered,
                                "rag_bypassed": self._state.rag_bypassed,
                                "llm_ms": self._state.first_llm_ms,
                                "speakable_text_ms": self._state.first_speakable_text_ms,
                                "tts_aggregation_ms": stage_delta(speakable_at, tts_request_at),
                                "tts_provider_ms": stage_delta(tts_request_at, audio_at),
                                "speakable_to_audio_ms": stage_delta(speakable_at, audio_at),
                                "answer_audio_ms": total_ms,
                                "final_stt_to_audio_ms": final_stt_to_audio_ms,
                                **self._state.telemetry_payload(),
                            },
                        },
                    })
                if self._state.response_finished:
                    self._state.turn_identity_open = False
            elif (
                self._boundary == "tts"
                and isinstance(frame, TTSStoppedFrame)
                and self._state.first_llm_seen
                and not self._state.first_audio_seen
            ):
                # Empty, cancelled, or provider-failed responses must never
                # leave background enrichment gated forever.
                self._state.finish_turn()
        await self.push_frame(frame, direction)
        if telemetry_frame:
            await self.push_frame(telemetry_frame, direction)


class LeadingSilenceTrimmerProcessor(FrameProcessor):
    """Remove initial 16-bit PCM silence while preserving a small speech preroll."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        threshold: int | None = None,
        preroll_ms: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._enabled = trim_tts_leading_silence() if enabled is None else enabled
        self._threshold = tts_silence_threshold() if threshold is None else threshold
        self._preroll_ms = tts_silence_preroll_ms() if preroll_ms is None else preroll_ms
        self._buffers: dict[str | None, bytearray] = {}
        self._audible_contexts: set[str | None] = set()

    def _reset_context(self, context_id: str | None) -> None:
        self._buffers.pop(context_id, None)
        self._audible_contexts.discard(context_id)

    def _trim_frame(self, frame: TTSAudioRawFrame) -> bool:
        context_id = frame.context_id
        if context_id in self._audible_contexts:
            return True

        buffer = self._buffers.setdefault(context_id, bytearray())
        audio = frame.audio
        first_audible_byte = None
        # Decode the PCM buffer in one C-level operation. The former loop made
        # a new two-byte slice and integer for every sample on the event loop.
        samples = array("h")
        samples.frombytes(audio[: len(audio) & ~1])
        if sys.byteorder != "little":
            samples.byteswap()
        for sample_index, sample in enumerate(samples):
            if abs(sample) > self._threshold:
                first_audible_byte = sample_index * 2
                break

        preroll_bytes = int(
            frame.sample_rate * frame.num_channels * 2 * self._preroll_ms / 1000
        )
        if first_audible_byte is None:
            buffer.extend(audio)
            if preroll_bytes == 0:
                buffer.clear()
            elif len(buffer) > preroll_bytes:
                del buffer[:-preroll_bytes]
            return False

        combined = bytes(buffer) + audio
        speech_offset = len(buffer) + first_audible_byte
        start = max(0, speech_offset - preroll_bytes)
        frame.audio = combined[start:]
        frame.num_frames = len(frame.audio) // (frame.num_channels * 2)
        self._buffers.pop(context_id, None)
        self._audible_contexts.add(context_id)
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM or not self._enabled:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TTSStartedFrame):
            self._reset_context(frame.context_id)
            self._buffers[frame.context_id] = bytearray()
        elif isinstance(frame, TTSAudioRawFrame) and not self._trim_frame(frame):
            return
        elif isinstance(frame, TTSStoppedFrame):
            self._reset_context(frame.context_id)
        elif isinstance(frame, InterruptionFrame):
            self._buffers.clear()
            self._audible_contexts.clear()

        await self.push_frame(frame, direction)


class BoundedContextProcessor(FrameProcessor):
    def __init__(
        self,
        context: LLMContext,
        protected_messages: list[dict] | None = None,
        max_messages: int = 24,
        max_chars: int = 18000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._context = context
        self._protected_ids = {id(message) for message in (protected_messages or [])}
        self._max_messages = max(2, max_messages)
        self._max_chars = max(1000, max_chars)

    def protect_messages(self, messages: list[dict]) -> None:
        self._protected_ids.update(id(message) for message in messages)

    @staticmethod
    def _message_chars(message) -> int:
        content = message.get("content", "") if isinstance(message, dict) else str(message)
        if isinstance(content, str):
            return len(content)
        try:
            return len(json.dumps(content, default=str))
        except (TypeError, ValueError):
            return len(str(content))

    def trim(self) -> int:
        messages = self._context.messages
        protected = [message for message in messages if id(message) in self._protected_ids]
        candidates = [message for message in messages if id(message) not in self._protected_ids]

        groups: list[list] = []
        current: list = []
        for message in candidates:
            role = message.get("role") if isinstance(message, dict) else None
            if role == "user" and current:
                groups.append(current)
                current = []
            current.append(message)
        if current:
            groups.append(current)

        selected_groups: list[list] = []
        selected_count = len(protected)
        selected_chars = sum(self._message_chars(message) for message in protected)
        for group in reversed(groups):
            group_count = len(group)
            group_chars = sum(self._message_chars(message) for message in group)
            if selected_groups and (
                selected_count + group_count > self._max_messages
                or selected_chars + group_chars > self._max_chars
            ):
                break
            selected_groups.append(group)
            selected_count += group_count
            selected_chars += group_chars

        selected = protected + [
            message
            for group in reversed(selected_groups)
            for message in group
        ]
        removed = len(messages) - len(selected)
        if removed:
            messages[:] = selected
            logger.info(
                "voice_context status=trimmed removed={} retained={} chars={}",
                removed,
                len(selected),
                selected_chars,
            )
        return removed

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMContextFrame):
            self.trim()
        await self.push_frame(frame, direction)


class ToolRoutingProcessor(FrameProcessor):
    SEARCH_PATTERNS = (
        re.compile(r"\b(?:search|browse)(?:\s+(?:the\s+)?(?:web|internet|online))?\b"),
        re.compile(r"\blook\s+up\b"),
        re.compile(r"\b(?:latest|today(?:'s)?|news|weather|stock\s+price)\b"),
        re.compile(r"\bcurrent\s+(?:price|weather|president|prime\s+minister|events?|news)\b"),
    )
    ISSUE_PATTERNS = (
        re.compile(r"\b(?:raise|create|open|file)\s+(?:an?\s+)?(?:issue|ticket|complaint)\b"),
        re.compile(r"\breport\s+(?:a\s+)?problem\b"),
    )
    ISSUE_WORKFLOW_PATTERNS = (
        # These must describe the issue-collection workflow, not merely contain
        # contact details. RAG answers can legitimately mention an email or ID.
        re.compile(
            r"\b(?:provide|share|send|need|require|enter|confirm)\b.*"
            r"\b(?:email|e-mail|mobile|phone|customer\s*(?:id|identifier)|"
            r"device\s*(?:id|identifier))\b"
        ),
        re.compile(r"\b(?:raise|create|open|file|log)\b.*\b(?:issue|ticket|complaint)\b"),
        re.compile(r"\b(?:issue|ticket|complaint)\b.*\b(?:raise|create|open|file|log|proceed|confirm)\b"),
    )
    ISSUE_CANCEL_PATTERN = re.compile(
        r"\b(?:cancel|never\s+mind|nevermind|do\s+not|don't|stop)\b"
    )
    ISSUE_SUCCESS_PATTERN = re.compile(
        r"\b(?:issue|ticket|complaint)(?:\s*#?\d+)?\b"
        r".*\b(?:successfully\s+)?(?:raised|created|opened|filed|logged)\b"
    )
    ISSUE_CONTINUATION_PATTERNS = (
        re.compile(r"\b(?:yes|yeah|yep|sure|proceed|confirm|go\s+ahead|please\s+do)\b"),
        re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b"),
        re.compile(r"\b\d{6,}\b"),
        re.compile(r"\b[a-z]{1,8}\d{4,}\b"),
    )

    @classmethod
    def needs_web_search(cls, text: str) -> bool:
        normalized = " ".join((text or "").lower().split())
        return any(pattern.search(normalized) for pattern in cls.SEARCH_PATTERNS)

    @classmethod
    def needs_issue_tool(cls, text: str) -> bool:
        normalized = " ".join((text or "").lower().split())
        return any(pattern.search(normalized) for pattern in cls.ISSUE_PATTERNS)

    def __init__(
        self,
        context: LLMContext,
        search_tool,
        issue_tool,
        retrieval=None,
        latency_state: TurnLatencyState | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._context = context
        self._search_tool = search_tool
        self._issue_tool = issue_tool
        self._retrieval = retrieval
        self._latency_state = latency_state
        self._issue_workflow_active = False

    @staticmethod
    def _latest_user_text(context: LLMContext) -> str:
        for message in reversed(context.messages):
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content", "")
                return content if isinstance(content, str) else json.dumps(content, default=str)
        return ""

    @staticmethod
    def _message_text(message) -> str:
        if not isinstance(message, dict):
            return ""
        content = message.get("content", "")
        return content if isinstance(content, str) else json.dumps(content, default=str)

    def _issue_workflow_is_active(self, latest_user_text: str) -> bool:
        normalized_user = " ".join((latest_user_text or "").lower().split())
        if self.ISSUE_CANCEL_PATTERN.search(normalized_user):
            self._issue_workflow_active = False
            return False
        if self.needs_issue_tool(normalized_user):
            self._issue_workflow_active = True
            return True

        # Do not carry a pending complaint into an unrelated new question.
        # Only inspect historical issue prompts when the current utterance
        # itself looks like a field value or explicit confirmation.
        if not any(
            pattern.search(normalized_user)
            for pattern in self.ISSUE_CONTINUATION_PATTERNS
        ):
            self._issue_workflow_active = False
            return False

        # Continuations are commonly just an email, ID, "yes", or "proceed".
        # Recover the intent from recent assistant field/confirmation prompts.
        for message in reversed(self._context.messages[-16:]):
            role = message.get("role") if isinstance(message, dict) else None
            text = " ".join(self._message_text(message).lower().split())
            if not text:
                continue
            if role in {"assistant", "tool", "function"} and self.ISSUE_SUCCESS_PATTERN.search(text):
                self._issue_workflow_active = False
                return False
            if role == "assistant" and any(
                pattern.search(text) for pattern in self.ISSUE_WORKFLOW_PATTERNS
            ):
                self._issue_workflow_active = True
                return True
            if role == "user" and self.needs_issue_tool(text):
                self._issue_workflow_active = True
                return True
        return self._issue_workflow_active

    def route(self) -> list:
        text = self._latest_user_text(self._context).lower()
        tools = []
        web_search_already_attempted = bool(
            self._retrieval and self._retrieval.web_search_attempted
        )
        if self.needs_web_search(text) and not web_search_already_attempted:
            tools.append(self._search_tool)
        issue_workflow_active = self._issue_workflow_is_active(text)
        if issue_workflow_active:
            tools.append(self._issue_tool)
        self._context.set_tools(tools)
        logger.info(
            "voice_tools exposed={} issue_workflow_active={} query={!r}",
            [getattr(tool, "__name__", str(tool)) for tool in tools],
            issue_workflow_active,
            text[:120],
        )
        return tools

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMContextFrame):
            tools = self.route()
            if tools:
                if self._latency_state:
                    self._latency_state.tool_used = True
                    self._latency_state.mark_stage("tool_routed")
        await self.push_frame(frame, direction)

class ConversationMemoryProcessor(FrameProcessor):
    def __init__(self, conversation_id: int | None, capture: str, **kwargs):
        super().__init__(**kwargs)
        self._conversation_id = conversation_id
        self._capture = capture
        self._assistant_chunks: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if (
            direction == FrameDirection.DOWNSTREAM
            and self._capture == "user"
            and isinstance(frame, TranscriptionFrame)
        ):
            from core.task_queue import task_queue
            task_queue.enqueue(save_conversation_message, self._conversation_id, "You", frame.text, key=self._conversation_id)
        elif (
            direction == FrameDirection.DOWNSTREAM
            and self._capture == "assistant"
            and isinstance(frame, LLMTextFrame)
        ):
            self._assistant_chunks.append(frame.text)
        elif (
            direction == FrameDirection.DOWNSTREAM
            and self._capture == "assistant"
            and isinstance(frame, LLMFullResponseEndFrame)
        ):
            assistant_text = "".join(self._assistant_chunks).strip()
            self._assistant_chunks.clear()
            from core.task_queue import task_queue
            task_queue.enqueue(save_conversation_message, self._conversation_id, "Aura", assistant_text, key=self._conversation_id)
        elif (
            direction == FrameDirection.DOWNSTREAM
            and self._capture == "assistant"
            and isinstance(frame, FunctionCallResultFrame)
        ):
            payload = {
                "tool_call_id": frame.tool_call_id,
                "function_name": frame.function_name,
                "arguments": frame.arguments,
                "result": frame.result,
            }
            from core.task_queue import task_queue
            task_queue.enqueue(
                save_conversation_message,
                self._conversation_id,
                "ToolCall",
                json.dumps(payload, default=str),
                key=self._conversation_id,
            )

        await self.push_frame(frame, direction)


class ToolFillerProcessor(FrameProcessor):
    def __init__(
        self,
        latency_state: TurnLatencyState | None = None,
        delay_seconds: float | None = None,
        enabled: bool | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._latency_state = latency_state
        self._delay_seconds = tool_filler_delay_seconds() if delay_seconds is None else delay_seconds
        self._enabled = tool_filler_enabled() if enabled is None else enabled
        self._active_calls: set[str] = set()
        self._filler_task: asyncio.Task | None = None

    @staticmethod
    def _json_safe(value):
        return json.loads(json.dumps(value, default=str))

    async def _push_tool_event(
        self,
        frame: FunctionCallInProgressFrame | FunctionCallResultFrame | FunctionCallCancelFrame,
        direction: FrameDirection,
    ) -> None:
        payload = {
            "tool_call_id": frame.tool_call_id,
            "function_name": frame.function_name,
            "status": (
                "completed" if isinstance(frame, FunctionCallResultFrame)
                else "cancelled" if isinstance(frame, FunctionCallCancelFrame)
                else "in_progress"
            ),
        }
        if hasattr(frame, "arguments"):
            payload["arguments"] = self._json_safe(frame.arguments)
        if isinstance(frame, FunctionCallResultFrame):
            payload["result"] = self._json_safe(frame.result)
        await self.push_frame(transport_server_message("tool_call", payload), direction)

    def _cancel_filler(self):
        if self._filler_task and not self._filler_task.done():
            self._filler_task.cancel()
        self._filler_task = None

    async def _emit_filler(self, tool_call_id: str, direction: FrameDirection) -> None:
        if self._latency_state and self._latency_state.tool_filler_spoken:
            return
        self._filler_task = None
        if self._latency_state:
            self._latency_state.tool_filler_spoken = True
        filler_text = "Let me check that."
        await self.push_frame(
            transport_server_message(
                "assistant_transcript",
                {
                    "id": f"tool-filler-{tool_call_id}",
                    "text": filler_text,
                    "source": "tool_filler",
                },
            ),
            direction,
        )
        await self.push_frame(
            TTSSpeakFrame(filler_text, append_to_context=False),
            direction,
        )

    async def _delayed_filler(self, direction: FrameDirection):
        try:
            await asyncio.sleep(self._delay_seconds)
            if self._active_calls:
                tool_call_id = sorted(self._active_calls)[0]
                await self._emit_filler(tool_call_id, direction)
        except asyncio.CancelledError:
            return

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, FunctionCallInProgressFrame):
            if self._latency_state:
                self._latency_state.tool_used = True
            self._active_calls.add(frame.tool_call_id)
            filler_already_spoken = bool(
                self._latency_state and self._latency_state.tool_filler_spoken
            )
            if (
                self._enabled
                and not filler_already_spoken
            ):
                if self._delay_seconds == 0:
                    await self._emit_filler(frame.tool_call_id, direction)
                elif not self._filler_task or self._filler_task.done():
                    self._filler_task = asyncio.create_task(self._delayed_filler(direction))
            await self._push_tool_event(frame, direction)
        elif direction == FrameDirection.DOWNSTREAM and isinstance(
            frame,
            (FunctionCallResultFrame, FunctionCallCancelFrame),
        ):
            await self._push_tool_event(frame, direction)
            self._active_calls.discard(frame.tool_call_id)
            if not self._active_calls:
                self._cancel_filler()
        elif isinstance(frame, (InterruptionFrame, LLMTextFrame)):
            self._active_calls.clear()
            self._cancel_filler()

        await self.push_frame(frame, direction)

    async def cleanup(self):
        self._active_calls.clear()
        self._cancel_filler()
        await super().cleanup()


class ContextRetrievalProcessor(FrameProcessor):
    def __init__(
        self,
        user_id: int | None,
        conversation_id: int | None,
        context: LLMContext,
        latency_state: TurnLatencyState | None = None,
        web_search=None,
        ready_corpus_check=None,
        filler_delay_seconds: float | None = None,
        filler_enabled: bool | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._user_id = user_id
        self._conversation_id = conversation_id
        self._context = context
        self._latency_state = latency_state
        self._web_search = web_search
        self._ready_corpus_check = ready_corpus_check
        self._active_task: asyncio.Task | None = None
        self._retrieval_generation = 0
        self._dynamic_messages: list[dict] = []
        self._web_search_resolved = False
        self._web_search_attempted = False
        self._tool_filler_emitted = False
        self._filler_delay_seconds = (
            tool_filler_delay_seconds() if filler_delay_seconds is None else filler_delay_seconds
        )
        self._filler_enabled = tool_filler_enabled() if filler_enabled is None else filler_enabled
        self._proactive_filler_task: asyncio.Task | None = None
        self._rag_filler_task: asyncio.Task | None = None
        self._active_web_call_id: str | None = None

    @property
    def web_search_resolved(self) -> bool:
        return self._web_search_resolved

    @property
    def web_search_attempted(self) -> bool:
        return self._web_search_attempted

    @property
    def tool_filler_emitted(self) -> bool:
        return self._tool_filler_emitted

    def _supersede_active_retrieval(self) -> int:
        self._retrieval_generation += 1
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
        self._active_task = None
        self._cancel_proactive_filler()
        return self._retrieval_generation

    def _cancel_proactive_filler(self) -> None:
        if self._proactive_filler_task and not self._proactive_filler_task.done():
            self._proactive_filler_task.cancel()
        self._proactive_filler_task = None
        if self._rag_filler_task and not self._rag_filler_task.done():
            self._rag_filler_task.cancel()
        self._rag_filler_task = None

    async def _emit_proactive_filler(
        self,
        tool_call_id: str,
        direction: FrameDirection,
    ) -> None:
        if self._latency_state and self._latency_state.tool_filler_spoken:
            return
        filler_text = "Let me check that."
        self._tool_filler_emitted = True
        if self._latency_state:
            self._latency_state.tool_filler_spoken = True
        await self.push_frame(
            transport_server_message(
                "assistant_transcript",
                {
                    "id": f"tool-filler-{tool_call_id}",
                    "text": filler_text,
                    "source": "tool_filler",
                },
            ),
            direction,
        )
        await self.push_frame(
            TTSSpeakFrame(filler_text, append_to_context=False),
            direction,
        )

    async def _delayed_proactive_filler(
        self,
        tool_call_id: str,
        generation: int,
        direction: FrameDirection,
    ) -> None:
        try:
            await asyncio.sleep(self._filler_delay_seconds)
            if (
                self._is_current_generation(generation)
                and self._active_web_call_id == tool_call_id
            ):
                # Once emission starts, result handling must not cancel between
                # the transcript event and its matching TTS frame.
                self._proactive_filler_task = None
                await self._emit_proactive_filler(tool_call_id, direction)
        except asyncio.CancelledError:
            return

    async def _delayed_rag_filler(
        self,
        filler_id: str,
        generation: int,
        direction: FrameDirection,
    ) -> None:
        try:
            await asyncio.sleep(self._filler_delay_seconds)
            if self._is_current_generation(generation):
                self._rag_filler_task = None
                await self._emit_proactive_filler(filler_id, direction)
        except asyncio.CancelledError:
            return

    async def _start_web_tool(
        self,
        query: str,
        generation: int,
        direction: FrameDirection,
    ) -> str:
        turn_id = self._latency_state.turn_id if self._latency_state else generation
        tool_call_id = f"web-search-{turn_id}-{generation}"
        self._active_web_call_id = tool_call_id
        # This deterministic search owns web lookup for the turn even if the
        # provider times out. The LLM must consume its fallback, not call the
        # same tool a second time.
        self._web_search_attempted = True
        if self._filler_enabled and self._filler_delay_seconds == 0:
            await self._emit_proactive_filler(tool_call_id, direction)
        await self.push_frame(
            transport_server_message(
                "tool_call",
                {
                    "tool_call_id": tool_call_id,
                    "function_name": "tavily_search",
                    "arguments": {"query": query},
                    "status": "in_progress",
                },
            ),
            direction,
        )
        if self._filler_enabled and self._filler_delay_seconds > 0:
            self._proactive_filler_task = asyncio.create_task(
                self._delayed_proactive_filler(tool_call_id, generation, direction)
            )
        return tool_call_id

    async def _finish_web_tool(
        self,
        tool_call_id: str | None,
        result,
        status: str,
        direction: FrameDirection,
    ) -> None:
        if not tool_call_id or self._active_web_call_id != tool_call_id:
            return
        self._cancel_proactive_filler()
        self._active_web_call_id = None
        await self.push_frame(
            transport_server_message(
                "tool_call",
                {
                    "tool_call_id": tool_call_id,
                    "function_name": "tavily_search",
                    "result": ToolFillerProcessor._json_safe(result),
                    "status": status,
                },
            ),
            direction,
        )

    def _is_current_generation(self, generation: int) -> bool:
        return generation == self._retrieval_generation

    @staticmethod
    def _route_deadline(needs_memory: bool, needs_rag: bool, needs_web: bool) -> float:
        branch_deadlines = [
            timeout
            for enabled, timeout in (
                (needs_memory, RAG_VOICE_MEMORY_TIMEOUT_SECONDS),
                (needs_rag, RAG_VOICE_RAG_TIMEOUT_SECONDS),
                (needs_web, RAG_VOICE_WEB_TIMEOUT_SECONDS),
            )
            if enabled
        ]
        if not branch_deadlines:
            return 0.1
        # Branches run concurrently. Allow only a small orchestration margin,
        # while retaining the global value as an absolute safety ceiling.
        return min(RAG_VOICE_RETRIEVAL_TIMEOUT_SECONDS, max(branch_deadlines) + 0.1)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if (
            direction == FrameDirection.DOWNSTREAM
            and self._user_id
            and isinstance(frame, LLMContextFrame)
        ):
            # This processor runs after the user aggregator. Route the complete
            # spoken turn once instead of launching work for each final STT
            # fragment emitted by the provider.
            self.clear_dynamic_context()
            combined_query = ToolRoutingProcessor._latest_user_text(self._context).strip()
            if not combined_query:
                await self.push_frame(frame, direction)
                return

            needs_memory = is_recall_query(combined_query)
            rag_considered = should_attempt_rag_retrieval(combined_query)
            needs_rag = rag_considered
            if self._latency_state:
                self._latency_state.rag_considered = rag_considered
            if rag_considered and self._ready_corpus_check:
                try:
                    has_ready_corpus = await asyncio.wait_for(
                        self._ready_corpus_check(self._user_id),
                        timeout=0.1,
                    )
                except Exception as exc:
                    # Corpus discovery is an optimization. Fail open for an
                    # explicit source question so RAG availability is not lost
                    # during a transient status-check failure.
                    logger.warning(
                        "voice_route corpus_check=failed action=retrieve error_type={}",
                        type(exc).__name__,
                    )
                    has_ready_corpus = True
                if not has_ready_corpus:
                    needs_rag = False
                    if self._latency_state:
                        self._latency_state.rag_bypassed = True
                    logger.info(
                        "voice_route route=direct reason=no_ready_corpus query={!r}",
                        combined_query[:120],
                    )
            needs_web = bool(
                self._web_search
                and ToolRoutingProcessor.needs_web_search(combined_query)
            )
            if not needs_memory and not needs_rag and not needs_web:
                self._supersede_active_retrieval()
                logger.info("voice_route route=direct query={!r}", combined_query[:120])
                await self.push_frame(frame, direction)
                return

            if needs_web:
                if self._latency_state:
                    self._latency_state.tool_used = True
                    self._latency_state.mark_stage("tool_started")

            generation = self._supersede_active_retrieval()
            tool_call_id = None
            if needs_web:
                tool_call_id = await self._start_web_tool(
                    combined_query, generation, direction
                )
            if needs_rag and self._filler_enabled:
                turn_id = self._latency_state.turn_id if self._latency_state else generation
                rag_filler_id = f"rag-{turn_id}-{generation}"
                if self._filler_delay_seconds == 0:
                    await self._emit_proactive_filler(rag_filler_id, direction)
                elif not self._rag_filler_task or self._rag_filler_task.done():
                    self._rag_filler_task = asyncio.create_task(
                        self._delayed_rag_filler(rag_filler_id, generation, direction)
                    )
            task = asyncio.create_task(
                self._retrieve_and_push(
                    frame,
                    combined_query,
                    direction,
                    needs_memory,
                    needs_rag,
                    needs_web,
                    generation,
                    tool_call_id,
                )
            )
            if self._latency_state:
                self._latency_state.mark_stage("retrieval_queued")
            self._active_task = task
            task.add_done_callback(
                lambda completed: setattr(self, "_active_task", None)
                if self._active_task is completed else None
            )
            return

        await self.push_frame(frame, direction)

    async def _retrieve_and_push(
        self,
        frame,
        combined_query,
        direction,
        needs_memory,
        needs_rag,
        needs_web,
        generation,
        tool_call_id=None,
    ):
        started = time.monotonic()
        delivered = False
        try:
            async def retrieve_and_deliver():
                nonlocal delivered
                if self._is_current_generation(generation):
                    if self._latency_state:
                        self._latency_state.mark_stage("retrieval_started")

                    shared_embedding = None
                    if needs_memory and needs_rag:
                        shared_embedding = asyncio.create_task(embed_text(combined_query))

                    async def bounded_branch(awaitable, timeout, fallback, label):
                        try:
                            return await asyncio.wait_for(awaitable, timeout=timeout)
                        except TimeoutError:
                            logger.warning(
                                "voice_retrieval branch={} status=timeout budget_ms={} query={!r}",
                                label,
                                round(timeout * 1000),
                                combined_query[:120],
                            )
                            return fallback
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            logger.warning(
                                "voice_retrieval branch={} status=error error_type={} query={!r}",
                                label,
                                type(exc).__name__,
                                combined_query[:120],
                            )
                            return fallback

                    # Each waiter needs its own shield wrapper. Sharing one
                    # shielded Future lets a timeout in either branch cancel
                    # the other branch's wait even though the Task survives.
                    memory_query_embedding = (
                        asyncio.shield(shared_embedding) if shared_embedding else None
                    )
                    rag_query_embedding = (
                        asyncio.shield(shared_embedding) if shared_embedding else None
                    )
                    memory_task = build_turn_memory_context(
                        self._user_id,
                        combined_query,
                        query_embedding=memory_query_embedding,
                        current_conversation_id=self._conversation_id,
                    ) if needs_memory else asyncio.sleep(0, result=None)
                    rag_task = build_rag_context_with_payload(
                        self._user_id,
                        combined_query,
                        query_embedding=rag_query_embedding,
                    ) if needs_rag else asyncio.sleep(0, result=(None, None))
                    web_task = self._web_search(combined_query) if needs_web else asyncio.sleep(0, result=None)
                    web_fallback = {
                        "status": "timeout",
                        "message": (
                            "Web search timed out. Answer briefly from available knowledge "
                            "and disclose that live verification was unavailable."
                        ),
                    }
                    memory_context, (rag_context, rag_payload), web_payload = await asyncio.gather(
                        bounded_branch(memory_task, RAG_VOICE_MEMORY_TIMEOUT_SECONDS, None, "memory"),
                        bounded_branch(rag_task, RAG_VOICE_RAG_TIMEOUT_SECONDS, (None, None), "rag"),
                        bounded_branch(
                            web_task,
                            RAG_VOICE_WEB_TIMEOUT_SECONDS,
                            web_fallback if needs_web else None,
                            "web",
                        ),
                    )
                    # A fast retrieval no longer needs its delayed spoken
                    # filler. Cancel it before releasing the context to the LLM
                    # so it cannot race the answer into the TTS service.
                    if self._rag_filler_task and not self._rag_filler_task.done():
                        self._rag_filler_task.cancel()
                    self._rag_filler_task = None
                    if shared_embedding and not shared_embedding.done():
                        shared_embedding.cancel()
                        await asyncio.gather(shared_embedding, return_exceptions=True)
                    if not self._is_current_generation(generation):
                        logger.info(
                            "voice_retrieval status=superseded generation={} query={!r}",
                            generation,
                            combined_query[:120],
                        )
                        return None
                    if self._latency_state:
                        self._latency_state.mark_stage("retrieval_finished")

                    for content in (memory_context, rag_context):
                        if content:
                            message = {"role": "developer", "content": content}
                            self._context.add_message(message)
                            self._dynamic_messages.append(message)
                    if needs_web and web_payload is not None:
                        web_context = (
                            "Live web search result for the current user request. "
                            "Use this result directly, be concise, and mention when live data was unavailable:\n"
                            + json.dumps(web_payload, default=str)
                        )
                        message = {"role": "developer", "content": web_context}
                        self._context.add_message(message)
                        self._dynamic_messages.append(message)
                        self._web_search_resolved = True
                        if self._latency_state:
                            self._latency_state.mark_stage("tool_finished")
                    if needs_web:
                        await self._finish_web_tool(
                            tool_call_id,
                            web_payload,
                            "completed" if web_payload is not None else "failed",
                            direction,
                        )
                    if rag_payload:
                        if self._latency_state:
                            self._latency_state.rag_used = True
                        from core.task_queue import task_queue
                        task_queue.enqueue(
                            save_conversation_message,
                            self._conversation_id,
                            "RagCall",
                            json.dumps(rag_payload, default=str),
                            key=self._conversation_id,
                        )
                        # Deliver diagnostics before releasing the LLM context;
                        # otherwise a provider can begin/finish its response
                        # before the browser ever receives the RAG transcript.
                        await self.push_frame(
                            transport_server_message(
                                "rag_call",
                                ToolFillerProcessor._json_safe(rag_payload),
                            ),
                            direction,
                        )
                    await self.push_frame(frame, direction)
                    delivered = True
                    return rag_payload

            # The deadline covers provider work, context installation, and
            # release of the completed-turn context frame.
            route_deadline = self._route_deadline(needs_memory, needs_rag, needs_web)
            await asyncio.wait_for(
                retrieve_and_deliver(),
                timeout=route_deadline,
            )
        except TimeoutError:
            logger.warning(
                "voice_retrieval status=timeout budget_ms={} query={!r}",
                round(self._route_deadline(needs_memory, needs_rag, needs_web) * 1000), combined_query[:120],
            )
        except asyncio.CancelledError:
            logger.info(
                "voice_retrieval status=cancelled generation={} query={!r}",
                generation,
                combined_query[:120],
            )
            await self._finish_web_tool(tool_call_id, None, "cancelled", direction)
            raise
        except Exception as e:
            logger.error(f"Context retrieval error: {e}")
        finally:
            logger.info(
                "voice_retrieval status=complete duration_ms={} rag={} memory={} query={!r}",
                round((time.monotonic() - started) * 1000, 1), needs_rag, needs_memory, combined_query[:120],
            )
            if not delivered and self._is_current_generation(generation):
                await self._finish_web_tool(tool_call_id, None, "failed", direction)
                await self.push_frame(frame, direction)

    def clear_dynamic_context(self):
        if self._dynamic_messages:
            ids = {id(message) for message in self._dynamic_messages}
            self._context.messages[:] = [message for message in self._context.messages if id(message) not in ids]
            self._dynamic_messages.clear()

    def start_user_turn(self) -> None:
        """Reset query-scoped state at the aggregator's authoritative boundary."""
        self._supersede_active_retrieval()
        self.clear_dynamic_context()
        self._web_search_resolved = False
        self._web_search_attempted = False
        self._tool_filler_emitted = False
        self._active_web_call_id = None

    def finish_response(self):
        # Response completion is not a user-turn boundary: an interrupted old
        # response may finish after barge-in has already started a new turn.
        pass

    async def cleanup(self):
        task = self._active_task
        self._supersede_active_retrieval()
        if task:
            await asyncio.gather(task, return_exceptions=True)
        self._cancel_proactive_filler()
        await super().cleanup()


class TurnContextCleanupProcessor(FrameProcessor):
    def __init__(self, retrieval: ContextRetrievalProcessor, latency_state: TurnLatencyState | None = None, **kwargs):
        super().__init__(**kwargs)
        self._retrieval = retrieval
        self._latency_state = latency_state

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMFullResponseEndFrame):
            self._retrieval.finish_response()
            if self._latency_state:
                self._latency_state.finish_response()
        await self.push_frame(frame, direction)
