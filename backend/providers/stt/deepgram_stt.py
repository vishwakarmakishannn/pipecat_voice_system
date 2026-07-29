import os
import asyncio
import time
from collections import deque
from loguru import logger
from deepgram.listen.v1.types.listen_v1keep_alive import ListenV1KeepAlive
from pipecat.services.deepgram.stt import DeepgramSTTService
from core.audio_config import audio_input_sample_rate


class ResilientDeepgramSTTService(DeepgramSTTService):
    """Reconnect when the SDK socket can no longer send keepalives."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._keepalive_recovery_task = None
        self._media_recovery_task = None
        self._reconnect_media_buffer = deque()
        self._reconnect_media_timestamps = deque()
        self._reconnect_media_bytes = 0
        self._reconnect_media_max_bytes = int(
            os.getenv("STT_RECONNECT_BUFFER_MAX_BYTES", "256000")
        )
        self._reconnect_media_max_age_seconds = self._buffer_max_age_seconds()
        self._reconnect_replay_batch_bytes = int(
            os.getenv("STT_RECONNECT_REPLAY_BATCH_BYTES", "32000")
        )

    @staticmethod
    def _connect_timeout_seconds() -> float:
        raw = os.getenv("STT_CONNECT_TIMEOUT_SECONDS", "5")
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"STT_CONNECT_TIMEOUT_SECONDS must be a number, got {raw!r}") from exc
        if not 0.25 <= value <= 30:
            raise ValueError(
                f"STT_CONNECT_TIMEOUT_SECONDS must be between 0.25 and 30, got {value}"
            )
        return value

    @staticmethod
    def _keepalive_interval_seconds() -> float:
        raw = os.getenv("STT_KEEPALIVE_INTERVAL_SECONDS", "2")
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"STT_KEEPALIVE_INTERVAL_SECONDS must be a number, got {raw!r}") from exc
        if not 0.5 <= value <= 10:
            raise ValueError(
                f"STT_KEEPALIVE_INTERVAL_SECONDS must be between 0.5 and 10, got {value}"
            )
        return value

    @staticmethod
    def _buffer_max_age_seconds() -> float:
        raw = os.getenv("STT_RECONNECT_BUFFER_MAX_AGE_SECONDS", "2")
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"STT_RECONNECT_BUFFER_MAX_AGE_SECONDS must be a number, got {raw!r}") from exc
        if not 0.25 <= value <= 10:
            raise ValueError(
                f"STT_RECONNECT_BUFFER_MAX_AGE_SECONDS must be between 0.25 and 10, got {value}"
            )
        return value

    async def _connect(self):
        started = time.monotonic()
        try:
            await super()._connect()
        except BaseException:
            logger.warning(
                "voice_startup stage=provider_connected service=deepgram_stt "
                "status=failed duration_ms={}",
                round((time.monotonic() - started) * 1000, 1),
            )
            raise
        try:
            await asyncio.wait_for(
                self._connection_ready.wait(),
                timeout=self._connect_timeout_seconds(),
            )
        except BaseException:
            # Never leave a retrying connection task behind after startup has
            # failed or the pipeline has been cancelled.
            if self._connection_task:
                await self.cancel_task(self._connection_task)
                self._connection_task = None
            self._connection = None
            self._connection_ready.clear()
            raise
        logger.info(
            "voice_startup stage=provider_connected service=deepgram_stt "
            "status=ready duration_ms={}",
            round((time.monotonic() - started) * 1000, 1),
        )

    async def _reset_stt_ttfb_state(self):
        """Cancel abandoned-turn timing without emitting a stale TTFB sample."""
        await super()._reset_stt_ttfb_state()
        metrics = getattr(self, "_metrics", None)
        if metrics is not None and getattr(metrics, "_start_ttfb_time", 0):
            logger.info("{}: discarded abandoned STT TTFB measurement", self)
            metrics._start_ttfb_time = 0
            metrics._ttfa_active = False
            metrics._ttfa_buffer = b""
        self._last_transcript_time = 0

    def _ensure_media_buffer(self) -> None:
        if not hasattr(self, "_reconnect_media_buffer"):
            self._reconnect_media_buffer = deque()
            self._reconnect_media_bytes = 0
            self._reconnect_media_max_bytes = int(
                os.getenv("STT_RECONNECT_BUFFER_MAX_BYTES", "256000")
            )
        if not hasattr(self, "_reconnect_media_timestamps"):
            now = time.monotonic()
            self._reconnect_media_timestamps = deque(
                now for _ in self._reconnect_media_buffer
            )
        if not hasattr(self, "_reconnect_media_max_age_seconds"):
            self._reconnect_media_max_age_seconds = self._buffer_max_age_seconds()
        if not hasattr(self, "_reconnect_replay_batch_bytes"):
            self._reconnect_replay_batch_bytes = int(
                os.getenv("STT_RECONNECT_REPLAY_BATCH_BYTES", "32000")
            )

    def _drop_expired_media(self, now: float | None = None) -> int:
        self._ensure_media_buffer()
        now = time.monotonic() if now is None else now
        dropped = 0
        while (
            self._reconnect_media_buffer
            and self._reconnect_media_timestamps
            and now - self._reconnect_media_timestamps[0] > self._reconnect_media_max_age_seconds
        ):
            audio = self._reconnect_media_buffer.popleft()
            self._reconnect_media_timestamps.popleft()
            self._reconnect_media_bytes -= len(audio)
            dropped += len(audio)
        if dropped:
            logger.warning("{}: dropped {} expired STT reconnect audio bytes", self, dropped)
        return dropped

    def _buffer_media(self, audio: bytes) -> None:
        self._ensure_media_buffer()
        if not audio:
            return
        now = time.monotonic()
        self._drop_expired_media(now)
        self._reconnect_media_buffer.append(audio)
        self._reconnect_media_timestamps.append(now)
        self._reconnect_media_bytes += len(audio)
        dropped = 0
        while (
            self._reconnect_media_bytes > self._reconnect_media_max_bytes
            and len(self._reconnect_media_buffer) > 1
        ):
            removed = self._reconnect_media_buffer.popleft()
            self._reconnect_media_timestamps.popleft()
            self._reconnect_media_bytes -= len(removed)
            dropped += len(removed)
        if dropped:
            logger.warning(
                "{}: STT reconnect buffer overflow; dropped {} oldest audio bytes",
                self,
                dropped,
            )

    async def _flush_media_buffer(self) -> None:
        self._ensure_media_buffer()
        connection = self._connection
        if not connection:
            return
        self._drop_expired_media()
        while self._reconnect_media_buffer:
            parts = []
            batch_bytes = 0
            for audio in self._reconnect_media_buffer:
                if parts and batch_bytes + len(audio) > self._reconnect_replay_batch_bytes:
                    break
                parts.append(audio)
                batch_bytes += len(audio)
            payload = b"".join(parts)
            try:
                await connection.send_media(payload)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if self._connection is connection:
                    self._connection = None
                    self._connection_ready.clear()
                self._schedule_media_recovery(error)
                return
            for _ in parts:
                self._reconnect_media_buffer.popleft()
                self._reconnect_media_timestamps.popleft()
            self._reconnect_media_bytes -= batch_bytes

    async def _do_reconnect(self):
        # Flush failed/deferred raw media before STTService replays frames that
        # arrived during the reconnect itself, preserving utterance order.
        await super()._do_reconnect()
        await self._flush_media_buffer()

    def _schedule_keepalive_recovery(self, error: Exception) -> None:
        if self._keepalive_recovery_task and not self._keepalive_recovery_task.done():
            return
        logger.warning("{}: Keepalive failed; scheduling STT reconnect: {!r}", self, error)
        self._keepalive_recovery_task = self.create_task(
            self._request_reconnect(), name=f"{self}::keepalive-reconnect"
        )

    async def _keepalive_handler(self):
        while True:
            await asyncio.sleep(self._keepalive_interval_seconds())
            connection = self._connection
            if not connection:
                continue
            try:
                await connection.send_keep_alive(ListenV1KeepAlive(type="KeepAlive"))
                logger.trace("{}: Sent keepalive", self)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._schedule_keepalive_recovery(error)
                return

    def _schedule_media_recovery(self, error: Exception) -> None:
        if self._media_recovery_task and not self._media_recovery_task.done():
            return
        logger.warning("{}: Media send failed; scheduling follow-up STT reconnect: {!r}", self, error)
        self._media_recovery_task = self.create_task(
            self._recover_after_active_reconnect(), name=f"{self}::media-reconnect"
        )

    async def _recover_after_active_reconnect(self) -> None:
        """Do not let a failed buffered replay strand the replacement socket."""
        active_recovery = self._keepalive_recovery_task
        if active_recovery and active_recovery is not asyncio.current_task():
            try:
                await asyncio.shield(active_recovery)
            except asyncio.CancelledError:
                raise
            except Exception:
                # The follow-up reconnect is still required when the first
                # recovery itself failed.
                pass
        if self._connection is None:
            await self._request_reconnect()

    async def run_stt(self, audio: bytes):
        """Buffer unsent media and replay it after a replacement socket is ready."""
        connection = self._connection
        if connection:
            try:
                await connection.send_media(audio)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning("{}: send_media failed; connection will reconnect: {!r}", self, error)
                if self._connection is connection:
                    self._connection = None
                    self._connection_ready.clear()
                self._buffer_media(audio)
                self._schedule_media_recovery(error)
        else:
            self._buffer_media(audio)
            self._schedule_media_recovery(RuntimeError("STT connection is unavailable"))
        yield None

    async def _disconnect(self):
        """Tear down even when the broken socket cannot send CloseStream."""
        media_recovery = getattr(self, "_media_recovery_task", None)
        if media_recovery and media_recovery is not asyncio.current_task() and not media_recovery.done():
            await self.cancel_task(media_recovery)
            self._media_recovery_task = None

        if not self._connection_task:
            return

        logger.debug("Disconnecting resilient Deepgram STT")
        self._connection_ready.clear()
        connection = self._connection
        self._connection = None

        if connection:
            try:
                await connection.send_close_stream()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # A failed keepalive commonly means all writes fail. The
                # connection task cancellation below is the authoritative
                # teardown; CloseStream is best-effort only.
                logger.warning("{}: CloseStream failed during recovery: {!r}", self, error)

        await self.cancel_task(self._connection_task)
        self._connection_task = None


def get_deepgram_stt():
    return ResilientDeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        sample_rate=audio_input_sample_rate(),
    )
