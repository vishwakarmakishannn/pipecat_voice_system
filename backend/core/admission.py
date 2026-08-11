"""Fail-fast capacity control for latency-sensitive voice sessions."""

import asyncio
import os
from contextvars import ContextVar


current_voice_slot_id: ContextVar[int | None] = ContextVar(
    "current_voice_slot_id",
    default=None,
)


class VoiceAdmissionController:
    """Bound concurrent voice pipelines without queueing callers."""

    def __init__(self, limit: int | None = None):
        configured = limit if limit is not None else int(os.getenv("VOICE_MAX_CONCURRENT_SESSIONS", "8"))
        if configured < 1:
            raise ValueError("VOICE_MAX_CONCURRENT_SESSIONS must be at least 1")
        self.limit = configured
        self._leased_slots: set[int] = set()
        self._legacy_leases: list[int] = []
        self._lock = asyncio.Lock()
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def active(self) -> int:
        return len(self._leased_slots)

    @property
    def has_capacity(self) -> bool:
        return self.active < self.limit

    async def try_acquire_slot(self) -> int | None:
        """Reserve the lowest free voice/llama slot without queueing."""
        async with self._lock:
            for slot_id in range(self.limit):
                if slot_id not in self._leased_slots:
                    self._leased_slots.add(slot_id)
                    self._idle.clear()
                    return slot_id
            return None

    async def try_acquire(self) -> bool:
        """Backward-compatible boolean admission API."""
        slot_id = await self.try_acquire_slot()
        if slot_id is None:
            return False
        self._legacy_leases.append(slot_id)
        return True

    async def release(self, slot_id: int | None = None) -> None:
        async with self._lock:
            if slot_id is None:
                if not self._legacy_leases:
                    raise RuntimeError(
                        "voice admission release without an active lease"
                    )
                slot_id = self._legacy_leases.pop(0)
            if slot_id not in self._leased_slots:
                raise RuntimeError("voice admission release without an active lease")
            self._leased_slots.remove(slot_id)
            if not self._leased_slots:
                self._idle.set()

    async def wait_until_idle(self) -> None:
        """Wait until no latency-sensitive voice pipeline is active."""
        await self._idle.wait()


voice_admission = VoiceAdmissionController()
