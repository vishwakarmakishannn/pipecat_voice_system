"""Coordinate best-effort background work with latency-critical voice turns."""

import asyncio


class RealtimeTurnGate:
    def __init__(self):
        self._active: set[object] = set()
        self._idle = asyncio.Event()
        self._idle.set()

    def begin(self, key: object) -> None:
        self._active.add(key)
        self._idle.clear()

    def end(self, key: object) -> None:
        self._active.discard(key)
        if not self._active:
            self._idle.set()

    async def wait_until_idle(self) -> None:
        await self._idle.wait()

    @property
    def active(self) -> int:
        return len(self._active)


realtime_turn_gate = RealtimeTurnGate()
