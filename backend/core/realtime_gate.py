"""Coordinate best-effort background work with latency-critical voice turns."""

import asyncio


class RealtimeTurnGate:
    def __init__(self):
        self._active: set[object] = set()
        self._idle_tasks: set[asyncio.Task] = set()
        self._idle = asyncio.Event()
        self._idle.set()

    def begin(self, key: object) -> None:
        self._active.add(key)
        self._idle.clear()
        # Best-effort enrichment is always preemptible. Cancelling registered
        # child tasks keeps embeddings/summaries off the live STT/LLM/TTS path;
        # the queue worker retries the same item after voice becomes idle.
        for task in tuple(self._idle_tasks):
            if not task.done():
                task.cancel()

    def end(self, key: object) -> None:
        self._active.discard(key)
        if not self._active:
            self._idle.set()

    async def wait_until_idle(self) -> None:
        await self._idle.wait()

    def register_idle_task(self, task: asyncio.Task) -> bool:
        """Atomically claim the current idle window for cancellable work."""
        if self._active:
            return False
        self._idle_tasks.add(task)
        return True

    def unregister_idle_task(self, task: asyncio.Task) -> None:
        self._idle_tasks.discard(task)

    @property
    def active(self) -> int:
        return len(self._active)


realtime_turn_gate = RealtimeTurnGate()
