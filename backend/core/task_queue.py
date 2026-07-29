import asyncio
import os
import time
from typing import Callable, Coroutine, Any
from loguru import logger

class BackgroundTaskQueue:
    def __init__(self, maxsize: int | None = None):
        configured = int(os.getenv("BACKGROUND_TASK_QUEUE_MAXSIZE", "256")) if maxsize is None else maxsize
        if configured < 1:
            raise ValueError("BACKGROUND_TASK_QUEUE_MAXSIZE must be at least 1")
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=configured)
        self._enrichment_queue: asyncio.Queue = asyncio.Queue(maxsize=configured)
        self._workers: list[asyncio.Task] = []
        self._enrichment_worker: asyncio.Task | None = None
        self._is_running = False
        self._key_locks: dict[Any, asyncio.Lock] = {}

    @property
    def is_running(self) -> bool:
        """Whether background persistence workers are accepting work."""
        return self._is_running and bool(self._workers) and all(
            not worker.done() for worker in self._workers
        )

    async def _worker(self):
        while self._is_running:
            try:
                enqueued_at, task_func, args, kwargs, key = await self._queue.get()
                logger.info(
                    "background_task_queue wait_ms={} depth={} task={}",
                    round((time.monotonic() - enqueued_at) * 1000, 1),
                    self._queue.qsize(),
                    task_func.__name__,
                )
                try:
                    if key is None:
                        await task_func(*args, **kwargs)
                    else:
                        lock = self._key_locks.setdefault(key, asyncio.Lock())
                        async with lock:
                            await task_func(*args, **kwargs)
                except Exception as e:
                    logger.exception(f"Error in background task {task_func.__name__}: {e}")
                finally:
                    self._queue.task_done()
            except asyncio.CancelledError:
                break

    async def _run_enrichment(self):
        """Run non-urgent memory work only when live voice has yielded."""
        from core.realtime_gate import realtime_turn_gate

        while self._is_running:
            try:
                enqueued_at, task_func, args, kwargs, key = await self._enrichment_queue.get()
                await realtime_turn_gate.wait_until_idle()
                logger.info(
                    "background_enrichment_queue wait_ms={} depth={} task={}",
                    round((time.monotonic() - enqueued_at) * 1000, 1),
                    self._enrichment_queue.qsize(),
                    task_func.__name__,
                )
                try:
                    if key is None:
                        await task_func(*args, **kwargs)
                    else:
                        lock = self._key_locks.setdefault(key, asyncio.Lock())
                        async with lock:
                            await task_func(*args, **kwargs)
                except Exception as e:
                    logger.exception(f"Error in background enrichment {task_func.__name__}: {e}")
                finally:
                    self._enrichment_queue.task_done()
            except asyncio.CancelledError:
                break

    def start(self, num_workers: int = 3):
        if self._is_running:
            return
        self._is_running = True
        for _ in range(num_workers):
            self._workers.append(asyncio.create_task(self._worker()))
        self._enrichment_worker = asyncio.create_task(
            self._run_enrichment(), name="voice-memory-enrichment"
        )
        logger.info(f"Started BackgroundTaskQueue with {num_workers} workers.")

    async def stop(self):
        if not self._is_running:
            return
        await self._queue.join()
        await self._enrichment_queue.join()
        self._is_running = False
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        if self._enrichment_worker:
            self._enrichment_worker.cancel()
            await asyncio.gather(self._enrichment_worker, return_exceptions=True)
            self._enrichment_worker = None
        self._workers.clear()
        self._key_locks.clear()
        logger.info("Stopped BackgroundTaskQueue.")

    @property
    def depth(self) -> int:
        return self._queue.qsize() + self._enrichment_queue.qsize()

    @property
    def capacity(self) -> int:
        return self._queue.maxsize

    def enqueue(
        self,
        task_func: Callable[..., Coroutine[Any, Any, Any]],
        *args,
        key=None,
        enrichment: bool = False,
        **kwargs,
    ) -> bool:
        queue = self._enrichment_queue if enrichment else self._queue
        try:
            queue.put_nowait((time.monotonic(), task_func, args, kwargs, key))
            return True
        except asyncio.QueueFull:
            logger.error(
                "background_task_queue status=rejected lane={} depth={} capacity={} task={}",
                "enrichment" if enrichment else "persistence",
                queue.qsize(), self.capacity, task_func.__name__,
            )
            return False

task_queue = BackgroundTaskQueue()
