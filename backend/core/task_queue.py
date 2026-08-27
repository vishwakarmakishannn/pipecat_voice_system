import asyncio
import base64
import importlib
import inspect
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Coroutine, Any
from loguru import logger


_UUID_SCOPED_CALL_TASKS = {
    "save_call_event",
    "save_call_operation",
    "save_call_summary",
    "save_call_turn",
    "save_transcript_entry",
}


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
        self._enrichment_key_locks: dict[Any, asyncio.Lock] = {}
        self._persistence_pending: dict[Any, int] = {}
        self._persistence_drained: dict[Any, asyncio.Event] = {}
        self._rejection_handlers: dict[Any, Callable[[str, str], None]] = {}
        self._overflow_tasks: set[asyncio.Task] = set()
        self._journal_active: set[str] = set()
        self._journal_replay_task: asyncio.Task | None = None

    @staticmethod
    def _encode_journal_value(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, uuid.UUID):
            return {"__aura_type__": "uuid", "value": str(value)}
        if isinstance(value, datetime):
            return {"__aura_type__": "datetime", "value": value.isoformat()}
        if isinstance(value, Path):
            return {"__aura_type__": "path", "value": str(value)}
        if isinstance(value, bytes):
            return {
                "__aura_type__": "bytes",
                "value": base64.b64encode(value).decode("ascii"),
            }
        if isinstance(value, dict):
            return {
                str(key): BackgroundTaskQueue._encode_journal_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [BackgroundTaskQueue._encode_journal_value(item) for item in value]
        return str(value)

    @staticmethod
    def _decode_journal_value(value: Any) -> Any:
        if isinstance(value, list):
            return [BackgroundTaskQueue._decode_journal_value(item) for item in value]
        if not isinstance(value, dict):
            return value
        value_type = value.get("__aura_type__")
        if value_type == "uuid":
            return uuid.UUID(value["value"])
        if value_type == "datetime":
            return datetime.fromisoformat(value["value"])
        if value_type == "path":
            return Path(value["value"])
        if value_type == "bytes":
            return base64.b64decode(value["value"])
        return {
            key: BackgroundTaskQueue._decode_journal_value(item)
            for key, item in value.items()
        }

    @staticmethod
    def _journal_root() -> Path:
        return Path(
            os.getenv("PERSISTENCE_SPOOL_DIR", "data/persistence-spool")
        ).expanduser().resolve()

    def _write_journal(
        self,
        task_func: Callable[..., Coroutine[Any, Any, Any]],
        args: tuple,
        kwargs: dict,
        key: Any,
    ) -> str | None:
        module = getattr(task_func, "__module__", "")
        qualname = getattr(task_func, "__qualname__", "")
        if module != "services.calls" or not qualname or "<locals>" in qualname:
            return None
        journal_id = uuid.uuid4().hex
        if task_func.__name__ in {"save_transcript_entry", "save_call_operation"}:
            kwargs = {**kwargs, "persistence_id": journal_id}
        root = self._journal_root()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = root / f"{journal_id}.json"
        temporary = root / f".{journal_id}.tmp"
        payload = {
            "version": 1,
            "id": journal_id,
            "module": module,
            "qualname": qualname,
            "args": self._encode_journal_value(list(args)),
            "kwargs": self._encode_journal_value(kwargs),
            "key": self._encode_journal_value(key),
        }
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        temporary.replace(target)
        kwargs.clear()
        kwargs.update(self._decode_journal_value(payload["kwargs"]))
        return str(target)

    @staticmethod
    def _validated_invocation(
        task_func: Callable[..., Coroutine[Any, Any, Any]],
        args: tuple,
        kwargs: dict,
        key: Any,
        *,
        repair_from_key: bool = False,
    ) -> tuple[tuple, dict, bool]:
        """Validate durable calls and repair the one legacy missing-ID shape.

        A call-scoped journal key is the call ID. Version-1 RAG-operation
        journals written by the affected release omitted that same value from
        ``save_call_operation`` args. Replaying it from the durable key
        preserves the operation and prevents a deterministic TypeError from
        retrying forever.
        """
        repaired = False
        try:
            inspect.signature(task_func).bind(*args, **kwargs)
        except TypeError as original:
            repairable = (
                repair_from_key
                and not args
                and key is not None
                and getattr(task_func, "__module__", "") == "services.calls"
                and getattr(task_func, "__name__", "") == "save_call_operation"
            )
            if not repairable:
                raise
            try:
                call_id = key if isinstance(key, uuid.UUID) else uuid.UUID(str(key))
                repaired_args = (call_id,)
                inspect.signature(task_func).bind(*repaired_args, **kwargs)
            except (TypeError, ValueError, AttributeError):
                raise original
            args = repaired_args
            repaired = True

        if (
            getattr(task_func, "__module__", "") == "services.calls"
            and getattr(task_func, "__name__", "") in _UUID_SCOPED_CALL_TASKS
        ):
            if args:
                call_id = args[0]
                location = "args[0]"
            else:
                call_id = kwargs.get("call_id")
                location = "call_id"
            if not isinstance(call_id, uuid.UUID):
                try:
                    normalized_call_id = uuid.UUID(str(call_id))
                except (TypeError, ValueError, AttributeError) as exc:
                    raise ValueError(
                        f"{task_func.__name__} requires a UUID call_id; "
                        f"received {type(call_id).__name__} at {location}"
                    ) from exc
                if args:
                    args = (normalized_call_id, *args[1:])
                else:
                    kwargs = {**kwargs, "call_id": normalized_call_id}

        return args, kwargs, repaired

    async def _replay_journals(self) -> None:
        while self._is_running:
            root = self._journal_root()
            if root.exists():
                for path in sorted(root.glob("*.json")):
                    path_string = str(path)
                    if path_string in self._journal_active:
                        continue
                    try:
                        payload = json.loads(path.read_text(encoding="utf-8"))
                        module = importlib.import_module(payload["module"])
                        task_func = module
                        for part in payload["qualname"].split("."):
                            task_func = getattr(task_func, part)
                        args = tuple(self._decode_journal_value(payload["args"]))
                        kwargs = self._decode_journal_value(payload["kwargs"])
                        key = self._decode_journal_value(payload["key"])
                        args, kwargs, repaired = self._validated_invocation(
                            task_func,
                            args,
                            kwargs,
                            key,
                            repair_from_key=True,
                        )
                        if repaired:
                            logger.warning(
                                "persistence_journal status=repaired "
                                "reason=missing_call_id task={} journal_id={}",
                                task_func.__name__,
                                payload.get("id", path.stem),
                            )
                        self._queue.put_nowait(
                            (time.monotonic(), task_func, args, kwargs, key, path_string)
                        )
                        self._journal_active.add(path_string)
                        self._mark_persistence_queued(key)
                    except asyncio.QueueFull:
                        break
                    except Exception as exc:
                        # A malformed durable record cannot become valid by
                        # retrying every 500 ms. Keep it for inspection while
                        # removing it from the replay glob so healthy work can
                        # continue.
                        rejected_path = path.with_suffix(".rejected")
                        try:
                            path.replace(rejected_path)
                        except OSError:
                            logger.exception(
                                "persistence_journal status=quarantine_failed path={}",
                                path,
                            )
                            continue
                        logger.error(
                            "persistence_journal status=quarantined path={} "
                            "rejected_path={} error_type={} error={}",
                            path,
                            rejected_path,
                            type(exc).__name__,
                            exc,
                        )
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                return

    async def _put_overflow(self, item) -> None:
        await self._queue.put(item)

    def _mark_persistence_queued(self, key: Any) -> None:
        if key is None:
            return
        self._persistence_pending[key] = self._persistence_pending.get(key, 0) + 1
        event = self._persistence_drained.setdefault(key, asyncio.Event())
        event.clear()

    def _mark_persistence_finished(self, key: Any) -> None:
        if key is None:
            return
        remaining = max(0, self._persistence_pending.get(key, 0) - 1)
        if remaining:
            self._persistence_pending[key] = remaining
            return
        self._persistence_pending.pop(key, None)
        self._persistence_drained.setdefault(key, asyncio.Event()).set()

    @property
    def is_running(self) -> bool:
        """Whether background persistence workers are accepting work."""
        return self._is_running and bool(self._workers) and all(
            not worker.done() for worker in self._workers
        )

    async def _worker(self):
        while self._is_running:
            try:
                enqueued_at, task_func, args, kwargs, key, journal_path = await self._queue.get()
                logger.info(
                    "background_task_queue wait_ms={} depth={} task={}",
                    round((time.monotonic() - enqueued_at) * 1000, 1),
                    self._queue.qsize(),
                    task_func.__name__,
                )
                succeeded = False
                attempt = 0
                try:
                    while self._is_running and not succeeded:
                        try:
                            if key is None:
                                result = await task_func(*args, **kwargs)
                            else:
                                lock = self._key_locks.setdefault(key, asyncio.Lock())
                                async with lock:
                                    result = await task_func(*args, **kwargs)
                            if journal_path and (result is None or result is False):
                                # Journaled call writers use a falsey result to
                                # report a permanent no-op (invalid payload,
                                # missing call, or immutable terminal call).
                                # Retrying cannot turn that result into a write
                                # and would occupy this worker forever. Remove
                                # the durable record while retaining one clear,
                                # sanitized diagnostic.
                                logger.error(
                                    "persistence_journal status=discarded "
                                    "reason=task_declined_write task={} journal_id={}",
                                    task_func.__name__,
                                    Path(journal_path).stem,
                                )
                            succeeded = True
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            logger.exception(
                                f"Error in background task {task_func.__name__}: {e}"
                            )
                            if not journal_path:
                                break
                            attempt += 1
                            await asyncio.sleep(min(5.0, 0.25 * (2 ** min(attempt, 5))))
                finally:
                    if journal_path:
                        if succeeded:
                            Path(journal_path).unlink(missing_ok=True)
                        self._journal_active.discard(journal_path)
                    self._queue.task_done()
                    self._mark_persistence_finished(key)
            except asyncio.CancelledError:
                break

    async def _run_enrichment(self):
        """Run non-urgent memory work only when live voice has yielded."""
        from core.realtime_gate import realtime_turn_gate

        while self._is_running:
            try:
                enqueued_at, task_func, args, kwargs, key, _journal_path = await self._enrichment_queue.get()
                try:
                    while self._is_running:
                        await realtime_turn_gate.wait_until_idle()

                        async def run_item():
                            if key is None:
                                await task_func(*args, **kwargs)
                            else:
                                lock = self._enrichment_key_locks.setdefault(
                                    key,
                                    asyncio.Lock(),
                                )
                                async with lock:
                                    await task_func(*args, **kwargs)

                        child = asyncio.create_task(
                            run_item(),
                            name=f"enrichment-{task_func.__name__}",
                        )
                        if not realtime_turn_gate.register_idle_task(child):
                            child.cancel()
                            await asyncio.gather(child, return_exceptions=True)
                            continue
                        logger.info(
                            "background_enrichment_queue wait_ms={} depth={} task={}",
                            round((time.monotonic() - enqueued_at) * 1000, 1),
                            self._enrichment_queue.qsize(),
                            task_func.__name__,
                        )
                        try:
                            await child
                            break
                        except asyncio.CancelledError:
                            if asyncio.current_task().cancelling():
                                raise
                            logger.info(
                                "background_enrichment_queue status=preempted "
                                "action=retry_when_idle task={}",
                                task_func.__name__,
                            )
                        finally:
                            realtime_turn_gate.unregister_idle_task(child)
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
        self._journal_replay_task = asyncio.create_task(
            self._replay_journals(), name="call-persistence-journal-replay"
        )
        logger.info(f"Started BackgroundTaskQueue with {num_workers} workers.")

    async def stop(self):
        if not self._is_running:
            return
        if self._overflow_tasks:
            await asyncio.gather(*tuple(self._overflow_tasks), return_exceptions=True)
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
        if self._journal_replay_task:
            self._journal_replay_task.cancel()
            await asyncio.gather(self._journal_replay_task, return_exceptions=True)
            self._journal_replay_task = None
        self._workers.clear()
        self._key_locks.clear()
        self._enrichment_key_locks.clear()
        self._persistence_pending.clear()
        self._persistence_drained.clear()
        self._rejection_handlers.clear()
        self._overflow_tasks.clear()
        self._journal_active.clear()
        logger.info("Stopped BackgroundTaskQueue.")

    def register_rejection_handler(
        self,
        key: Any,
        handler: Callable[[str, str], None],
    ) -> None:
        self._rejection_handlers[key] = handler

    def unregister_rejection_handler(self, key: Any) -> None:
        self._rejection_handlers.pop(key, None)

    async def wait_for_key(self, key: Any, timeout: float = 5.0) -> bool:
        """Wait for durable work for one call without waiting on enrichment work."""
        if key is None or self._persistence_pending.get(key, 0) == 0:
            return True
        event = self._persistence_drained.setdefault(key, asyncio.Event())
        try:
            await asyncio.wait_for(event.wait(), timeout=max(0.01, timeout))
            return True
        except TimeoutError:
            logger.error(
                "background_task_queue status=drain_timeout key={} pending={}",
                key,
                self._persistence_pending.get(key, 0),
            )
            return False

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
        journal_path = None
        mutable_kwargs = dict(kwargs)
        try:
            args, mutable_kwargs, _ = self._validated_invocation(
                task_func,
                args,
                mutable_kwargs,
                key,
            )
        except (TypeError, ValueError) as exc:
            logger.error(
                "background_task_queue status=rejected reason=invalid_invocation "
                "task={} error={}",
                task_func.__name__,
                exc,
            )
            handler = self._rejection_handlers.get(key)
            if handler and task_func.__name__ != "save_call_event":
                try:
                    handler(task_func.__name__, "persistence")
                except Exception:
                    logger.exception(
                        "background_task_queue rejection_handler=failed key={}", key
                    )
            return False
        if not enrichment and key is not None:
            try:
                journal_path = self._write_journal(
                    task_func, args, mutable_kwargs, key
                )
                if journal_path:
                    self._journal_active.add(journal_path)
            except Exception:
                logger.exception(
                    "persistence_journal status=write_failed task={}",
                    task_func.__name__,
                )
                return False
        if not enrichment:
            self._mark_persistence_queued(key)
        item = (
            time.monotonic(),
            task_func,
            args,
            mutable_kwargs,
            key,
            journal_path,
        )
        try:
            queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            if journal_path and not enrichment:
                overflow = asyncio.create_task(
                    self._put_overflow(item),
                    name=f"persistence-overflow-{task_func.__name__}",
                )
                self._overflow_tasks.add(overflow)
                overflow.add_done_callback(self._overflow_tasks.discard)
                logger.warning(
                    "background_task_queue status=spooled depth={} task={}",
                    queue.qsize(),
                    task_func.__name__,
                )
                return True
            if not enrichment:
                self._mark_persistence_finished(key)
            logger.error(
                "background_task_queue status=rejected lane={} depth={} capacity={} task={}",
                "enrichment" if enrichment else "persistence",
                queue.qsize(), self.capacity, task_func.__name__,
            )
            handler = self._rejection_handlers.get(key)
            if handler and task_func.__name__ != "save_call_event":
                try:
                    handler(
                        task_func.__name__,
                        "enrichment" if enrichment else "persistence",
                    )
                except Exception:
                    logger.exception(
                        "background_task_queue rejection_handler=failed key={}", key
                    )
            return False

task_queue = BackgroundTaskQueue()
