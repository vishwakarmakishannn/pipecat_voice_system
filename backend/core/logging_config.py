"""Non-blocking, bounded logging for the real-time voice process."""

import atexit
import os
from queue import Empty, Full, Queue
import sys
from threading import Event, Thread

from loguru import logger


class BoundedLogSink:
    def __init__(self, *, max_messages: int = 2048):
        if max_messages < 1:
            raise ValueError("max_messages must be positive")
        self._queue: Queue[str] = Queue(maxsize=max_messages)
        self._stopping = Event()
        self._dropped = 0
        self._thread = Thread(target=self._write, name="voice-log-writer", daemon=True)
        self._thread.start()

    def __call__(self, message) -> None:
        try:
            self._queue.put_nowait(str(message))
        except Full:
            self._dropped += 1

    def _write(self) -> None:
        while not self._stopping.is_set() or not self._queue.empty():
            try:
                message = self._queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                sys.stderr.write(message)
                sys.stderr.flush()
            finally:
                self._queue.task_done()
        if self._dropped:
            sys.stderr.write(f"voice_logging dropped_messages={self._dropped}\n")
            sys.stderr.flush()

    def stop(self) -> None:
        self._stopping.set()
        self._thread.join(timeout=1.0)


_sink: BoundedLogSink | None = None


def configure_nonblocking_logging(*, force: bool = False) -> BoundedLogSink:
    global _sink
    if _sink is not None and not force:
        return _sink
    if _sink is None:
        raw_capacity = os.getenv("VOICE_LOG_QUEUE_SIZE", "2048")
        try:
            capacity = int(raw_capacity)
        except ValueError as exc:
            raise ValueError(f"VOICE_LOG_QUEUE_SIZE must be an integer, got {raw_capacity!r}") from exc
        _sink = BoundedLogSink(max_messages=capacity)
        atexit.register(_sink.stop)
    logger.remove()
    logger.add(_sink, level=os.getenv("LOG_LEVEL", "INFO"), catch=True)
    return _sink
