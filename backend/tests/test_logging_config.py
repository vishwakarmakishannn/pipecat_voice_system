from queue import Full

import core.logging_config as logging_config
from core.logging_config import BoundedLogSink


def test_bounded_log_sink_drops_instead_of_blocking(monkeypatch):
    sink = object.__new__(BoundedLogSink)
    sink._dropped = 0

    class FullQueue:
        @staticmethod
        def put_nowait(_message):
            raise Full

    sink._queue = FullQueue()
    sink("message")

    assert sink._dropped == 1


def test_force_reinstalls_existing_bounded_sink_after_runner_override(monkeypatch):
    existing_sink = object()
    removed = []
    added = []
    monkeypatch.setattr(logging_config, "_sink", existing_sink)
    monkeypatch.setattr(logging_config.logger, "remove", lambda: removed.append(True))
    monkeypatch.setattr(
        logging_config.logger,
        "add",
        lambda sink, **kwargs: added.append((sink, kwargs)),
    )
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    assert logging_config.configure_nonblocking_logging(force=True) is existing_sink
    assert removed == [True]
    assert added == [(existing_sink, {"level": "INFO", "catch": True})]
