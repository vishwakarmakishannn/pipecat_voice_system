import asyncio

import pytest

from providers.stt.deepgram_stt import ResilientDeepgramSTTService


class _FailingConnection:
    async def send_keep_alive(self, _message):
        raise AssertionError("socket drain failed")

    async def send_close_stream(self):
        raise AssertionError("socket close failed")

    async def send_media(self, _audio):
        raise AssertionError("socket media failed")


class _RecordingConnection:
    def __init__(self):
        self.media = []

    async def send_media(self, audio):
        self.media.append(audio)


@pytest.mark.anyio
async def test_keepalive_failure_schedules_one_reconnect(monkeypatch):
    service = object.__new__(ResilientDeepgramSTTService)
    service._name = "TestDeepgramSTT"
    service._connection = _FailingConnection()
    service._connection_ready = asyncio.Event()
    service._connection_ready.set()
    service._keepalive_recovery_task = None
    reconnects = 0

    async def reconnect():
        nonlocal reconnects
        reconnects += 1

    def create_task(coro, name=None):
        return asyncio.create_task(coro, name=name)

    async def no_wait(_seconds):
        return None

    service._request_reconnect = reconnect
    service.create_task = create_task
    monkeypatch.setattr("providers.stt.deepgram_stt.asyncio.sleep", no_wait)

    await service._keepalive_handler()
    await service._keepalive_recovery_task

    assert reconnects == 1
    assert service._connection_ready.is_set()


@pytest.mark.anyio
async def test_disconnect_continues_when_close_stream_fails():
    service = object.__new__(ResilientDeepgramSTTService)
    service._name = "TestDeepgramSTT"
    service._connection = _FailingConnection()
    service._connection_ready = asyncio.Event()
    service._connection_ready.set()
    service._connection_task = object()
    cancelled = []

    async def cancel_task(task):
        cancelled.append(task)

    service.cancel_task = cancel_task

    await service._disconnect()

    assert len(cancelled) == 1
    assert service._connection is None
    assert service._connection_task is None
    assert not service._connection_ready.is_set()


@pytest.mark.anyio
async def test_media_failure_reconnects_after_active_keepalive_recovery():
    service = object.__new__(ResilientDeepgramSTTService)
    service._name = "TestDeepgramSTT"
    service._connection = _FailingConnection()
    service._connection_ready = asyncio.Event()
    service._connection_ready.set()
    service._media_recovery_task = None
    first_recovery_finished = asyncio.Event()
    reconnects = 0

    async def active_recovery():
        await first_recovery_finished.wait()

    service._keepalive_recovery_task = asyncio.create_task(active_recovery())

    async def reconnect():
        nonlocal reconnects
        reconnects += 1

    service._request_reconnect = reconnect
    service.create_task = lambda coro, name=None: asyncio.create_task(coro, name=name)

    yielded = [item async for item in service.run_stt(b"audio")]
    assert yielded == [None]
    assert service._connection is None
    assert reconnects == 0

    first_recovery_finished.set()
    await service._media_recovery_task
    assert reconnects == 1


@pytest.mark.anyio
async def test_media_failure_schedules_only_one_follow_up_reconnect():
    service = object.__new__(ResilientDeepgramSTTService)
    service._name = "TestDeepgramSTT"
    service._connection = _FailingConnection()
    service._connection_ready = asyncio.Event()
    service._connection_ready.set()
    service._keepalive_recovery_task = None
    service._media_recovery_task = None
    release_reconnect = asyncio.Event()
    reconnects = 0

    async def reconnect():
        nonlocal reconnects
        reconnects += 1
        await release_reconnect.wait()

    service._request_reconnect = reconnect
    service.create_task = lambda coro, name=None: asyncio.create_task(coro, name=name)

    [item async for item in service.run_stt(b"first")]
    first_task = service._media_recovery_task
    service._connection = _FailingConnection()
    [item async for item in service.run_stt(b"second")]

    assert service._media_recovery_task is first_task
    release_reconnect.set()
    await first_task
    assert reconnects == 1


@pytest.mark.anyio
async def test_disconnected_media_is_buffered_and_replayed_in_order():
    service = object.__new__(ResilientDeepgramSTTService)
    service._name = "TestDeepgramSTT"
    service._connection = None
    service._connection_ready = asyncio.Event()
    service._media_recovery_task = asyncio.create_task(asyncio.Event().wait())
    service._reconnect_media_buffer = __import__("collections").deque()
    service._reconnect_media_bytes = 0
    service._reconnect_media_max_bytes = 1024

    [item async for item in service.run_stt(b"first")]
    [item async for item in service.run_stt(b"second")]
    assert list(service._reconnect_media_buffer) == [b"first", b"second"]

    connection = _RecordingConnection()
    service._connection = connection
    await service._flush_media_buffer()

    assert connection.media == [b"firstsecond"]
    assert not service._reconnect_media_buffer
    service._media_recovery_task.cancel()
    await asyncio.gather(service._media_recovery_task, return_exceptions=True)


def test_reconnect_buffer_drops_oldest_audio_when_bounded():
    service = object.__new__(ResilientDeepgramSTTService)
    service._name = "TestDeepgramSTT"
    service._reconnect_media_buffer = __import__("collections").deque()
    service._reconnect_media_bytes = 0
    service._reconnect_media_max_bytes = 6

    service._buffer_media(b"1111")
    service._buffer_media(b"2222")

    assert list(service._reconnect_media_buffer) == [b"2222"]


def test_reconnect_buffer_drops_audio_older_than_age_limit(monkeypatch):
    service = object.__new__(ResilientDeepgramSTTService)
    service._name = "TestDeepgramSTT"
    service._reconnect_media_buffer = __import__("collections").deque()
    service._reconnect_media_timestamps = __import__("collections").deque()
    service._reconnect_media_bytes = 0
    service._reconnect_media_max_bytes = 1024
    service._reconnect_media_max_age_seconds = 1
    times = iter([10.0, 12.0])
    monkeypatch.setattr("providers.stt.deepgram_stt.time.monotonic", lambda: next(times))

    service._buffer_media(b"stale")
    service._buffer_media(b"fresh")

    assert list(service._reconnect_media_buffer) == [b"fresh"]
    assert service._reconnect_media_bytes == len(b"fresh")


@pytest.mark.anyio
async def test_reconnect_replay_uses_bounded_coalesced_batches():
    service = object.__new__(ResilientDeepgramSTTService)
    service._name = "TestDeepgramSTT"
    service._connection = _RecordingConnection()
    service._reconnect_media_buffer = __import__("collections").deque([b"1111", b"2222", b"3333"])
    service._reconnect_media_timestamps = __import__("collections").deque([1.0, 1.0, 1.0])
    service._reconnect_media_bytes = 12
    service._reconnect_media_max_bytes = 1024
    service._reconnect_media_max_age_seconds = 1000000000
    service._reconnect_replay_batch_bytes = 8

    await service._flush_media_buffer()

    assert service._connection.media == [b"11112222", b"3333"]
    assert service._reconnect_media_bytes == 0


@pytest.mark.anyio
async def test_connect_waits_until_socket_is_ready(monkeypatch):
    service = object.__new__(ResilientDeepgramSTTService)
    service._name = "TestDeepgramSTT"
    service._connection = None
    service._connection_task = None
    service._connection_ready = asyncio.Event()

    async def connection_handler():
        await asyncio.sleep(0.01)
        service._connection = object()
        service._connection_ready.set()

    service._connection_handler = connection_handler
    service.create_task = lambda coro, name=None: asyncio.create_task(coro, name=name)
    service.cancel_task = lambda task: asyncio.gather(task, return_exceptions=True)
    monkeypatch.setenv("STT_CONNECT_TIMEOUT_SECONDS", "1")

    await service._connect()

    assert service._connection_ready.is_set()
    await service.cancel_task(service._connection_task)


def test_connect_timeout_is_validated(monkeypatch):
    monkeypatch.setenv("STT_CONNECT_TIMEOUT_SECONDS", "60")
    with pytest.raises(ValueError, match="STT_CONNECT_TIMEOUT_SECONDS"):
        ResilientDeepgramSTTService._connect_timeout_seconds()


def test_keepalive_interval_is_validated(monkeypatch):
    monkeypatch.setenv("STT_KEEPALIVE_INTERVAL_SECONDS", "2.5")
    assert ResilientDeepgramSTTService._keepalive_interval_seconds() == 2.5
    monkeypatch.setenv("STT_KEEPALIVE_INTERVAL_SECONDS", "0.1")
    with pytest.raises(ValueError):
        ResilientDeepgramSTTService._keepalive_interval_seconds()


@pytest.mark.anyio
async def test_abandoned_turn_reset_discards_stale_ttfb_clock(monkeypatch):
    service = object.__new__(ResilientDeepgramSTTService)
    service._name = "TestDeepgramSTT"
    service._metrics = type(
        "Metrics",
        (),
        {"_start_ttfb_time": 10.0, "_ttfa_active": True, "_ttfa_buffer": b"audio"},
    )()
    service._last_transcript_time = 12.0

    async def base_reset(_self):
        return None

    monkeypatch.setattr(
        "providers.stt.deepgram_stt.DeepgramSTTService._reset_stt_ttfb_state",
        base_reset,
    )

    await service._reset_stt_ttfb_state()

    assert service._metrics._start_ttfb_time == 0
    assert service._metrics._ttfa_active is False
    assert service._metrics._ttfa_buffer == b""
    assert service._last_transcript_time == 0
