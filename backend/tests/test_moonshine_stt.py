import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from providers.local.stt.moonshine_config import (
    MoonshineConfig,
    load_moonshine_config,
)
from providers.local.stt.moonshine_stt import (
    MoonshineRuntime,
    MoonshineSTTService,
)


def _config(**overrides) -> MoonshineConfig:
    values = {
        "model": "medium-streaming",
        "language": "en",
        "update_interval_seconds": 0.25,
        "ttfs_p99_latency_seconds": 0.75,
        "model_dir": None,
        "cache_dir": None,
    }
    values.update(overrides)
    return MoonshineConfig(**values)


def test_moonshine_config_defaults(monkeypatch):
    for name in (
        "MOONSHINE_MODEL",
        "MOONSHINE_LANGUAGE",
        "MOONSHINE_UPDATE_INTERVAL_SECONDS",
        "MOONSHINE_TTFS_P99_SECONDS",
        "MOONSHINE_MODEL_DIR",
        "MOONSHINE_VOICE_CACHE",
        "AUDIO_INPUT_SAMPLE_RATE",
    ):
        monkeypatch.delenv(name, raising=False)

    assert load_moonshine_config() == _config()


def test_moonshine_v2_medium_alias_is_canonicalized(monkeypatch):
    monkeypatch.setenv("MOONSHINE_MODEL", "moonshine-v2-medium")

    assert load_moonshine_config().model == "medium-streaming"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("MOONSHINE_MODEL", "base-streaming", "245M streaming model"),
        ("MOONSHINE_LANGUAGE", "hi", "English-only"),
        ("MOONSHINE_UPDATE_INTERVAL_SECONDS", "fast", "must be a number"),
        ("MOONSHINE_UPDATE_INTERVAL_SECONDS", "0.05", "between 0.1 and 2.0"),
        ("MOONSHINE_TTFS_P99_SECONDS", "fast", "must be a number"),
        ("MOONSHINE_TTFS_P99_SECONDS", "0.01", "between 0.05 and 5.0"),
        ("AUDIO_INPUT_SAMPLE_RATE", "24000", "must be 16000"),
    ],
)
def test_moonshine_config_rejects_invalid_values(
    monkeypatch,
    name,
    value,
    message,
):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        load_moonshine_config()


def test_moonshine_model_dir_rejects_a_file(monkeypatch, tmp_path):
    model_file = tmp_path / "not-a-directory"
    model_file.write_text("x", encoding="utf-8")
    monkeypatch.setenv("MOONSHINE_MODEL_DIR", str(model_file))

    with pytest.raises(ValueError, match="must be a directory"):
        load_moonshine_config()


class LineTextChanged:
    def __init__(self, line):
        self.line = line


class LineCompleted:
    def __init__(self, line):
        self.line = line


class Error:
    def __init__(self, error):
        self.error = error


def _line(text, *, complete=False, line_id=7, latency_ms=18):
    return SimpleNamespace(
        text=text,
        is_complete=complete,
        line_id=line_id,
        last_transcription_latency_ms=latency_ms,
    )


class _FakeStream:
    def __init__(self, *, error=None):
        self.listener = None
        self.error = error
        self.audio = []
        self.sample_rates = []
        self.starts = 0
        self.stops = 0
        self.closed = False

    def add_listener(self, listener):
        self.listener = listener

    def remove_listener(self, listener):
        if self.listener == listener:
            self.listener = None

    def start(self):
        self.starts += 1

    def stop(self):
        self.stops += 1
        if self.listener:
            completed = _line("hello world", complete=True)
            self.listener(LineTextChanged(completed))
            self.listener(LineCompleted(completed))

    def add_audio(self, audio, sample_rate):
        self.audio.append(audio)
        self.sample_rates.append(sample_rate)
        if self.error:
            self.listener(Error(self.error))
            return
        interim = _line("hello")
        self.listener(LineTextChanged(interim))
        self.listener(LineTextChanged(interim))

    def close(self):
        self.closed = True
        self.listener = None


class _FakeRuntime:
    def __init__(self, config=None, *, error=None):
        self.config = config or _config()
        self.error = error
        self.streams = []

    def create_stream(self):
        stream = _FakeStream(error=self.error)
        self.streams.append(stream)
        return stream


def _disable_service_metrics(monkeypatch, service):
    async def no_metrics(*args, **kwargs):
        return None

    monkeypatch.setattr(service, "start_processing_metrics", no_metrics)
    monkeypatch.setattr(service, "stop_processing_metrics", no_metrics)
    monkeypatch.setattr(service, "stop_all_metrics", no_metrics)


@pytest.mark.anyio
async def test_moonshine_stream_emits_interim_then_graceful_final(monkeypatch):
    runtime = _FakeRuntime()
    service = MoonshineSTTService(runtime=runtime, config=runtime.config)
    service._user_id = "speaker-1"
    _disable_service_metrics(monkeypatch, service)
    audio = np.array([-32768, 0, 16384, 32767], dtype=np.int16).tobytes()

    assert service.service_metadata_frame().ttfs_p99_latency == 0.75
    interim = [frame async for frame in service.run_stt(audio)]
    await service._run_native(service._finish_stream_sync)
    final = await service._pending_frames()
    stream = runtime.streams[0]
    await service.cleanup()

    assert len(interim) == 1
    assert isinstance(interim[0], InterimTranscriptionFrame)
    assert interim[0].text == "hello"
    assert interim[0].user_id == "speaker-1"
    assert len(final) == 1
    assert isinstance(final[0], TranscriptionFrame)
    assert final[0].text == "hello world"
    assert final[0].finalized is True
    assert final[0].result == {"line_id": 7, "latency_ms": 18}
    assert stream.sample_rates == [16000]
    np.testing.assert_allclose(
        stream.audio[0],
        [-1.0, 0.0, 0.5, 32767 / 32768],
    )
    assert stream.starts == 1
    assert stream.stops == 1
    assert stream.closed is True


@pytest.mark.anyio
async def test_moonshine_accepts_multiple_completed_lines_on_one_stream(monkeypatch):
    runtime = _FakeRuntime()
    service = MoonshineSTTService(runtime=runtime, config=runtime.config)
    _disable_service_metrics(monkeypatch, service)

    await service._run_native(service._ensure_stream_sync)
    stream = runtime.streams[0]
    stream.listener(LineCompleted(_line("first", complete=True, line_id=7)))
    first = await service._pending_frames()
    stream.listener(LineCompleted(_line("second", complete=True, line_id=8)))
    second = await service._pending_frames()
    await service.cleanup()

    assert [frame.text for frame in first] == ["first"]
    assert [frame.text for frame in second] == ["second"]
    assert stream.starts == 1
    assert stream.stops == 0


@pytest.mark.anyio
async def test_vad_stop_keeps_one_continuous_moonshine_stream(monkeypatch):
    runtime = _FakeRuntime()
    service = MoonshineSTTService(runtime=runtime, config=runtime.config)
    _disable_service_metrics(monkeypatch, service)
    pushed = []

    async def capture(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append((frame, direction))

    monkeypatch.setattr(service, "push_frame", capture)
    await service.start(StartFrame(audio_in_sample_rate=16000))
    await service._handle_vad_user_stopped_speaking(
        VADUserStoppedSpeakingFrame(stop_secs=0)
    )
    trailing = [
        frame
        async for frame in service.run_stt(
            np.array([100, -100], dtype=np.int16).tobytes()
        )
    ]
    stream = runtime.streams[0]

    assert pushed == []
    assert stream.starts == 1
    assert stream.stops == 0
    assert stream.closed is False
    assert len(runtime.streams) == 1
    assert len(stream.audio) == 1
    assert [frame.text for frame in trailing] == ["hello"]

    await service.cleanup()

    assert stream.closed is True


@pytest.mark.anyio
async def test_session_end_flushes_final_transcript_downstream(monkeypatch):
    runtime = _FakeRuntime()
    service = MoonshineSTTService(runtime=runtime, config=runtime.config)
    _disable_service_metrics(monkeypatch, service)
    pushed = []

    async def capture(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append((frame, direction))

    monkeypatch.setattr(service, "push_frame", capture)
    await service.start(StartFrame(audio_in_sample_rate=16000))
    stream = runtime.streams[0]
    await service.stop(EndFrame())
    await service.cleanup()

    assert stream.starts == 1
    assert stream.stops == 1
    assert stream.closed is True
    transcripts = [
        item for item in pushed if isinstance(item[0], TranscriptionFrame)
    ]
    assert len(transcripts) == 1
    assert transcripts[0][0].text == "hello world"
    assert transcripts[0][0].finalized is True
    assert transcripts[0][1] == FrameDirection.DOWNSTREAM


@pytest.mark.anyio
async def test_moonshine_stream_error_is_fatal_and_sanitized(monkeypatch):
    failure = RuntimeError("native details")
    runtime = _FakeRuntime(error=failure)
    service = MoonshineSTTService(runtime=runtime, config=runtime.config)
    _disable_service_metrics(monkeypatch, service)

    frames = [
        frame
        async for frame in service.run_stt(
            np.array([100, -100], dtype=np.int16).tobytes()
        )
    ]
    await service.cleanup()

    assert len(frames) == 1
    assert isinstance(frames[0], ErrorFrame)
    assert frames[0].fatal is True
    assert frames[0].exception is failure
    assert "native details" not in frames[0].error


@pytest.mark.anyio
async def test_moonshine_runtime_loads_one_model_for_multiple_streams():
    loads = []

    class FakeTranscriber:
        def __init__(self):
            self.streams = []
            self.closed = False

        def create_stream(self, update_interval):
            stream = SimpleNamespace(update_interval=update_interval)
            self.streams.append(stream)
            return stream

        def close(self):
            self.closed = True

    transcriber = FakeTranscriber()

    def resolver(config):
        loads.append(("resolve", config.model))
        return "/model", "medium-arch"

    def factory(path, architecture, interval):
        loads.append(("load", path, architecture, interval))
        return transcriber

    runtime = MoonshineRuntime(
        _config(),
        model_resolver=resolver,
        transcriber_factory=factory,
    )
    await asyncio.gather(runtime.warm(), runtime.warm())

    first = runtime.create_stream()
    second = runtime.create_stream()
    await runtime.close()

    assert loads == [
        ("resolve", "medium-streaming"),
        ("load", "/model", "medium-arch", 0.25),
    ]
    assert first is not second
    assert first.update_interval == 0.25
    assert second.update_interval == 0.25
    assert transcriber.closed is True


@pytest.mark.anyio
async def test_closed_moonshine_runtime_rejects_work():
    runtime = MoonshineRuntime(
        _config(),
        model_resolver=lambda _config: ("/model", "medium-arch"),
        transcriber_factory=lambda *_args: SimpleNamespace(close=lambda: None),
    )
    await runtime.close()

    with pytest.raises(RuntimeError, match="closed"):
        await runtime.warm()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.create_stream()
