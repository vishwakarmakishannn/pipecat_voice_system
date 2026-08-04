import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from pipecat.frames.frames import ErrorFrame, TranscriptionFrame

from providers.local.stt.config import WhisperConfig, load_whisper_config
from providers.local.stt.whisper_stt import (
    WhisperRuntime,
    WhisperSTTService,
    _engine_language,
)


def _config(**overrides) -> WhisperConfig:
    values = {
        "model": "small",
        "language": "auto",
        "threads": 4,
        "models_dir": None,
    }
    values.update(overrides)
    return WhisperConfig(**values)


def test_whisper_config_defaults(monkeypatch):
    for name in (
        "WHISPER_MODEL",
        "WHISPER_LANGUAGE",
        "WHISPER_THREADS",
        "WHISPER_MODELS_DIR",
        "AUDIO_INPUT_SAMPLE_RATE",
    ):
        monkeypatch.delenv(name, raising=False)

    assert load_whisper_config() == _config()


def test_auto_language_uses_whispercpp_detection_sentinel():
    assert _engine_language("auto") == ""
    assert _engine_language("hi") == "hi"


@pytest.mark.parametrize("name", ["WHISPER_MODEL", "WHISPER_LANGUAGE"])
def test_whisper_text_settings_must_not_be_empty(monkeypatch, name):
    monkeypatch.setenv(name, "   ")

    with pytest.raises(ValueError, match=name):
        load_whisper_config()


@pytest.mark.parametrize("value", ["0", "65", "many"])
def test_whisper_thread_count_is_validated(monkeypatch, value):
    monkeypatch.setenv("WHISPER_THREADS", value)

    with pytest.raises(ValueError, match="WHISPER_THREADS"):
        load_whisper_config()


def test_whisper_requires_16khz_input(monkeypatch):
    monkeypatch.setenv("AUDIO_INPUT_SAMPLE_RATE", "24000")

    with pytest.raises(ValueError, match="must be 16000"):
        load_whisper_config()


def test_whisper_models_dir_rejects_a_file(monkeypatch, tmp_path):
    model_file = tmp_path / "not-a-directory"
    model_file.write_text("x", encoding="utf-8")
    monkeypatch.setenv("WHISPER_MODELS_DIR", str(model_file))

    with pytest.raises(ValueError, match="must be a directory"):
        load_whisper_config()


class _FakeRuntime:
    def __init__(self, result=None, error=None):
        self.config = _config()
        self.result = result or []
        self.error = error
        self.audio = None

    async def transcribe(self, audio):
        self.audio = audio
        if self.error:
            raise self.error
        return self.result


def _disable_service_metrics(monkeypatch, service):
    async def no_metrics():
        return None

    monkeypatch.setattr(service, "start_processing_metrics", no_metrics)
    monkeypatch.setattr(service, "stop_processing_metrics", no_metrics)


@pytest.mark.anyio
async def test_whisper_service_normalizes_pcm_and_joins_all_segments(monkeypatch):
    runtime = _FakeRuntime(
        [
            SimpleNamespace(text=" first phrase "),
            SimpleNamespace(text=""),
            SimpleNamespace(text="second phrase"),
        ]
    )
    service = WhisperSTTService(runtime=runtime, config=runtime.config)
    service._user_id = "speaker-1"
    _disable_service_metrics(monkeypatch, service)
    audio = np.array([-32768, 0, 16384, 32767], dtype=np.int16).tobytes()

    assert service._settings.language is None
    frames = [frame async for frame in service.run_stt(audio)]

    assert len(frames) == 1
    assert isinstance(frames[0], TranscriptionFrame)
    assert frames[0].text == "first phrase second phrase"
    assert frames[0].user_id == "speaker-1"
    assert frames[0].finalized is True
    np.testing.assert_allclose(
        runtime.audio,
        np.array([-1.0, 0.0, 0.5, 32767 / 32768], dtype=np.float32),
    )


@pytest.mark.anyio
async def test_whisper_service_drops_empty_transcript(monkeypatch):
    runtime = _FakeRuntime([SimpleNamespace(text="  ")])
    service = WhisperSTTService(runtime=runtime, config=runtime.config)
    _disable_service_metrics(monkeypatch, service)

    assert [frame async for frame in service.run_stt(b"\0\0")] == []
    assert runtime.audio is None


@pytest.mark.anyio
async def test_whisper_service_drops_known_non_speech_marker(monkeypatch):
    runtime = _FakeRuntime([SimpleNamespace(text="[BLANK_AUDIO]")])
    service = WhisperSTTService(runtime=runtime, config=runtime.config)
    _disable_service_metrics(monkeypatch, service)

    frames = [
        frame
        async for frame in service.run_stt(
            np.array([100, -100], dtype=np.int16).tobytes()
        )
    ]

    assert frames == []


@pytest.mark.anyio
async def test_whisper_service_emits_fatal_error_without_fallback(monkeypatch):
    failure = RuntimeError("inference failed")
    runtime = _FakeRuntime(error=failure)
    service = WhisperSTTService(runtime=runtime, config=runtime.config)
    _disable_service_metrics(monkeypatch, service)

    audio = np.array([100, -100], dtype=np.int16).tobytes()
    frames = [frame async for frame in service.run_stt(audio)]

    assert len(frames) == 1
    assert isinstance(frames[0], ErrorFrame)
    assert frames[0].fatal is True
    assert frames[0].exception is failure
    assert "inference failed" not in frames[0].error


@pytest.mark.anyio
async def test_runtime_loads_once_serializes_calls_and_keeps_loop_responsive():
    loads = 0
    active = 0
    max_active = 0
    state_lock = threading.Lock()

    class FakeModel:
        def transcribe(self, _audio):
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.04)
            with state_lock:
                active -= 1
            return [SimpleNamespace(text="ok")]

    def model_factory(_config):
        nonlocal loads
        loads += 1
        return FakeModel()

    runtime = WhisperRuntime(_config(), model_factory=model_factory)
    await runtime.warm()
    loop_ticked = asyncio.Event()

    async def tick():
        await asyncio.sleep(0.005)
        loop_ticked.set()

    first = asyncio.create_task(runtime.transcribe(np.zeros(160, dtype=np.float32)))
    second = asyncio.create_task(runtime.transcribe(np.zeros(160, dtype=np.float32)))
    ticker = asyncio.create_task(tick())
    await asyncio.wait_for(loop_ticked.wait(), timeout=0.03)
    results = await asyncio.gather(first, second, ticker)
    await runtime.close()

    assert loads == 1
    assert max_active == 1
    assert [segment.text for segment in results[0]] == ["ok"]
    assert [segment.text for segment in results[1]] == ["ok"]


@pytest.mark.anyio
async def test_runtime_warm_propagates_model_load_failure():
    def fail_load(_config):
        raise RuntimeError("download failed")

    runtime = WhisperRuntime(_config(), model_factory=fail_load)
    try:
        with pytest.raises(RuntimeError, match="download failed"):
            await runtime.warm()
    finally:
        await runtime.close()


@pytest.mark.anyio
async def test_closed_runtime_rejects_work():
    runtime = WhisperRuntime(
        _config(),
        model_factory=lambda _config: SimpleNamespace(transcribe=lambda _audio: []),
    )
    await runtime.close()

    with pytest.raises(RuntimeError, match="closed"):
        await runtime.warm()
    with pytest.raises(RuntimeError, match="closed"):
        await runtime.transcribe(np.zeros(1, dtype=np.float32))
