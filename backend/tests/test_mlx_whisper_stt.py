import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from pipecat.frames.frames import ErrorFrame, TranscriptionFrame

import providers.local.stt.mlx_config as mlx_config
from providers.local.stt.mlx_config import (
    MLXWhisperConfig,
    load_mlx_whisper_config,
)
from providers.local.stt.mlx_whisper_stt import (
    MLXWhisperRuntime,
    MLXWhisperSTTService,
    _engine_language,
)


def _config(**overrides) -> MLXWhisperConfig:
    values = {
        "model": "mlx-community/whisper-small-mlx",
        "language": "auto",
        "temperature": 0.0,
        "no_speech_threshold": 0.6,
    }
    values.update(overrides)
    return MLXWhisperConfig(**values)


def _set_apple_silicon(monkeypatch):
    monkeypatch.setattr(mlx_config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(mlx_config.platform, "machine", lambda: "arm64")


def test_mlx_whisper_config_defaults(monkeypatch):
    _set_apple_silicon(monkeypatch)
    for name in (
        "MLX_WHISPER_MODEL",
        "MLX_WHISPER_LANGUAGE",
        "MLX_WHISPER_TEMPERATURE",
        "MLX_WHISPER_NO_SPEECH_THRESHOLD",
        "AUDIO_INPUT_SAMPLE_RATE",
    ):
        monkeypatch.delenv(name, raising=False)

    assert load_mlx_whisper_config() == _config()


def test_mlx_whisper_requires_apple_silicon(monkeypatch):
    monkeypatch.setattr(mlx_config.platform, "system", lambda: "Linux")

    with pytest.raises(ValueError, match="requires an Apple Silicon Mac"):
        load_mlx_whisper_config()


def test_mlx_whisper_requires_16khz_input(monkeypatch):
    _set_apple_silicon(monkeypatch)
    monkeypatch.setenv("AUDIO_INPUT_SAMPLE_RATE", "24000")

    with pytest.raises(ValueError, match="must be 16000"):
        load_mlx_whisper_config()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MLX_WHISPER_MODEL", " "),
        ("MLX_WHISPER_LANGUAGE", " "),
        ("MLX_WHISPER_TEMPERATURE", "fast"),
        ("MLX_WHISPER_TEMPERATURE", "1.1"),
        ("MLX_WHISPER_NO_SPEECH_THRESHOLD", "-0.1"),
    ],
)
def test_mlx_whisper_settings_are_validated(monkeypatch, name, value):
    _set_apple_silicon(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        load_mlx_whisper_config()


def test_auto_language_uses_mlx_detection_sentinel():
    assert _engine_language("auto") is None
    assert _engine_language("hi") == "hi"


class _FakeRuntime:
    def __init__(self, result=None, error=None):
        self.config = _config()
        self.result = result or {}
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
async def test_mlx_service_normalizes_pcm_and_joins_segments(monkeypatch):
    runtime = _FakeRuntime(
        {
            "text": "fallback",
            "segments": [
                {"text": " first phrase "},
                {"text": ""},
                {"text": "second phrase"},
            ],
        }
    )
    service = MLXWhisperSTTService(runtime=runtime, config=runtime.config)
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
async def test_mlx_service_drops_silence_without_inference(monkeypatch):
    runtime = _FakeRuntime({"text": "should not be used"})
    service = MLXWhisperSTTService(runtime=runtime, config=runtime.config)
    _disable_service_metrics(monkeypatch, service)

    assert [frame async for frame in service.run_stt(b"\0\0")] == []
    assert runtime.audio is None


@pytest.mark.anyio
async def test_mlx_service_emits_fatal_error_without_fallback(monkeypatch):
    failure = RuntimeError("inference failed")
    runtime = _FakeRuntime(error=failure)
    service = MLXWhisperSTTService(runtime=runtime, config=runtime.config)
    _disable_service_metrics(monkeypatch, service)

    audio = np.array([100, -100], dtype=np.int16).tobytes()
    frames = [frame async for frame in service.run_stt(audio)]

    assert len(frames) == 1
    assert isinstance(frames[0], ErrorFrame)
    assert frames[0].fatal is True
    assert frames[0].exception is failure
    assert "inference failed" not in frames[0].error


@pytest.mark.anyio
async def test_mlx_runtime_warms_once_and_serializes_calls():
    active = 0
    max_active = 0
    calls = 0
    state_lock = threading.Lock()

    def transcriber(_audio, _config):
        nonlocal active, calls, max_active
        with state_lock:
            calls += 1
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with state_lock:
            active -= 1
        return {"text": "ok", "segments": [{"text": "ok"}]}

    runtime = MLXWhisperRuntime(_config(), transcriber=transcriber)
    await runtime.warm()
    await runtime.warm()
    first = asyncio.create_task(
        runtime.transcribe(np.zeros(160, dtype=np.float32))
    )
    second = asyncio.create_task(
        runtime.transcribe(np.zeros(160, dtype=np.float32))
    )
    results = await asyncio.gather(first, second)
    await runtime.close()

    assert runtime.warmed is True
    assert calls == 3
    assert max_active == 1
    assert results == [
        {"text": "ok", "segments": [{"text": "ok"}]},
        {"text": "ok", "segments": [{"text": "ok"}]},
    ]


@pytest.mark.anyio
async def test_closed_mlx_runtime_rejects_work():
    runtime = MLXWhisperRuntime(
        _config(),
        transcriber=lambda _audio, _config: {},
    )
    await runtime.close()

    with pytest.raises(RuntimeError, match="closed"):
        await runtime.warm()
    with pytest.raises(RuntimeError, match="closed"):
        await runtime.transcribe(np.zeros(1, dtype=np.float32))
