import threading
import time
from dataclasses import replace

import numpy as np
import pytest
from pipecat.transcriptions.language import Language

from providers.tts.kokoro_config import KokoroConfig
from providers.tts.kokoro_runtime import KokoroRuntime


def _config(tmp_path, **overrides):
    config = KokoroConfig(
        voice="af_heart",
        language=Language.EN_US,
        precision="fp16",
        model_path=tmp_path / "kokoro-v1.0.fp16.onnx",
        voices_path=tmp_path / "voices-v1.0.bin",
        low_latency_enabled=True,
        warmup_enabled=True,
        first_chunk_chars=12,
        first_chunk_min_words=1,
        chunk_chars=80,
        min_chunk_words=2,
        intra_op_threads=4,
        inter_op_threads=1,
        allow_spinning=False,
        download_timeout_seconds=30.0,
    )
    return replace(config, **overrides)


class FakeModel:
    def __init__(self):
        self.calls = []

    def create(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return np.array([-1.0, 0.0, 0.5], dtype=np.float32), 24000


@pytest.mark.anyio
async def test_runtime_loads_and_warms_model_once(tmp_path):
    config = _config(tmp_path)
    config.model_path.write_bytes(b"model")
    config.voices_path.write_bytes(b"voices")
    model = FakeModel()
    factory_calls = []
    runtime = KokoroRuntime(
        config,
        model_factory=lambda received: factory_calls.append(received) or model,
    )

    await runtime.warm()
    await runtime.warm()
    audio, sample_rate = await runtime.synthesize(
        "Hello.",
        voice="af_heart",
        language="en-us",
    )

    assert runtime.loaded is True
    assert runtime.warmed is True
    assert factory_calls == [config]
    assert [call[0] for call in model.calls] == ["Ready.", "Hello."]
    assert np.frombuffer(audio, dtype=np.int16).tolist() == [-32767, 0, 16383]
    assert sample_rate == 24000
    await runtime.close()


@pytest.mark.anyio
async def test_runtime_downloads_missing_assets_to_shared_cache(tmp_path):
    config = _config(tmp_path, warmup_enabled=False)
    downloads = []

    def downloader(url, destination, timeout):
        downloads.append((url, destination, timeout))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"asset")

    runtime = KokoroRuntime(
        config,
        model_factory=lambda received: FakeModel(),
        downloader=downloader,
    )

    await runtime.warm()

    assert runtime.loaded is True
    assert runtime.warmed is False
    assert [item[1] for item in downloads] == [
        config.model_path,
        config.voices_path,
    ]
    assert downloads[0][0] == config.model_url
    assert downloads[1][0] == config.voices_url
    assert all(item[2] == 30.0 for item in downloads)
    await runtime.close()


@pytest.mark.anyio
async def test_runtime_serializes_concurrent_inference(tmp_path):
    config = _config(tmp_path, warmup_enabled=False)
    config.model_path.write_bytes(b"model")
    config.voices_path.write_bytes(b"voices")
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    class SlowModel(FakeModel):
        def create(self, text, **kwargs):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.01)
            with lock:
                active -= 1
            return super().create(text, **kwargs)

    runtime = KokoroRuntime(config, model_factory=lambda received: SlowModel())

    import asyncio

    await asyncio.gather(
        runtime.synthesize("one", voice="af_heart", language="en-us"),
        runtime.synthesize("two", voice="af_heart", language="en-us"),
    )

    assert maximum_active == 1
    await runtime.close()


@pytest.mark.anyio
async def test_closed_runtime_rejects_new_work(tmp_path):
    config = _config(tmp_path, warmup_enabled=False)
    runtime = KokoroRuntime(config, model_factory=lambda received: FakeModel())
    await runtime.close()

    with pytest.raises(RuntimeError, match="closed"):
        await runtime.warm()
    with pytest.raises(RuntimeError, match="closed"):
        await runtime.synthesize("Hello", voice="af_heart", language="en-us")
