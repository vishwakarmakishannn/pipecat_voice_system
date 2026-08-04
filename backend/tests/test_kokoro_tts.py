import importlib.util
import os
from types import SimpleNamespace

import pytest
from pipecat.services.tts_service import TextAggregationMode
from pipecat.transcriptions.language import Language

from providers.tts.kokoro_config import (
    KOKORO_MODEL_FILES,
    load_kokoro_config,
    validate_kokoro_runtime,
)
from providers.tts.kokoro_text_aggregator import KokoroTextAggregator


def _clear_model_paths(monkeypatch):
    monkeypatch.delenv("KOKORO_MODEL_PATH", raising=False)
    monkeypatch.delenv("KOKORO_VOICES_PATH", raising=False)
    monkeypatch.delenv("KOKORO_MODEL_PRECISION", raising=False)
    monkeypatch.delenv("KOKORO_CACHE_DIR", raising=False)


def _clear_latency_settings(monkeypatch):
    for name in (
        "KOKORO_LOW_LATENCY",
        "KOKORO_WARMUP_ENABLED",
        "KOKORO_FIRST_CHUNK_CHARS",
        "KOKORO_CHUNK_CHARS",
        "KOKORO_MIN_CHUNK_WORDS",
        "KOKORO_INTRA_OP_THREADS",
        "KOKORO_INTER_OP_THREADS",
        "KOKORO_ALLOW_SPINNING",
        "KOKORO_DOWNLOAD_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_kokoro_config_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("KOKORO_VOICE_ID", raising=False)
    monkeypatch.delenv("KOKORO_LANGUAGE", raising=False)
    _clear_model_paths(monkeypatch)
    _clear_latency_settings(monkeypatch)
    monkeypatch.setenv("KOKORO_CACHE_DIR", str(tmp_path))

    config = load_kokoro_config()

    assert config.voice == "af_heart"
    assert config.language is Language.EN_US
    assert config.precision == "fp16"
    assert config.model_path == tmp_path / "kokoro-v1.0.fp16.onnx"
    assert config.voices_path == tmp_path / "voices-v1.0.bin"
    assert config.model_id == "kokoro-v1.0.fp16.onnx"
    assert config.low_latency_enabled is True
    assert config.warmup_enabled is True
    assert config.first_chunk_chars == 12
    assert config.chunk_chars == 80
    assert config.min_chunk_words == 2
    assert config.intra_op_threads == min(4, os.cpu_count() or 1)
    assert config.inter_op_threads == 1
    assert config.allow_spinning is False


@pytest.mark.parametrize("precision", ["fp32", "fp16", "int8"])
def test_kokoro_precision_selects_cached_model(
    monkeypatch,
    tmp_path,
    precision,
):
    _clear_model_paths(monkeypatch)
    monkeypatch.setenv("KOKORO_MODEL_PRECISION", precision)
    monkeypatch.setenv("KOKORO_CACHE_DIR", str(tmp_path))

    config = load_kokoro_config()

    assert config.precision == precision
    assert config.model_path == tmp_path / KOKORO_MODEL_FILES[precision]


def test_kokoro_config_normalizes_language(monkeypatch):
    monkeypatch.setenv("KOKORO_LANGUAGE", " EN_GB ")
    _clear_model_paths(monkeypatch)

    assert load_kokoro_config().language is Language.EN_GB


def test_kokoro_config_rejects_invalid_language(monkeypatch):
    monkeypatch.setenv("KOKORO_LANGUAGE", "de")
    _clear_model_paths(monkeypatch)

    with pytest.raises(ValueError, match="KOKORO_LANGUAGE"):
        load_kokoro_config()


def test_kokoro_model_paths_must_be_configured_together(monkeypatch, tmp_path):
    monkeypatch.setenv("KOKORO_MODEL_PATH", str(tmp_path / "model.onnx"))
    monkeypatch.delenv("KOKORO_VOICES_PATH", raising=False)

    with pytest.raises(ValueError, match="must be configured together"):
        load_kokoro_config()


def test_kokoro_custom_model_paths_must_exist(monkeypatch, tmp_path):
    model = tmp_path / "model.onnx"
    voices = tmp_path / "voices.bin"
    monkeypatch.setenv("KOKORO_MODEL_PATH", str(model))
    monkeypatch.setenv("KOKORO_VOICES_PATH", str(voices))

    with pytest.raises(ValueError, match="must point to an existing file"):
        load_kokoro_config()

    model.write_bytes(b"model")
    voices.write_bytes(b"voices")
    config = load_kokoro_config()
    assert config.model_path == model
    assert config.voices_path == voices
    assert config.model_id == "model.onnx"


def test_kokoro_runtime_validation_reports_missing_extra(monkeypatch):
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name == "kokoro_onnx" else original_find_spec(name),
    )

    with pytest.raises(ValueError, match=r"pipecat-ai\[kokoro\]"):
        validate_kokoro_runtime()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("KOKORO_LOW_LATENCY", "sometimes"),
        ("KOKORO_WARMUP_ENABLED", "sometimes"),
        ("KOKORO_FIRST_CHUNK_CHARS", "7"),
        ("KOKORO_CHUNK_CHARS", "501"),
        ("KOKORO_MIN_CHUNK_WORDS", "0"),
        ("KOKORO_INTRA_OP_THREADS", "0"),
        ("KOKORO_INTER_OP_THREADS", "17"),
        ("KOKORO_ALLOW_SPINNING", "sometimes"),
        ("KOKORO_DOWNLOAD_TIMEOUT_SECONDS", "9"),
        ("KOKORO_MODEL_PRECISION", "bf16"),
    ],
)
def test_kokoro_latency_configuration_is_validated(monkeypatch, name, value):
    _clear_model_paths(monkeypatch)
    _clear_latency_settings(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        load_kokoro_config()


def test_kokoro_builder_maps_configuration_without_loading_model(monkeypatch):
    from providers.tts import kokoro_tts

    captured = {}

    class FakeLowLatencyKokoroTTSService:
        class Settings:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        kokoro_tts,
        "LowLatencyKokoroTTSService",
        FakeLowLatencyKokoroTTSService,
    )
    runtime = object()
    monkeypatch.setattr(kokoro_tts, "get_kokoro_runtime", lambda config: runtime)
    monkeypatch.setenv("KOKORO_VOICE_ID", "bf_emma")
    monkeypatch.setenv("KOKORO_LANGUAGE", "en-GB")
    monkeypatch.delenv("AUDIO_OUTPUT_SAMPLE_RATE", raising=False)
    monkeypatch.delenv("TTS_TEXT_AGGREGATION_MODE", raising=False)
    _clear_model_paths(monkeypatch)
    _clear_latency_settings(monkeypatch)

    service = kokoro_tts.get_kokoro_tts()

    assert isinstance(service, FakeLowLatencyKokoroTTSService)
    assert captured["runtime"] is runtime
    assert captured["sample_rate"] == 24000
    assert captured["settings"].model == "kokoro-v1.0.fp16.onnx"
    assert captured["settings"].voice == "bf_emma"
    assert captured["settings"].language is Language.EN_GB
    assert captured["text_aggregation_mode"].value == "sentence"
    assert captured["low_latency_enabled"] is True
    assert captured["first_chunk_chars"] == 12
    assert captured["chunk_chars"] == 80
    assert captured["min_chunk_words"] == 2


async def _aggregate(aggregator, text):
    return [item async for item in aggregator.aggregate(text)]


@pytest.mark.anyio
async def test_kokoro_aggregator_releases_short_first_phrase_before_sentence_end():
    aggregator = KokoroTextAggregator(
        first_chunk_chars=24,
        chunk_chars=80,
        min_chunk_words=3,
    )

    chunks = await _aggregate(
        aggregator,
        "AI stands for artificial intelligence and is still generating",
    )

    assert [chunk.text for chunk in chunks] == ["AI stands for artificial"]
    assert aggregator.text.text == "intelligence and is still generating"


@pytest.mark.anyio
async def test_kokoro_aggregator_releases_punctuation_without_lookahead():
    aggregator = KokoroTextAggregator(
        first_chunk_chars=24,
        chunk_chars=80,
        min_chunk_words=3,
    )

    strong = await _aggregate(aggregator, "Yes.")
    clause = await _aggregate(aggregator, "This is ready,")

    assert [chunk.text for chunk in strong] == ["Yes."]
    assert [chunk.text for chunk in clause] == ["This is ready,"]


@pytest.mark.anyio
async def test_kokoro_aggregator_flushes_remainder_and_resets_first_chunk():
    aggregator = KokoroTextAggregator(
        first_chunk_chars=12,
        chunk_chars=40,
        min_chunk_words=2,
    )

    first_turn = await _aggregate(aggregator, "one two three ")
    remainder = await aggregator.flush()
    second_turn = await _aggregate(aggregator, "four five six ")

    assert [chunk.text for chunk in first_turn] == ["one two three"]
    assert remainder is None
    assert [chunk.text for chunk in second_turn] == ["four five six"]


@pytest.mark.anyio
async def test_kokoro_aggregator_discards_partial_text_on_interruption():
    aggregator = KokoroTextAggregator(
        first_chunk_chars=24,
        chunk_chars=80,
        min_chunk_words=3,
    )
    await _aggregate(aggregator, "an unfinished response")

    await aggregator.handle_interruption()

    assert aggregator.text.text == ""
    assert await aggregator.flush() is None


def test_low_latency_service_installs_phrase_aggregator_only_for_sentence_mode(
    monkeypatch,
):
    from providers.tts import kokoro_tts

    original = object()

    def fake_base_init(self, **kwargs):
        self._text_aggregation_mode = kwargs["text_aggregation_mode"]
        self._text_aggregator = original
        self._settings = SimpleNamespace(voice="af_heart", language="en-us")
        self._kokoro = SimpleNamespace()

    monkeypatch.setattr(kokoro_tts.TTSService, "__init__", fake_base_init)
    runtime = SimpleNamespace()

    sentence_service = kokoro_tts.LowLatencyKokoroTTSService(
        runtime=runtime,
        text_aggregation_mode=TextAggregationMode.SENTENCE,
        low_latency_enabled=True,
        first_chunk_chars=24,
        chunk_chars=80,
        min_chunk_words=3,
    )
    token_service = kokoro_tts.LowLatencyKokoroTTSService(
        runtime=runtime,
        text_aggregation_mode=TextAggregationMode.TOKEN,
        low_latency_enabled=True,
        first_chunk_chars=24,
        chunk_chars=80,
        min_chunk_words=3,
    )

    assert isinstance(sentence_service._text_aggregator, KokoroTextAggregator)
    assert token_service._text_aggregator is original
    assert sentence_service._runtime is runtime


def test_low_latency_service_does_not_warm_runtime_during_construction(monkeypatch):
    from providers.tts import kokoro_tts

    def fake_base_init(self, **kwargs):
        self._text_aggregation_mode = kwargs["text_aggregation_mode"]
        self._text_aggregator = object()
        self._settings = SimpleNamespace(voice="af_heart", language="en-us")

    monkeypatch.setattr(kokoro_tts.TTSService, "__init__", fake_base_init)
    runtime = SimpleNamespace(warm=lambda: pytest.fail("must warm at process startup"))

    service = kokoro_tts.LowLatencyKokoroTTSService(
        runtime=runtime,
        text_aggregation_mode=TextAggregationMode.SENTENCE,
        low_latency_enabled=False,
        first_chunk_chars=24,
        chunk_chars=80,
        min_chunk_words=3,
    )

    assert service._runtime is runtime
