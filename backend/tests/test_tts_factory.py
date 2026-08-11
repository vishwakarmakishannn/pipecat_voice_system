import sys
from types import SimpleNamespace

import pytest

from providers.tts.factory import (
    get_tts,
    shutdown_tts_provider,
    warm_tts_provider,
)


@pytest.mark.parametrize(
    ("provider", "selected_module", "selected_factory"),
    [
        ("cartesia", "providers.tts.cartesia_tts", "get_cartesia_tts"),
        ("deepgram", "providers.tts.deepgram_tts", "get_deepgram_tts"),
        ("kokoro", "providers.tts.kokoro_tts", "get_kokoro_tts"),
        ("piper", "providers.tts.piper_tts", "get_piper_tts"),
    ],
)
def test_factory_constructs_only_selected_tts(
    monkeypatch,
    provider,
    selected_module,
    selected_factory,
):
    calls = []
    marker = object()

    def build(name):
        def factory():
            calls.append(name)
            return marker if name == provider else None

        return factory

    modules = {
        "providers.tts.cartesia_tts": SimpleNamespace(
            get_cartesia_tts=build("cartesia")
        ),
        "providers.tts.deepgram_tts": SimpleNamespace(
            get_deepgram_tts=build("deepgram")
        ),
        "providers.tts.kokoro_tts": SimpleNamespace(
            get_kokoro_tts=build("kokoro")
        ),
        "providers.tts.piper_tts": SimpleNamespace(get_piper_tts=build("piper")),
    }
    for module_name, module in modules.items():
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setenv("TTS_PROVIDER", f" {provider.upper()} ")

    assert get_tts() is marker
    assert calls == [provider]
    assert selected_module in sys.modules
    assert hasattr(sys.modules[selected_module], selected_factory)


def test_factory_rejects_unsupported_tts(monkeypatch):
    monkeypatch.setenv("TTS_PROVIDER", "unknown")

    with pytest.raises(
        ValueError,
        match="Expected cartesia, deepgram, kokoro, or piper",
    ):
        get_tts()


def test_kokoro_failure_does_not_fall_back_to_piper(monkeypatch):
    piper_calls = []

    def fail_kokoro():
        raise RuntimeError("model unavailable")

    monkeypatch.setitem(
        sys.modules,
        "providers.tts.kokoro_tts",
        SimpleNamespace(get_kokoro_tts=fail_kokoro),
    )
    monkeypatch.setitem(
        sys.modules,
        "providers.tts.piper_tts",
        SimpleNamespace(get_piper_tts=lambda: piper_calls.append(True)),
    )
    monkeypatch.setenv("TTS_PROVIDER", "kokoro")

    with pytest.raises(RuntimeError, match="model unavailable"):
        get_tts()
    assert piper_calls == []


@pytest.mark.anyio
async def test_factory_warms_and_stops_selected_kokoro_runtime(monkeypatch):
    calls = []

    def warm_adapter():
        calls.append("adapter")

    async def warm():
        calls.append("warm")

    async def shutdown():
        calls.append("shutdown")

    monkeypatch.setitem(
        sys.modules,
        "providers.tts.kokoro_tts",
        SimpleNamespace(warm_kokoro_adapter=warm_adapter),
    )
    monkeypatch.setitem(
        sys.modules,
        "providers.tts.kokoro_runtime",
        SimpleNamespace(
            warm_kokoro_runtime=warm,
            shutdown_kokoro_runtime=shutdown,
        ),
    )
    monkeypatch.setenv("TTS_PROVIDER", "kokoro")

    await warm_tts_provider()
    await shutdown_tts_provider()

    assert calls == ["adapter", "warm", "shutdown"]


@pytest.mark.anyio
async def test_factory_skips_kokoro_lifecycle_for_cloud_provider(monkeypatch):
    monkeypatch.setenv("TTS_PROVIDER", "cartesia")

    await warm_tts_provider()
    await shutdown_tts_provider()
