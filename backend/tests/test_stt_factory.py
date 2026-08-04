import sys
from types import SimpleNamespace

import pytest

from providers.stt.factory import (
    get_stt,
    shutdown_stt_provider,
    warm_stt_provider,
)


@pytest.mark.parametrize(
    ("provider", "selected_module", "selected_factory"),
    [
        ("deepgram", "providers.stt.deepgram_stt", "get_deepgram_stt"),
        ("whisper", "providers.local.stt.whisper_stt", "get_whisper_stt"),
        (
            "mlxwhisper",
            "providers.local.stt.mlx_whisper_stt",
            "get_mlx_whisper_stt",
        ),
        (
            "moonshine",
            "providers.local.stt.moonshine_stt",
            "get_moonshine_stt",
        ),
    ],
)
def test_factory_constructs_only_selected_stt(
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
        "providers.stt.deepgram_stt": SimpleNamespace(
            get_deepgram_stt=build("deepgram")
        ),
        "providers.local.stt.whisper_stt": SimpleNamespace(
            get_whisper_stt=build("whisper")
        ),
        "providers.local.stt.mlx_whisper_stt": SimpleNamespace(
            get_mlx_whisper_stt=build("mlxwhisper")
        ),
        "providers.local.stt.moonshine_stt": SimpleNamespace(
            get_moonshine_stt=build("moonshine")
        ),
    }
    for module_name, module in modules.items():
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setenv("STT_PROVIDER", f" {provider.upper()} ")

    assert get_stt() is marker
    assert calls == [provider]
    assert selected_module in sys.modules
    assert hasattr(sys.modules[selected_module], selected_factory)


def test_factory_rejects_unsupported_stt(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "unknown")

    with pytest.raises(
        ValueError,
        match="Expected deepgram, whisper, mlxwhisper, or moonshine",
    ):
        get_stt()


def test_whisper_failure_does_not_fall_back_to_deepgram(monkeypatch):
    deepgram_calls = []

    def fail_whisper():
        raise RuntimeError("model unavailable")

    monkeypatch.setitem(
        sys.modules,
        "providers.local.stt.whisper_stt",
        SimpleNamespace(get_whisper_stt=fail_whisper),
    )
    monkeypatch.setitem(
        sys.modules,
        "providers.stt.deepgram_stt",
        SimpleNamespace(get_deepgram_stt=lambda: deepgram_calls.append(True)),
    )
    monkeypatch.setenv("STT_PROVIDER", "whisper")

    with pytest.raises(RuntimeError, match="model unavailable"):
        get_stt()
    assert deepgram_calls == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("provider", "module_name", "warm_name", "close_name"),
    [
        (
            "whisper",
            "providers.local.stt.whisper_stt",
            "warm_whisper_runtime",
            "shutdown_whisper_runtime",
        ),
        (
            "mlxwhisper",
            "providers.local.stt.mlx_whisper_stt",
            "warm_mlx_whisper_runtime",
            "shutdown_mlx_whisper_runtime",
        ),
        (
            "moonshine",
            "providers.local.stt.moonshine_stt",
            "warm_moonshine_runtime",
            "shutdown_moonshine_runtime",
        ),
    ],
)
async def test_lifecycle_only_warms_and_closes_local_provider(
    monkeypatch,
    provider,
    module_name,
    warm_name,
    close_name,
):
    calls = []

    async def warm():
        calls.append("warm")

    async def close():
        calls.append("close")

    monkeypatch.setitem(
        sys.modules,
        module_name,
        SimpleNamespace(**{warm_name: warm, close_name: close}),
    )
    monkeypatch.setenv("STT_PROVIDER", provider)

    await warm_stt_provider()
    await shutdown_stt_provider()

    assert calls == ["warm", "close"]

    calls.clear()
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    await warm_stt_provider()
    await shutdown_stt_provider()
    assert calls == []
