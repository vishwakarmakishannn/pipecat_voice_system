import asyncio

import pytest

import core.voice_services as voice_services


@pytest.mark.anyio
async def test_voice_service_constructors_are_scheduled_together(monkeypatch):
    scheduled = []
    release = asyncio.Event()
    all_scheduled = asyncio.Event()

    async def fake_to_thread(factory):
        scheduled.append(factory.__name__)
        if len(scheduled) == 3:
            all_scheduled.set()
        await release.wait()
        return factory()

    monkeypatch.setattr(voice_services.asyncio, "to_thread", fake_to_thread)
    task = asyncio.create_task(
        voice_services.initialize_voice_services(
            lambda: "stt",
            lambda: "tts",
            lambda: "llm",
        )
    )
    await asyncio.wait_for(all_scheduled.wait(), timeout=0.2)
    assert len(scheduled) == 3
    release.set()
    assert await task == ("stt", "tts", "llm")


@pytest.mark.anyio
async def test_session_identity_and_services_start_concurrently(monkeypatch):
    services_started = asyncio.Event()
    session_started = asyncio.Event()

    async def fake_services(*_factories):
        services_started.set()
        await asyncio.wait_for(session_started.wait(), timeout=0.2)
        return "stt", "tts", "llm"

    async def load_session(body):
        assert body == {"token": "x"}
        session_started.set()
        await asyncio.wait_for(services_started.wait(), timeout=0.2)
        return "session"

    monkeypatch.setattr(voice_services, "initialize_voice_services", fake_services)
    services, session = await voice_services.initialize_voice_runtime(
        lambda: None, lambda: None, lambda: None,
        load_session, {"token": "x"},
    )

    assert services == ("stt", "tts", "llm")
    assert session == "session"
