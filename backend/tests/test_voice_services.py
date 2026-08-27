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
async def test_session_is_authenticated_before_services_are_constructed(monkeypatch):
    events = []

    async def fake_services(*_factories):
        events.append("services")
        return "stt", "tts", "llm"

    async def load_session(body):
        assert body == {"token": "x"}
        events.append("session")
        return "session"

    monkeypatch.setattr(voice_services, "initialize_voice_services", fake_services)
    services, session = await voice_services.initialize_voice_runtime(
        lambda: None, lambda: None, lambda: None,
        load_session, {"token": "x"},
    )

    assert services == ("stt", "tts", "llm")
    assert session == "session"
    assert events == ["session", "services"]


@pytest.mark.anyio
async def test_invalid_session_does_not_construct_providers(monkeypatch):
    constructed = False

    async def fake_services(*_factories):
        nonlocal constructed
        constructed = True

    async def reject_session(_body):
        return None

    monkeypatch.setattr(voice_services, "initialize_voice_services", fake_services)

    with pytest.raises(
        voice_services.VoiceSessionAuthenticationError,
        match="authenticated voice session token",
    ):
        await voice_services.initialize_voice_runtime(
            lambda: None,
            lambda: None,
            lambda: None,
            reject_session,
            {},
        )

    assert constructed is False


@pytest.mark.anyio
async def test_hydrated_session_builds_bound_llm_while_stt_and_tts_overlap(
    monkeypatch,
):
    events = []

    async def fake_construct(name, factory):
        events.append(f"start:{name}")
        value = factory()
        events.append(f"done:{name}")
        return value

    async def load_session(_body):
        events.append("authenticated")
        return "base-session"

    async def hydrate(session):
        assert session == "base-session"
        events.append("hydrated")
        return "hydrated-session"

    monkeypatch.setattr(voice_services, "_construct_voice_service", fake_construct)

    services, session = await voice_services.initialize_voice_runtime(
        lambda: "stt",
        lambda: "tts",
        lambda: "unbound-llm",
        load_session,
        {},
        session_hydrator=hydrate,
        session_llm_factory=lambda hydrated: f"llm:{hydrated}",
    )

    assert services == ("stt", "tts", "llm:hydrated-session")
    assert session == "hydrated-session"
    assert events[0] == "authenticated"
    assert "hydrated" in events
    assert events.index("hydrated") < events.index("start:llm")


@pytest.mark.anyio
async def test_provider_failure_retains_authenticated_call_identity(monkeypatch):
    session = object()

    async def load_session(_body):
        return session

    async def hydrate(_session):
        return session

    async def construct(name, factory):
        if name == "llm":
            raise RuntimeError("provider startup failed")
        return factory()

    monkeypatch.setattr(voice_services, "_construct_voice_service", construct)
    with pytest.raises(RuntimeError) as captured:
        await voice_services.initialize_voice_runtime(
            lambda: "stt",
            lambda: "tts",
            lambda: "llm",
            load_session,
            {},
            session_hydrator=hydrate,
            session_llm_factory=lambda _session: "llm",
        )
    assert captured.value.voice_session is session
