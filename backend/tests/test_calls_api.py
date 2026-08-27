import uuid
from datetime import datetime, timedelta, timezone

import jwt
import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select, text

from api.auth import ALGORITHM, SECRET_KEY
from api.calls import (
    ClientEventInput,
    create_client_event,
    get_call,
    get_call_timeline,
    list_calls,
    stream_call_recording,
)
from core.database import AsyncSessionLocal, engine
from core.models import Call, CallRecording, User
from core.recording_config import local_recording_dir
from services.calls import (
    finalize_call,
    save_call_event,
    save_call_operation,
    save_transcript_entry,
)


pytestmark = pytest.mark.database


@pytest.fixture(autouse=True)
async def fresh_engine_pool():
    await engine.dispose()
    yield
    await engine.dispose()


async def _user_and_call(username: str) -> tuple[User, Call]:
    async with AsyncSessionLocal() as db:
        user = User(username=f"{username}-{uuid.uuid4().hex}", password_hash="test")
        db.add(user)
        await db.flush()
        call = Call(user_id=user.id, title=f"{username} call")
        db.add(call)
        await db.commit()
        await db.refresh(user)
        await db.refresh(call)
        return user, call


async def _delete_users(*user_ids: int) -> None:
    async with AsyncSessionLocal() as db:
        await db.execute(text("SET LOCAL aura.allow_call_purge = 'on'"))
        result = await db.execute(select(User).where(User.id.in_(user_ids)))
        for user in result.scalars().all():
            await db.delete(user)
        await db.commit()


@pytest.mark.anyio
async def test_call_list_and_detail_are_owner_scoped_and_hide_object_keys(monkeypatch):
    owner, call = await _user_and_call("owner")
    stranger, _ = await _user_and_call("stranger")
    try:
        async with AsyncSessionLocal() as db:
            db.add(CallRecording(
                call_id=call.id,
                status="available",
                object_key=f"calls/{owner.id}/{call.id}.mp3",
            ))
            await db.commit()
            page = await list_calls(
                current_user=owner,
                db=db,
                cursor=None,
                limit=30,
                call_status=None,
                started_from=None,
                started_to=None,
                provider=None,
                model=None,
                recording_status=None,
                has_errors=None,
                deleted=False,
            )
            assert [item["id"] for item in page["items"]] == [str(call.id)]
            detail = await get_call(call.id, current_user=owner, db=db)
            assert detail["recording"]["status"] == "available"
            assert "object_key" not in detail["recording"]
            with pytest.raises(HTTPException) as denied:
                await get_call(call.id, current_user=stranger, db=db)
            assert denied.value.status_code == 404
    finally:
        await _delete_users(owner.id, stranger.id)


@pytest.mark.anyio
async def test_timeline_cursor_has_no_fixed_history_loss(monkeypatch):
    owner, call = await _user_and_call("timeline")
    monkeypatch.setattr("services.calls.task_queue.enqueue", lambda *_args, **_kwargs: True)
    try:
        for index in range(5):
            await save_transcript_entry(
                call.id,
                "You" if index % 2 == 0 else "Aura",
                f"entry {index}",
                source="typed_user" if index % 2 == 0 else "llm",
                turn_id=index + 1,
            )
        async with AsyncSessionLocal() as db:
            first = await get_call_timeline(
                call.id, current_user=owner, db=db, after=0, limit=2
            )
            second = await get_call_timeline(
                call.id,
                current_user=owner,
                db=db,
                after=first["next_cursor"],
                limit=2,
            )
            third = await get_call_timeline(
                call.id,
                current_user=owner,
                db=db,
                after=second["next_cursor"],
                limit=2,
            )
        items = first["items"] + second["items"] + third["items"]
        assert [item["text"] for item in items] == [f"entry {index}" for index in range(5)]
        assert third["next_cursor"] is None
    finally:
        await _delete_users(owner.id)


@pytest.mark.anyio
async def test_timeline_tool_deduplication_survives_page_boundaries(monkeypatch):
    owner, call = await _user_and_call("timeline-dedup")
    monkeypatch.setattr("services.calls.task_queue.enqueue", lambda *_args, **_kwargs: True)
    try:
        await save_call_operation(
            call.id,
            operation_type="tool",
            name="tavily_search",
            arguments={"query": "current news"},
            result={"status": "timeout"},
            status="failed",
            request_id="tool-request-1",
            error_code="tool.execution_timeout",
        )
        await save_call_event(
            call.id,
            component="tool",
            code="tool.execution_timeout",
            severity="error",
            outcome="degraded",
            safe_message="The search timed out.",
            request_id="tool-request-1",
        )
        await save_transcript_entry(
            call.id,
            "Aura",
            "Live results were unavailable.",
            source="spoken_recovery",
            turn_id=1,
        )
        async with AsyncSessionLocal() as db:
            first = await get_call_timeline(
                call.id, current_user=owner, db=db, after=0, limit=1
            )
            second = await get_call_timeline(
                call.id,
                current_user=owner,
                db=db,
                after=first["next_cursor"],
                limit=1,
            )
        assert [item["item_type"] for item in first["items"]] == ["operation"]
        assert [item["item_type"] for item in second["items"]] == ["transcript"]
        assert second["next_cursor"] is None
    finally:
        await _delete_users(owner.id)


@pytest.mark.anyio
async def test_client_diagnostics_are_allowlisted_bounded_and_terminal_safe():
    owner, call = await _user_and_call("diagnostic")
    try:
        async with AsyncSessionLocal() as db:
            with pytest.raises(HTTPException) as unsupported:
                await create_client_event(
                    call.id,
                    ClientEventInput(code="browser.secret", message="bad"),
                    current_user=owner,
                    db=db,
                )
            assert unsupported.value.status_code == 422
            with pytest.raises(HTTPException) as oversized:
                await create_client_event(
                    call.id,
                    ClientEventInput(
                        code="transport.microphone_failed",
                        message="failed",
                        details={"value": "x" * 5000},
                    ),
                    current_user=owner,
                    db=db,
                )
            assert oversized.value.status_code == 413
        await finalize_call(call.id)
        async with AsyncSessionLocal() as db:
            with pytest.raises(HTTPException) as immutable:
                await create_client_event(
                    call.id,
                    ClientEventInput(
                        code="transport.connection_lost",
                        message="late",
                    ),
                    current_user=owner,
                    db=db,
                )
            assert immutable.value.status_code == 409
    finally:
        await _delete_users(owner.id)


@pytest.mark.anyio
async def test_signed_local_recording_access_enforces_owner(monkeypatch, tmp_path):
    monkeypatch.setenv("RECORDING_STORAGE_DIR", str(tmp_path))
    owner, call = await _user_and_call("media-owner")
    stranger, _ = await _user_and_call("media-stranger")
    object_key = f"calls/{owner.id}/{call.id}.mp3"
    path = local_recording_dir() / object_key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"ID3test")
    try:
        async with AsyncSessionLocal() as db:
            db.add(CallRecording(call_id=call.id, status="available", object_key=object_key))
            await db.commit()
        expires = datetime.now(timezone.utc) + timedelta(minutes=5)
        token = jwt.encode(
            {"sub": str(owner.id), "call_id": str(call.id), "aud": "call-recording", "exp": expires},
            SECRET_KEY,
            algorithm=ALGORITHM,
        )
        wrong_token = jwt.encode(
            {"sub": str(stranger.id), "call_id": str(call.id), "aud": "call-recording", "exp": expires},
            SECRET_KEY,
            algorithm=ALGORITHM,
        )
        async with AsyncSessionLocal() as db:
            response = await stream_call_recording(call.id, token, db)
            assert response.media_type == "audio/mpeg"
            with pytest.raises(HTTPException) as denied:
                await stream_call_recording(call.id, wrong_token, db)
            assert denied.value.status_code == 404
    finally:
        await _delete_users(owner.id, stranger.id)


@pytest.mark.anyio
async def test_local_recording_response_supports_http_range(tmp_path):
    recording = tmp_path / "recording.mp3"
    recording.write_bytes(b"ID3-0123456789")
    app = FastAPI()

    @app.get("/recording")
    async def media():
        return FileResponse(recording, media_type="audio/mpeg")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/recording", headers={"Range": "bytes=4-7"})

    assert response.status_code == 206
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-range"] == "bytes 4-7/14"
    assert response.content == b"0123"
