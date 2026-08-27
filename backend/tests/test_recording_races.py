from types import SimpleNamespace
import uuid

import pytest

import api.calls as calls_api
import services.recordings as recordings


@pytest.mark.anyio
async def test_delete_winning_during_finalize_removes_uploaded_object(
    monkeypatch, tmp_path
):
    call_id = uuid.uuid4()
    spool = tmp_path / f"{call_id}.pcm"
    spool.write_bytes(b"\x00\x00" * 100)
    transitions = iter([True, False])
    deleted = []

    async def set_state(*_args, **_kwargs):
        return next(transitions)

    def encode(_source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"ID3")
        return 3, 10.0, "checksum"

    async def upload(_path, _key, *, content_type):
        assert content_type == "audio/mpeg"

    async def delete(key):
        deleted.append(key)

    monkeypatch.setenv("RECORDING_STORAGE_DIR", str(tmp_path / "objects"))
    monkeypatch.setattr(recordings, "_set_recording_state", set_state)
    monkeypatch.setattr(recordings, "_encode_pcm_to_mp3", encode)
    monkeypatch.setattr(recordings.storage_client, "upload_path", upload)
    monkeypatch.setattr(recordings.storage_client, "delete_file_strict", delete)

    assert await recordings.finalize_recording(call_id, 7, spool) is False
    assert deleted == [f"calls/7/{call_id}.mp3"]
    assert not spool.exists()


@pytest.mark.anyio
async def test_recording_access_explicitly_excludes_deleted_calls(monkeypatch):
    owned_call_arguments = []

    async def owned_call(_db, call_id, user_id, include_deleted=True):
        owned_call_arguments.append((call_id, user_id, include_deleted))
        return SimpleNamespace(id=call_id)

    class EmptyResult:
        def scalars(self):
            return self

        def first(self):
            return None

    class Session:
        async def execute(self, _statement):
            return EmptyResult()

    call_id = uuid.uuid4()
    monkeypatch.setattr(calls_api, "_owned_call", owned_call)
    with pytest.raises(Exception) as error:
        await calls_api.create_recording_access(
            call_id,
            SimpleNamespace(),
            current_user=SimpleNamespace(id=9),
            db=Session(),
        )

    assert getattr(error.value, "status_code", None) == 409
    assert owned_call_arguments == [(call_id, 9, False)]
