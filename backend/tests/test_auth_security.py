from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import api.auth as auth


def _request(host: str = "203.0.113.10") -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/auth/login",
            "headers": [],
            "client": (host, 50000),
            "server": ("test", 80),
            "scheme": "http",
            "query_string": b"",
        }
    )


@pytest.mark.anyio
async def test_bcrypt_helpers_run_in_worker_thread(monkeypatch):
    calls = []

    async def fake_to_thread(function, *args):
        calls.append((function, args))
        return "hashed" if function is auth.get_password_hash else True

    monkeypatch.setattr(auth.asyncio, "to_thread", fake_to_thread)

    assert await auth.get_password_hash_async("Secret123") == "hashed"
    assert await auth.verify_password_async("Secret123", "hash") is True
    assert calls == [
        (auth.get_password_hash, ("Secret123",)),
        (auth.verify_password, ("Secret123", "hash")),
    ]


@pytest.mark.anyio
async def test_auth_rate_limiter_returns_429_after_bounded_attempts():
    limiter = auth.AuthRateLimiter(attempts=2, window_seconds=60)
    request = _request()

    await limiter.check("login", request, "alice")
    await limiter.check("login", request, "alice")
    with pytest.raises(HTTPException) as captured:
        await limiter.check("login", request, "alice")

    assert captured.value.status_code == 429
    assert int(captured.value.headers["Retry-After"]) >= 1


@pytest.mark.anyio
async def test_login_uses_async_password_verification(monkeypatch):
    class Result:
        def scalars(self):
            return self

        def first(self):
            return SimpleNamespace(username="alice", password_hash="hash")

    class Session:
        async def execute(self, _statement):
            return Result()

    async def allow(*_args):
        return None

    verified = []

    async def verify(password, password_hash):
        verified.append((password, password_hash))
        return True

    monkeypatch.setattr(auth.auth_rate_limiter, "check", allow)
    monkeypatch.setattr(auth, "verify_password_async", verify)

    token = await auth.login(
        auth.UserLogin(username="alice", password="Secret123"),
        _request(),
        Session(),
    )

    assert token["token_type"] == "bearer"
    assert verified == [("Secret123", "hash")]


def test_registration_rejects_passwords_over_bcrypt_byte_limit():
    with pytest.raises(ValueError, match="72 UTF-8 bytes"):
        auth.UserCreate(username="valid_user", password="Password1" + "x" * 64)
    with pytest.raises(ValueError, match="72 UTF-8 bytes"):
        auth.UserCreate(username="valid_user", password="Password1" + "🙂" * 20)


@pytest.mark.anyio
async def test_overlong_login_password_is_a_safe_authentication_failure():
    login = auth.UserLogin(
        username="valid_user", password="Password1" + "x" * 100
    )
    hashed = await auth.get_password_hash_async("Password1")

    assert await auth.verify_password_async(login.password, hashed) is False
