import base64
import hashlib
import hmac

import pytest

from api.transport import build_ice_servers


def test_turn_credentials_are_short_lived_and_user_scoped(monkeypatch):
    monkeypatch.setenv("STUN_URLS", "stun:one.example")
    monkeypatch.setenv("TURN_URLS", "turn:one.example,turns:two.example")
    monkeypatch.setenv("TURN_SHARED_SECRET", "shared-secret")
    monkeypatch.setenv("TURN_CREDENTIAL_TTL_SECONDS", "600")

    servers = build_ice_servers(42, now=1_000)

    assert servers[0] == {"urls": ["stun:one.example"]}
    turn = servers[1]
    assert turn["username"] == "1600:42"
    expected = base64.b64encode(
        hmac.new(b"shared-secret", b"1600:42", hashlib.sha1).digest()
    ).decode()
    assert turn["credential"] == expected
    assert turn["urls"] == ["turn:one.example", "turns:two.example"]


def test_turn_urls_require_a_shared_secret(monkeypatch):
    monkeypatch.setenv("TURN_URLS", "turn:one.example")
    monkeypatch.delenv("TURN_SHARED_SECRET", raising=False)

    with pytest.raises(ValueError, match="TURN_SHARED_SECRET"):
        build_ice_servers(42, now=1_000)
