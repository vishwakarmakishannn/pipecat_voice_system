"""Authenticated, short-lived browser transport configuration."""

import base64
import hashlib
import hmac
import os
import time

from fastapi import APIRouter, Depends, HTTPException

from api.auth import get_current_user
from core.models import User


router = APIRouter(prefix="/api/transport", tags=["transport"])


def _urls(name: str) -> list[str]:
    return [value.strip() for value in os.getenv(name, "").split(",") if value.strip()]


def build_ice_servers(user_id: int, now: int | None = None) -> list[dict]:
    servers: list[dict] = []
    stun_urls = _urls("STUN_URLS") or ["stun:stun.l.google.com:19302"]
    if stun_urls:
        servers.append({"urls": stun_urls})

    turn_urls = _urls("TURN_URLS")
    if not turn_urls:
        return servers
    secret = os.getenv("TURN_SHARED_SECRET", "").strip()
    if not secret:
        raise ValueError("TURN_SHARED_SECRET is required when TURN_URLS is configured")
    ttl = int(os.getenv("TURN_CREDENTIAL_TTL_SECONDS", "600"))
    if ttl < 60 or ttl > 3600:
        raise ValueError("TURN_CREDENTIAL_TTL_SECONDS must be between 60 and 3600")
    expires = (int(time.time()) if now is None else now) + ttl
    username = f"{expires}:{user_id}"
    credential = base64.b64encode(
        hmac.new(secret.encode(), username.encode(), hashlib.sha1).digest()
    ).decode()
    servers.append(
        {
            "urls": turn_urls,
            "username": username,
            "credential": credential,
            "credentialType": "password",
        }
    )
    return servers


@router.get("/ice-servers")
async def ice_servers(current_user: User = Depends(get_current_user)):
    try:
        return {"ice_servers": build_ice_servers(current_user.id)}
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
