import asyncio
from email.message import Message
import socket

import pytest

from services.knowledge import fetch
from services.knowledge.fetch import (
    SourceHTTPError,
    SourceSkipped,
    _fetch_sync,
    _resolve_public,
    canonicalize_url,
)


def test_canonical_url_removes_fragment_tracking_and_sorts_query():
    assert canonicalize_url(
        "https://WWW.MSWIPE.COM/products/?utm_source=ad&b=2&a=1#details"
    ) == "https://www.mswipe.com/products?a=1&b=2"


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost/admin",
        "https://example.com/",
        "https://user:password@mswipe.com/",
        "https://www.mswipe.com:8443/admin",
    ],
)
def test_canonical_url_rejects_unapproved_or_unsafe_targets(url):
    with pytest.raises(ValueError):
        canonicalize_url(url)


def test_dns_resolution_rejects_private_addresses(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    )
    with pytest.raises(ValueError, match="Private"):
        _resolve_public("https://www.mswipe.com/")


def test_dns_resolution_returns_only_validated_public_addresses(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))],
    )
    parsed, port, addresses = _resolve_public("https://www.mswipe.com/")
    assert parsed.hostname == "www.mswipe.com"
    assert port == 443
    assert addresses == ["8.8.8.8"]


def test_fetch_reports_robots_exclusion_as_policy_skip(monkeypatch):
    async def disallowed(_url):
        return False

    monkeypatch.setattr(fetch, "robots_allowed", disallowed)
    with pytest.raises(SourceSkipped, match="robots.txt"):
        asyncio.run(fetch.fetch_public_source("https://www.mswipe.com/sign-in"))


def test_conditional_refresh_forwards_only_validators(monkeypatch):
    seen = []

    async def allowed(_url):
        return True

    def request(url, max_bytes, conditional_headers):
        seen.append((url, max_bytes, conditional_headers))
        headers = Message()
        headers["ETag"] = '"v1"'
        return 304, headers, b""

    monkeypatch.setattr(fetch, "robots_allowed", allowed)
    monkeypatch.setattr(fetch, "_request", request)
    result = asyncio.run(
        fetch.fetch_public_source(
            "https://www.mswipe.com/support",
            etag='"v1"',
            last_modified="Wed, 02 Sep 2026 10:00:00 GMT",
        )
    )

    assert result.status == 304
    assert seen[0][2] == {
        "If-None-Match": '"v1"',
        "If-Modified-Since": "Wed, 02 Sep 2026 10:00:00 GMT",
    }


def test_http_failure_has_typed_status_for_crawl_classification(monkeypatch):
    headers = Message()
    headers["Content-Type"] = "text/html"
    monkeypatch.setattr(fetch, "_request", lambda *_args: (404, headers, b"missing"))

    with pytest.raises(SourceHTTPError) as raised:
        _fetch_sync("https://www.mswipe.com/stale")

    assert raised.value.status == 404
