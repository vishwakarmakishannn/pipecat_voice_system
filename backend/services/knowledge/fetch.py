"""Bounded, SSRF-resistant website acquisition for approved Mswipe domains."""

import asyncio
import http.client
import ipaddress
import socket
import ssl
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

from core.knowledge_config import (
    KNOWLEDGE_ALLOWED_DOMAINS,
    KNOWLEDGE_FETCH_TIMEOUT_SECONDS,
    KNOWLEDGE_MAX_RESPONSE_BYTES,
    KNOWLEDGE_RESPECT_ROBOTS,
    KNOWLEDGE_USER_AGENT,
)


_TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid"}


class SourceSkipped(ValueError):
    """The source is valid, but policy says it must not be ingested."""


class SourceHTTPError(ValueError):
    """A public source responded, but did not return an ingestible success."""

    def __init__(self, status: int):
        self.status = status
        super().__init__(f"Source returned HTTP {status}")


@dataclass(frozen=True)
class FetchResult:
    requested_url: str
    final_url: str
    status: int
    headers: dict[str, str]
    content: bytes
    charset: str


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only http and https URLs are supported")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed")
    hostname = parsed.hostname.lower().rstrip(".")
    if not any(hostname == domain or hostname.endswith(f".{domain}") for domain in KNOWLEDGE_ALLOWED_DOMAINS):
        raise ValueError(f"Domain {hostname!r} is outside the knowledge allow-list")
    port = parsed.port
    default_port = 443 if parsed.scheme.lower() == "https" else 80
    if port and port not in {80, 443}:
        raise ValueError("Only standard web ports 80 and 443 are allowed")
    netloc = hostname if not port or port == default_port else f"{hostname}:{port}"
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_PARAMS
        )
    )
    return urlunparse((parsed.scheme.lower(), netloc, path, "", query, ""))


def _is_public_ip(value: str) -> bool:
    address = ipaddress.ip_address(value)
    return not any(
        (
            address.is_private,
            address.is_loopback,
            address.is_link_local,
            address.is_multicast,
            address.is_reserved,
            address.is_unspecified,
        )
    )


def _resolve_public(url: str) -> tuple[object, int, list[str]]:
    canonical = canonicalize_url(url)
    parsed = urlparse(canonical)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError("Could not resolve source hostname") from exc
    addresses = sorted({info[4][0] for info in infos})
    if not addresses or any(not _is_public_ip(address) for address in addresses):
        raise ValueError("Private, local, or reserved network targets are not allowed")
    return parsed, port, addresses


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, hostname: str, address: str, port: int):
        super().__init__(hostname, port=port, timeout=KNOWLEDGE_FETCH_TIMEOUT_SECONDS)
        self._pinned_address = address

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._pinned_address, self.port), self.timeout, self.source_address
        )
        if self._tunnel_host:
            self._tunnel()
        self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host)


def _request(
    url: str,
    max_bytes: int,
    conditional_headers: dict[str, str] | None = None,
) -> tuple[int, object, bytes]:
    parsed, port, addresses = _resolve_public(url)
    path = parsed.path or "/"
    if parsed.query:
        path += f"?{parsed.query}"
    default_port = 443 if parsed.scheme == "https" else 80
    host = parsed.hostname if port == default_port else f"{parsed.hostname}:{port}"
    last_error: Exception | None = None
    for address in addresses:
        connection = (
            _PinnedHTTPSConnection(parsed.hostname, address, port)
            if parsed.scheme == "https"
            else http.client.HTTPConnection(
                address, port=port, timeout=KNOWLEDGE_FETCH_TIMEOUT_SECONDS
            )
        )
        try:
            headers = {
                "Host": host,
                "User-Agent": KNOWLEDGE_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml,text/xml,application/pdf,text/plain;q=0.8",
                "Connection": "close",
            }
            headers.update(conditional_headers or {})
            connection.request(
                "GET",
                path,
                headers=headers,
            )
            response = connection.getresponse()
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise ValueError("Source response exceeded the configured size limit")
            return response.status, response.headers, body
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            last_error = exc
        finally:
            connection.close()
    raise ValueError("Could not safely fetch source") from last_error


def _fetch_sync(
    url: str,
    max_redirects: int = 5,
    conditional_headers: dict[str, str] | None = None,
) -> FetchResult:
    requested = canonicalize_url(url)
    current = requested
    for redirect_number in range(max_redirects + 1):
        status, raw_headers, content = _request(
            current,
            KNOWLEDGE_MAX_RESPONSE_BYTES,
            conditional_headers,
        )
        headers = {key.lower(): value for key, value in raw_headers.items()}
        if status in {301, 302, 303, 307, 308}:
            if redirect_number >= max_redirects or not headers.get("location"):
                raise ValueError("Source redirected too many times or without a location")
            current = canonicalize_url(urljoin(current, headers["location"]))
            # A validator belongs to the originally resolved representation.
            # Do not send it to a different redirect target.
            conditional_headers = None
            continue
        if status == 304:
            return FetchResult(requested, current, status, headers, b"", "utf-8")
        if not 200 <= status < 300:
            raise SourceHTTPError(status)
        content_type = headers.get("content-type", "").lower()
        if not any(
            item in content_type
            for item in (
                "text/html",
                "application/xhtml",
                "application/xml",
                "text/xml",
                "text/plain",
                "application/pdf",
            )
        ):
            raise ValueError(f"Unsupported source content type: {content_type or 'missing'}")
        charset = raw_headers.get_content_charset() or "utf-8"
        return FetchResult(requested, current, status, headers, content, charset)
    raise ValueError("Source redirected too many times")


async def robots_allowed(url: str) -> bool:
    if not KNOWLEDGE_RESPECT_ROBOTS:
        return True
    parsed = urlparse(canonicalize_url(url))
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        result = await asyncio.to_thread(_fetch_sync, robots_url, 3)
    except Exception:
        # An absent/unavailable robots file is not an explicit prohibition.
        return True
    parser = RobotFileParser(robots_url)
    parser.parse(result.content.decode(result.charset, errors="replace").splitlines())
    return parser.can_fetch(KNOWLEDGE_USER_AGENT, url)


async def fetch_public_source(
    url: str,
    *,
    etag: str | None = None,
    last_modified: str | None = None,
) -> FetchResult:
    canonical = canonicalize_url(url)
    if not await robots_allowed(canonical):
        raise SourceSkipped("Source ingestion is disallowed by robots.txt")
    conditional_headers = {}
    if etag:
        conditional_headers["If-None-Match"] = etag
    if last_modified:
        conditional_headers["If-Modified-Since"] = last_modified
    return await asyncio.to_thread(
        _fetch_sync,
        canonical,
        5,
        conditional_headers or None,
    )
