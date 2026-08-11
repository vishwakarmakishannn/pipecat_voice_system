import os
from urllib.parse import urlsplit


DEFAULT_ALLOWED_ORIGINS = "http://localhost:5173,http://localhost:80"


def parse_allowed_origins(raw: str | None) -> list[str]:
    """Return a validated, explicit CORS origin allow-list."""
    origins: list[str] = []
    for value in (raw or DEFAULT_ALLOWED_ORIGINS).split(","):
        origin = value.strip().rstrip("/")
        if not origin:
            continue
        if origin == "*":
            raise ValueError("Wildcard CORS origins are not allowed with credentials")
        parsed = urlsplit(origin)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid CORS origin: {origin!r}")
        if parsed.path or parsed.query or parsed.fragment:
            raise ValueError(f"CORS origins must not include a path: {origin!r}")
        if origin not in origins:
            origins.append(origin)
    if not origins:
        raise ValueError("At least one CORS origin must be configured")
    return origins


def configure_pipecat_allowed_origins() -> list[str]:
    """Make the application allow-list authoritative for Pipecat's middleware."""
    raw = os.getenv("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
    origins = parse_allowed_origins(raw)
    os.environ["PIPECAT_ALLOWED_ORIGINS"] = ",".join(origins)
    return origins
