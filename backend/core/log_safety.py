"""Small helpers for useful diagnostics without logging user content."""

from __future__ import annotations

import hashlib
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit


_SENSITIVE_KEY = re.compile(
    r"(?:authorization|cookie|password|secret|api[_-]?key|access[_-]?token|refresh[_-]?token)",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_LONG_DIGITS = re.compile(r"(?<!\d)\d{7,16}(?!\d)")


def safe_text_metadata(value: object) -> str:
    """Return stable, non-reversible metadata for potentially sensitive text."""
    text = "" if value is None else str(value)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    word_count = len(re.findall(r"\S+", text))
    return f"chars={len(text)},words={word_count},sha256={digest}"


def redact_url(value: str) -> str:
    """Retain a useful route while dropping credentials, query values, and fragments."""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "[invalid-url]"
    if not parsed.scheme or not parsed.netloc:
        return "[redacted-url]"
    host = parsed.hostname or "redacted-host"
    port = f":{parsed.port}" if parsed.port else ""
    return urlunsplit((parsed.scheme, f"{host}{port}", parsed.path, "", ""))


def redact_log_value(value: Any, *, key: str | None = None) -> Any:
    """Recursively produce a log-safe diagnostic value without mutating input."""
    if key and _SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {
            str(item_key): redact_log_value(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_log_value(item) for item in value]
    if not isinstance(value, str):
        return value
    if key and (key.lower().endswith("url") or key.lower().endswith("uri")):
        return redact_url(value)
    redacted = _BEARER.sub("Bearer [REDACTED]", value)
    redacted = _EMAIL.sub("[REDACTED_EMAIL]", redacted)
    redacted = _LONG_DIGITS.sub("[REDACTED_NUMBER]", redacted)
    return redacted
