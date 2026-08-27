"""Small helpers for useful diagnostics without logging user content."""

from __future__ import annotations

import hashlib
import re


def safe_text_metadata(value: object) -> str:
    """Return stable, non-reversible metadata for potentially sensitive text."""
    text = "" if value is None else str(value)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    word_count = len(re.findall(r"\S+", text))
    return f"chars={len(text)},words={word_count},sha256={digest}"
