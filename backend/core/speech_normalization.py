"""Provider-independent conversion from display text to safe spoken text."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import unicodedata


_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)")
_RAW_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>]+", re.IGNORECASE)
_HEADING = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")
_BULLET = re.compile(r"(?m)^\s*(?:[-*+]\s+|[•◦▪‣]\s*)")
_NUMBERED_ITEM = re.compile(r"(?m)^\s*\d{1,3}[.)]\s+")
_CODE_FENCE = re.compile(r"```(?:[A-Za-z0-9_+-]+)?")
_MARKDOWN_MARKER = re.compile(r"(?:\*\*|__|~~|`)")
_SPACE = re.compile(r"[\t\v\f\r ]+")
_TOO_MANY_BREAKS = re.compile(r"\n{2,}")
_PUNCTUATION_SPACE = re.compile(r"\s+([,.;:!?])")


def speech_max_characters() -> int:
    raw = os.getenv("VOICE_SPEECH_MAX_CHARACTERS", "1200")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("VOICE_SPEECH_MAX_CHARACTERS must be an integer") from exc
    if not 160 <= value <= 5000:
        raise ValueError(
            "VOICE_SPEECH_MAX_CHARACTERS must be between 160 and 5000"
        )
    return value


def speech_stream_buffer_characters() -> int:
    raw = os.getenv("VOICE_SPEECH_STREAM_BUFFER_CHARACTERS", "240")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "VOICE_SPEECH_STREAM_BUFFER_CHARACTERS must be an integer"
        ) from exc
    if not 80 <= value <= 600:
        raise ValueError(
            "VOICE_SPEECH_STREAM_BUFFER_CHARACTERS must be between 80 and 600"
        )
    return value


def _lexicon_path() -> Path:
    configured = os.getenv("VOICE_PRONUNCIATION_LEXICON_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parent.parent / "config" / "pronunciations.json"


def load_pronunciation_lexicon() -> tuple[str, tuple[tuple[str, str], ...]]:
    path = _lexicon_path()
    if not path.exists():
        return "none", ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = str(payload.get("version") or "unversioned")
    entries = []
    for item in payload.get("entries", []):
        written = " ".join(str(item.get("written") or "").split())
        spoken = " ".join(str(item.get("spoken") or "").split())
        if not written or not spoken:
            raise ValueError("Pronunciation entries require written and spoken values")
        entries.append((written, spoken))
    return version, tuple(entries)


def normalize_speech_text(
    text: str,
    *,
    lexicon: tuple[tuple[str, str], ...] = (),
) -> str:
    """Normalize one complete, non-private assistant text segment for speech."""
    value = unicodedata.normalize("NFKC", text or "")
    value = value.replace("\u00a0", " ").replace("\u200b", "")
    value = value.translate(
        str.maketrans({
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "—": ", ",
            "–": ", ",
            "…": "...",
        })
    )
    value = _MARKDOWN_LINK.sub(r"\1", value)
    value = _RAW_URL.sub("the provided link", value)
    value = _CODE_FENCE.sub("", value)
    value = _HEADING.sub("", value)
    value = _BULLET.sub("", value)
    value = _NUMBERED_ITEM.sub("", value)
    value = _MARKDOWN_MARKER.sub("", value)
    value = value.replace("&", " and ")
    value = value.replace("%", " percent")
    value = _TOO_MANY_BREAKS.sub("\n", value)
    value = re.sub(r"\s*\n\s*", ". ", value)
    for written, spoken in lexicon:
        value = re.sub(
            rf"(?<!\w){re.escape(written)}(?!\w)",
            spoken,
            value,
            flags=re.IGNORECASE,
        )
    value = _SPACE.sub(" ", value)
    value = _PUNCTUATION_SPACE.sub(r"\1", value)
    value = re.sub(r"(?:\.\s*){2,}", ". ", value)
    return value.strip()


def character_category_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for character in text or "":
        category = unicodedata.category(character)
        counts[category] = counts.get(category, 0) + 1
    return counts

