"""Trusted session date context and timezone configuration."""

import os
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


SESSION_DATE_CONTEXT_MARKER = "TRUSTED_SESSION_DATE_CONTEXT"
DEFAULT_VOICE_TIMEZONE = "Asia/Kolkata"


def configured_timezone_name() -> str:
    """Return the configured IANA timezone or raise a non-secret error."""
    name = os.getenv("VOICE_TIMEZONE", DEFAULT_VOICE_TIMEZONE).strip()
    if not name:
        raise ValueError("VOICE_TIMEZONE must not be empty")
    try:
        ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"VOICE_TIMEZONE must be a valid IANA timezone, got {name!r}"
        ) from exc
    return name


def session_date_context(
    *,
    timezone_name: str | None = None,
    now: datetime | None = None,
) -> str:
    """Build stable date-only metadata for one voice session."""
    name = timezone_name or configured_timezone_name()
    try:
        zone = ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Invalid IANA timezone: {name!r}") from exc
    local_now = datetime.now(zone) if now is None else now.astimezone(zone)
    return (
        f"{SESSION_DATE_CONTEXT_MARKER}: Trusted backend metadata.\n"
        f"Current date: {local_now.strftime('%B %d, %Y')} "
        f"({local_now.date().isoformat()})\n"
        f"Configured timezone: {name}\n"
        "No clock time is injected. When exact current time, a timezone "
        "conversion, or a deadline based on now is required, use the "
        "get_current_datetime tool. This metadata does not make model knowledge "
        "current; use tavily_search for current news or external facts."
    )
