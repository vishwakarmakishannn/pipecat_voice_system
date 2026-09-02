"""Redaction policy for tool telemetry, persistence, and ordinary browser events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from core.issue_contract import ISSUE_FIELDS


_SECRET_KEY = re.compile(
    r"(?:authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)",
    re.IGNORECASE,
)


def _copy_without_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[redacted]"
                if _SECRET_KEY.search(str(key))
                else _copy_without_secrets(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_copy_without_secrets(item) for item in value]
    return value


def _redact_issue_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted = {}
        for key, item in value.items():
            name = str(key)
            contract = ISSUE_FIELDS.get(name)
            if contract is not None and item is not None:
                redacted[name] = contract.masked(str(item))
            elif name == "customer_name" and item is not None:
                # Backward-compatible protection for stale clients/models after
                # the field was removed from the demo contract.
                redacted[name] = "[provided]"
            else:
                redacted[name] = _redact_issue_mapping(item)
        return redacted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_redact_issue_mapping(item) for item in value]
    return value


def sanitize_tool_data(function_name: str, value: Any) -> Any:
    """Return a defensive, PII-safe copy for non-privileged destinations."""
    safe = _copy_without_secrets(value)
    if function_name == "manage_issue_draft":
        safe = _redact_issue_mapping(safe)
    return safe
