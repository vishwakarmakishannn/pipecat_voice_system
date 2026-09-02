"""Versioned, demo-only complaint field contract.

The production Mswipe API contract has not been supplied.  This module keeps
the temporary demo shapes in one typed boundary so they cannot silently become
business truth or drift between validation, prompts, masking, and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Literal


FieldKind = Literal["identifier", "email", "phone", "text"]
PrivacyClass = Literal["identity", "contact", "restricted"]
ValidationCode = Literal[
    "missing",
    "incomplete_speech",
    "invalid_format",
    "unverified",
]


@dataclass(frozen=True)
class IssueFieldContract:
    name: str
    label: str
    kind: FieldKind
    privacy: PrivacyClass
    required: bool = True
    pattern: str | None = None
    prefix: str = ""
    digit_count: int | None = None
    allowed_first_digits: tuple[str, ...] = ()
    minimum_length: int | None = None

    def canonicalize(self, value: str) -> str:
        compact = " ".join((value or "").strip().split())
        if self.kind in {"identifier", "phone"}:
            digits = "".join(character for character in compact if character.isdigit())
            return f"{self.prefix}{digits}" if self.prefix else digits
        if self.kind == "email":
            return re.sub(r"\s+", "", compact).casefold()
        return compact

    def validate(self, value: str | None) -> tuple[ValidationCode, str | None]:
        if not value:
            return "missing", None
        canonical = self.canonicalize(value)
        if self.digit_count is not None:
            digits = canonical[len(self.prefix) :] if self.prefix else canonical
            if len(digits) < self.digit_count:
                return "incomplete_speech", canonical
            if len(digits) != self.digit_count:
                return "invalid_format", canonical
            if self.allowed_first_digits and digits[:1] not in self.allowed_first_digits:
                return "invalid_format", canonical
        if self.minimum_length is not None and len(canonical) < self.minimum_length:
            return "incomplete_speech", canonical
        if self.pattern and not re.fullmatch(self.pattern, canonical):
            return "invalid_format", canonical
        # The demo has no customer/device verification API.  Format-valid
        # values are usable for the demo draft but must remain explicitly
        # unverified rather than being described as verified.
        return "unverified", canonical

    def correction(self, code: ValidationCode) -> str:
        if code == "missing":
            return f"Please provide the {self.label}."
        if self.digit_count is not None:
            prefix = f" after {self.prefix}" if self.prefix else ""
            guidance = f"exactly {self.digit_count} digits{prefix}"
            if self.allowed_first_digits:
                starts = ", ".join(self.allowed_first_digits[:-1])
                if starts:
                    starts += f", or {self.allowed_first_digits[-1]}"
                else:
                    starts = self.allowed_first_digits[0]
                guidance += f", beginning with {starts}"
            if code == "incomplete_speech":
                return f"I heard only part of the {self.label}. Please repeat {guidance}."
            return f"The {self.label} must contain {guidance}. Please repeat it."
        if self.kind == "email":
            return (
                "Please repeat the complete email address, including the part "
                "before at and the domain after it."
            )
        if self.minimum_length is not None:
            return f"Please give a little more detail for the {self.label}."
        return f"Please repeat the {self.label}."

    def masked(self, value: str | None) -> str | None:
        if not value:
            return None
        canonical = self.canonicalize(value)
        if self.kind == "email":
            local, separator, domain = canonical.partition("@")
            if not separator:
                return "***"
            visible = local[:1] if local else ""
            return f"{visible}***@{domain}"
        if self.kind in {"identifier", "phone"}:
            suffix = canonical[-4:] if len(canonical) >= 4 else canonical[-1:]
            return f"***{suffix}"
        return "[provided]"


ISSUE_CONTRACT_VERSION = os.getenv(
    "MSWIPE_ISSUE_CONTRACT_VERSION", "demo-v1-unverified"
).strip()
if ISSUE_CONTRACT_VERSION != "demo-v1-unverified":
    raise ValueError(
        "Only the demo-v1-unverified issue contract is implemented; an approved "
        "Mswipe API contract is required before selecting another version"
    )


ISSUE_FIELDS: dict[str, IssueFieldContract] = {
    "cust_id": IssueFieldContract(
        name="cust_id",
        label="customer ID",
        kind="identifier",
        privacy="identity",
        prefix="C",
        digit_count=6,
        pattern=r"C\d{6}",
    ),
    "email": IssueFieldContract(
        name="email",
        label="email address",
        kind="email",
        privacy="contact",
        pattern=r"[^\s@]+@[^\s@]+\.[^\s@]+",
    ),
    "mobile": IssueFieldContract(
        name="mobile",
        label="mobile number",
        kind="phone",
        privacy="contact",
        digit_count=10,
        allowed_first_digits=("6", "7", "8", "9"),
        pattern=r"[6-9]\d{9}",
    ),
    "device_id": IssueFieldContract(
        name="device_id",
        label="device ID",
        kind="identifier",
        privacy="identity",
        prefix="MSW",
        digit_count=8,
        pattern=r"MSW\d{8}",
    ),
    "description": IssueFieldContract(
        name="description",
        label="issue description",
        kind="text",
        privacy="restricted",
        minimum_length=8,
    ),
}

ISSUE_REQUIRED_FIELDS = tuple(
    name for name, contract in ISSUE_FIELDS.items() if contract.required
)


def safe_issue_field_states(
    values: dict[str, str | None],
    states: dict[str, ValidationCode],
) -> dict[str, dict[str, str | bool | None]]:
    """Return the only complaint-field representation allowed in normal UI events."""
    return {
        name: {
            "present": bool(values.get(name)),
            "state": states.get(name, "missing"),
            "masked": contract.masked(values.get(name)),
        }
        for name, contract in ISSUE_FIELDS.items()
    }

