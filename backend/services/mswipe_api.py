"""Typed contract boundary for future Mswipe customer and ticket APIs.

No endpoint, credential header, or request shape is invented here. Production
adapters should implement this protocol only after Mswipe supplies approved
contracts and UAT access.
"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class CustomerIdentity:
    customer_id: str | None = None
    mobile: str | None = None
    device_id: str | None = None


@dataclass(frozen=True)
class LiveLookupRequest:
    operation: str
    identity: CustomerIdentity
    parameters: dict = field(default_factory=dict)
    idempotency_key: str | None = None


@dataclass(frozen=True)
class LiveLookupResult:
    status: str
    data: dict
    reference_id: str | None = None


@dataclass(frozen=True)
class TicketCreateRequest:
    identity: CustomerIdentity
    ticket_code: str
    ticket_subcode: str
    remark: str
    description: str
    confirmed: bool
    idempotency_key: str


@dataclass(frozen=True)
class TicketCreateResult:
    status: str
    ticket_id: str


class MswipeApi(Protocol):
    async def verify_customer(self, identity: CustomerIdentity) -> bool: ...

    async def lookup(self, request: LiveLookupRequest) -> LiveLookupResult: ...

    async def create_ticket(self, request: TicketCreateRequest) -> TicketCreateResult: ...


class UnconfiguredMswipeApi:
    async def verify_customer(self, identity: CustomerIdentity) -> bool:
        raise RuntimeError("Mswipe customer API contract is not configured")

    async def lookup(self, request: LiveLookupRequest) -> LiveLookupResult:
        raise RuntimeError("Mswipe live lookup API contract is not configured")

    async def create_ticket(self, request: TicketCreateRequest) -> TicketCreateResult:
        if not request.confirmed:
            raise ValueError("Ticket creation requires explicit caller confirmation")
        raise RuntimeError("Mswipe ticket API contract is not configured")


mswipe_api: MswipeApi = UnconfiguredMswipeApi()
