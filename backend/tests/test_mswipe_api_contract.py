import asyncio

import pytest

from services.mswipe_api import (
    CustomerIdentity,
    TicketCreateRequest,
    UnconfiguredMswipeApi,
)


def test_ticket_contract_requires_explicit_confirmation_before_adapter_call():
    async def exercise():
        adapter = UnconfiguredMswipeApi()
        request = TicketCreateRequest(
            identity=CustomerIdentity(customer_id="C123456"),
            ticket_code="Device",
            ticket_subcode="Activation",
            remark="Activation pending",
            description="Terminal is not active",
            confirmed=False,
            idempotency_key="call-turn-1",
        )
        with pytest.raises(ValueError, match="confirmation"):
            await adapter.create_ticket(request)

    asyncio.run(exercise())


def test_unconfigured_adapter_never_fakes_a_ticket_endpoint():
    async def exercise():
        adapter = UnconfiguredMswipeApi()
        request = TicketCreateRequest(
            identity=CustomerIdentity(customer_id="C123456"),
            ticket_code="Device",
            ticket_subcode="Activation",
            remark="Activation pending",
            description="Terminal is not active",
            confirmed=True,
            idempotency_key="call-turn-1",
        )
        with pytest.raises(RuntimeError, match="not configured"):
            await adapter.create_ticket(request)

    asyncio.run(exercise())
