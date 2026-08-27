"""Dedicated restart-recovery, abandonment, and retention worker."""

import asyncio

from core.logging_config import configure_nonblocking_logging
from services.call_maintenance import call_maintenance_loop
from services.calls import abandon_stale_calls
from services.recordings import recover_unfinished_recordings


async def main() -> None:
    configure_nonblocking_logging()
    await abandon_stale_calls(30)
    await recover_unfinished_recordings()
    stop = asyncio.Event()
    try:
        await call_maintenance_loop(stop)
    finally:
        stop.set()


if __name__ == "__main__":
    asyncio.run(main())
