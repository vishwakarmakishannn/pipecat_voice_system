import asyncio
import signal

from core.database import engine, voice_engine
from core.logging_config import configure_nonblocking_logging
from core.rag_worker import run_rag_worker


async def main() -> None:
    configure_nonblocking_logging()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signal_name, stop_event.set)
    try:
        await run_rag_worker(stop_event)
    finally:
        await engine.dispose()
        await voice_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
