import asyncio
import os
import time
from loguru import logger
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import AsyncAdaptedQueuePool
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    # Docker, load balancers, and database failovers can leave an idle socket
    # in the pool after the server side has already closed it. Validate every
    # checkout so a dead connection is replaced before a call transaction
    # starts.
    pool_pre_ping=True,
    pool_size=int(os.getenv("DB_POOL_SIZE", "20")),
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
    pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
    pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "1800")),
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

_voice_budget_ms = int(float(os.getenv("VOICE_RETRIEVAL_BUDGET_SECONDS", "1.0")) * 1000)
_voice_statement_timeout_ms = int(os.getenv("DB_VOICE_STATEMENT_TIMEOUT_MS", "750"))
_voice_hnsw_iterative_scan = os.getenv("DB_VOICE_HNSW_ITERATIVE_SCAN", "relaxed_order").strip()
if _voice_hnsw_iterative_scan not in {"off", "strict_order", "relaxed_order"}:
    raise ValueError(
        "DB_VOICE_HNSW_ITERATIVE_SCAN must be off, strict_order, or relaxed_order"
    )
if not 100 <= _voice_statement_timeout_ms < _voice_budget_ms:
    raise ValueError(
        "DB_VOICE_STATEMENT_TIMEOUT_MS must be at least 100 ms and below "
        f"the {_voice_budget_ms} ms voice retrieval budget"
    )

# Reserve database capacity for live turns so REST and ingestion cannot consume
# every connection needed by latency-sensitive retrieval and tool calls.
class InstrumentedVoicePool(AsyncAdaptedQueuePool):
    """Measure the hidden wait inside async pool checkout."""

    def _do_get(self):
        started = time.monotonic()
        try:
            return super()._do_get()
        finally:
            logger.info(
                "voice_db_pool checkout_wait_ms={} checked_out={} size={} overflow={}",
                round((time.monotonic() - started) * 1000, 1),
                self.checkedout(), self.size(), self.overflow(),
            )


voice_engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=int(os.getenv("DB_VOICE_POOL_SIZE", "10")),
    max_overflow=int(os.getenv("DB_VOICE_MAX_OVERFLOW", "0")),
    pool_timeout=float(os.getenv("DB_VOICE_POOL_TIMEOUT", "0.25")),
    pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "1800")),
    poolclass=InstrumentedVoicePool,
    connect_args={
        "server_settings": {
            "statement_timeout": str(_voice_statement_timeout_ms),
            # pgvector 0.8+ continues an approximate scan until enough rows
            # survive tenant/status filters instead of returning a sparse top-k.
            "hnsw.iterative_scan": _voice_hnsw_iterative_scan,
            "hnsw.max_scan_tuples": os.getenv("DB_VOICE_HNSW_MAX_SCAN_TUPLES", "10000"),
        }
    },
)
VoiceSessionLocal = async_sessionmaker(voice_engine, expire_on_commit=False)


async def warm_voice_pool(connection_count: int = 2) -> None:
    """Establish a small number of latency-critical connections at startup."""
    from sqlalchemy import text

    count = max(1, min(connection_count, voice_engine.pool.size()))

    async def warm_one() -> None:
        async with voice_engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    started = time.monotonic()
    await asyncio.gather(*(warm_one() for _ in range(count)))
    logger.info(
        "voice_db_pool status=warmed connections={} duration_ms={}",
        count,
        round((time.monotonic() - started) * 1000, 1),
    )


def voice_pool_snapshot() -> dict[str, int]:
    pool = voice_engine.pool
    return {
        "size": pool.size(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "checked_in": pool.checkedin(),
    }
Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
