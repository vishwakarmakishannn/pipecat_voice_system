"""Guarded development reset required before the Voice System 2.0 migration."""

import argparse
import asyncio
import os
from pathlib import Path
from urllib.parse import urlsplit

from sqlalchemy import inspect, text

from core.database import engine
from core.storage import storage_client


CONFIRMATION = "RESET_ALL_APPLICATION_DATA"
TABLE_ALLOWLIST = {
    "call_events",
    "call_operations",
    "call_recordings",
    "call_turns",
    "calls",
    "conversations",
    "issues",
    "memory_chunks",
    "messages",
    "transcript_entries",
    "user_memories",
    "users",
}


def safe_database_target() -> str:
    parsed = urlsplit(os.environ.get("DATABASE_URL", ""))
    return (
        f"{parsed.scheme or 'unknown'}://{parsed.username or 'unset'}@"
        f"{parsed.hostname or 'unset'}:{parsed.port or 'default'}"
        f"/{(parsed.path or '/').lstrip('/')}"
    )


def _safe_storage_roots() -> list[Path]:
    configured = (
        os.getenv("PERSISTENCE_SPOOL_DIR", "data/persistence-spool"),
        os.getenv("RECORDING_SPOOL_DIR", os.getenv("VOICE_RECORDING_SPOOL_DIR", "recording-spool")),
        os.getenv("RECORDING_STORAGE_DIR", os.getenv("VOICE_RECORDING_STORAGE_DIR", "recordings")),
    )
    roots: list[Path] = []
    project = Path.cwd().resolve()
    for raw in configured:
        path = Path(raw).expanduser().resolve()
        if path in {Path("/").resolve(), Path.home().resolve(), project}:
            raise RuntimeError(f"Refusing unsafe storage reset target: {path}")
        roots.append(path)
    return roots


async def reset_database(*, dry_run: bool) -> list[str]:
    async with engine.begin() as connection:
        existing = set(await connection.run_sync(lambda sync: inspect(sync).get_table_names()))
        targets = sorted(existing & TABLE_ALLOWLIST)
        if targets and not dry_run:
            quoted = ", ".join(f'"{name}"' for name in targets)
            await connection.execute(text(f"TRUNCATE TABLE {quoted} RESTART IDENTITY CASCADE"))
        return targets


async def collect_s3_objects() -> list[str]:
    if not storage_client.use_s3:
        return []
    async with engine.connect() as connection:
        existing = set(await connection.run_sync(lambda sync: inspect(sync).get_table_names()))
        objects: set[str] = set()
        if "call_recordings" in existing:
            rows = await connection.execute(
                text("SELECT object_key FROM call_recordings WHERE object_key IS NOT NULL")
            )
            objects.update(str(value) for value in rows.scalars() if value)
        return sorted(objects)


async def clear_s3_objects(objects: list[str], *, dry_run: bool) -> None:
    print(f"s3_objects={len(objects)}")
    if dry_run:
        return
    for object_key in objects:
        await storage_client.delete_file_strict(object_key)


def clear_storage(roots: list[Path], *, dry_run: bool) -> None:
    for root in roots:
        print(f"storage_target={root}")
        if dry_run or not root.exists():
            continue
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"database_target={safe_database_target()} credentials=REDACTED")
    if not args.dry_run and args.confirm != CONFIRMATION:
        raise SystemExit(f"Refusing reset. Pass --confirm {CONFIRMATION}")

    roots = _safe_storage_roots()
    s3_objects = await collect_s3_objects()
    await clear_s3_objects(s3_objects, dry_run=args.dry_run)
    tables = await reset_database(dry_run=args.dry_run)
    print(f"database_tables={','.join(tables) or 'none'}")
    clear_storage(roots, dry_run=args.dry_run)
    print("status=dry_run" if args.dry_run else "status=reset_complete")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
