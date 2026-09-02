"""Operator CLI for the Mswipe knowledge control plane.

Run ``python knowledge_cli.py --help`` from the backend directory.
"""

import argparse
import asyncio
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urlparse
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.future import select

from core.database import AsyncSessionLocal, engine, voice_engine
from core.models import (
    KnowledgeDocument,
    KnowledgeEmbedding,
    KnowledgeJob,
    KnowledgeRelease,
    KnowledgeReleaseUnit,
    KnowledgeSnapshot,
    KnowledgeSource,
    KnowledgeUnit,
)
from services.knowledge.fetch import canonicalize_url
from services.knowledge.aliases import seed_default_aliases
from services.knowledge.jobs import enqueue_knowledge_job
from services.knowledge.releases import (
    create_release,
    publish_release,
    rollback_release,
    validate_release,
)
from services.knowledge.retrieval import retrieve_knowledge
from services.knowledge.taxonomy import import_ticket_taxonomy
from services.knowledge.units import approve_unit, retire_unit


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Mswipe knowledge control plane")
    commands = root.add_subparsers(dest="command", required=True)

    add_source = commands.add_parser("source-add", help="register an approved website source")
    add_source.add_argument("--name", required=True)
    add_source.add_argument("--url", required=True)
    add_source.add_argument("--authority", type=int, default=3, choices=range(1, 6))
    add_source.add_argument("--max-pages", type=int, default=500)
    add_source.add_argument("--max-depth", type=int, default=6)

    crawl = commands.add_parser("crawl", help="enqueue a website crawl")
    crawl.add_argument("source_id", type=UUID)

    embed = commands.add_parser("embed", help="enqueue knowledge embeddings")
    embed.add_argument(
        "--release-id",
        type=UUID,
        help="embed only the exact units in one release (recommended)",
    )
    commands.add_parser("conflicts-detect", help="enqueue contradiction detection")
    commands.add_parser("aliases-seed", help="seed conservative Mswipe/STT aliases")

    taxonomy = commands.add_parser("taxonomy-import", help="synchronize active mDesk taxonomy")
    taxonomy.add_argument("csv_path")

    units = commands.add_parser("units", help="list reviewable units")
    units.add_argument("--status", default="draft", choices=("draft", "approved", "retired"))
    units.add_argument("--limit", type=int, default=100)
    units.add_argument("--offset", type=int, default=0)
    units.add_argument("--type", dest="unit_type")
    units.add_argument("--source-contains")

    show = commands.add_parser("unit-show", help="show the complete review record")
    show.add_argument("unit_id", type=UUID)

    approve = commands.add_parser("unit-approve", help="approve one reviewed unit")
    approve.add_argument("unit_id", type=UUID)
    approve.add_argument("--voice-answer")
    approve.add_argument("--atomic-answer", action="store_true")
    approve.add_argument("--answerability-reviewed", action="store_true")
    approve.add_argument("--notes")

    retire = commands.add_parser("unit-retire", help="reject or retire one unit")
    retire.add_argument("unit_id", type=UUID)
    retire.add_argument("--notes")

    create = commands.add_parser("release-create", help="create an immutable draft release")
    create.add_argument("version")
    create.add_argument("unit_ids", nargs="*", type=UUID)
    create.add_argument("--all-approved", action="store_true")
    create.add_argument("--description")

    validate = commands.add_parser("release-validate")
    validate.add_argument("release_id", type=UUID)
    publish = commands.add_parser("release-publish")
    publish.add_argument("release_id", type=UUID)
    rollback = commands.add_parser("release-rollback")
    rollback.add_argument("release_id", type=UUID)
    commands.add_parser("status", help="show release state")
    commands.add_parser("jobs", help="show the most recent control-plane jobs")
    commands.add_parser("corpus-audit", help="summarize corpus quality and review state")

    demo = commands.add_parser(
        "demo-prepare",
        help="create a clearly labelled, non-production public-site demo release",
    )
    demo.add_argument("version")
    demo.add_argument("--source-id", required=True, type=UUID)

    search = commands.add_parser("search", help="test the currently published release")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=4)
    return root


def _answer_fingerprint(answer: str) -> str:
    normalized = re.sub(r"\s+", " ", answer).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


_DEMO_PATH_PREFIXES = (
    "/contact-us",
    "/in-store-solutions",
    "/online-solutions",
    "/support",
    "/tracking-solutions",
)


def _is_demo_source_uri(uri: str) -> bool:
    parsed = urlparse(uri)
    path = parsed.path.rstrip("/") or "/"
    return parsed.hostname in {"mswipe.com", "www.mswipe.com"} and any(
        path == prefix or path.startswith(f"{prefix}/")
        for prefix in _DEMO_PATH_PREFIXES
    )


async def _corpus_audit(db) -> dict:
    source_rows = (await db.execute(select(KnowledgeSource))).scalars().all()
    snapshot_rows = (await db.execute(select(KnowledgeSnapshot))).scalars().all()
    document_rows = (await db.execute(select(KnowledgeDocument))).scalars().all()
    unit_rows = (await db.execute(select(KnowledgeUnit))).scalars().all()
    release_rows = (await db.execute(select(KnowledgeRelease))).scalars().all()
    embedding_rows = await db.execute(
        select(
            KnowledgeEmbedding.provider,
            KnowledgeEmbedding.model,
            func.count(KnowledgeEmbedding.id),
        ).group_by(KnowledgeEmbedding.provider, KnowledgeEmbedding.model)
    )

    active_units = [unit for unit in unit_rows if unit.status in {"draft", "approved"}]
    answer_counts = Counter(_answer_fingerprint(unit.answer) for unit in active_units)
    source_counts = Counter(unit.source_uri for unit in active_units)
    quality_values = [
        float(document.quality_score)
        for document in document_rows
        if document.quality_score is not None
    ]
    return {
        "sources": [
            {
                "id": str(source.id),
                "name": source.name,
                "url": source.canonical_uri,
                "enabled": source.enabled,
            }
            for source in source_rows
        ],
        "snapshots_by_status": dict(Counter(item.status for item in snapshot_rows)),
        "documents": {
            "total": len(document_rows),
            "with_warnings": sum(bool(item.warnings) for item in document_rows),
            "average_quality": (
                round(sum(quality_values) / len(quality_values), 4)
                if quality_values
                else None
            ),
            "minimum_quality": round(min(quality_values), 4) if quality_values else None,
        },
        "units_by_status": dict(Counter(item.status for item in unit_rows)),
        "active_units_by_type": dict(Counter(item.unit_type for item in active_units)),
        "active_exact_duplicate_units": sum(
            count - 1 for count in answer_counts.values() if count > 1
        ),
        "top_active_source_pages": [
            {"url": url, "units": count}
            for url, count in source_counts.most_common(15)
        ],
        "embeddings": [
            {"provider": provider, "model": model, "count": count}
            for provider, model, count in embedding_rows.all()
        ],
        "releases_by_status": dict(Counter(item.status for item in release_rows)),
        # This only answers whether manual unit review is complete. Embedding,
        # conflict detection, evaluation, and release validation are separate
        # publication gates.
        "review_complete": bool(active_units)
        and all(unit.status == "approved" for unit in active_units)
        and not any(count > 1 for count in answer_counts.values()),
    }


async def run(args: argparse.Namespace) -> None:
    async with AsyncSessionLocal() as db:
        if args.command == "source-add":
            source = KnowledgeSource(
                name=args.name,
                canonical_uri=canonicalize_url(args.url),
                source_type="website",
                authority=args.authority,
                crawl_policy={
                    "max_pages": args.max_pages,
                    "max_depth": args.max_depth,
                    "use_sitemap": True,
                    "exclude_unit_path_prefixes": [
                        "/about-us",
                        "/blog",
                        "/careers",
                        "/knowledge",
                        "/mswipe-career",
                    ],
                },
            )
            db.add(source)
            await db.commit()
            await db.refresh(source)
            print(json.dumps({"source_id": str(source.id), "url": source.canonical_uri}))
        elif args.command == "crawl":
            job = await enqueue_knowledge_job(db, "crawl_source", source_id=args.source_id)
            print(json.dumps({"job_id": str(job.id), "status": job.status}))
        elif args.command == "embed":
            payload = {}
            if args.release_id:
                result = await db.execute(
                    select(KnowledgeReleaseUnit.unit_id).where(
                        KnowledgeReleaseUnit.release_id == args.release_id
                    )
                )
                unit_ids = list(result.scalars().all())
                if not unit_ids:
                    raise ValueError("Release not found or contains no units")
                payload["unit_ids"] = [str(unit_id) for unit_id in unit_ids]
            job = await enqueue_knowledge_job(db, "embed_units", payload=payload)
            print(json.dumps({"job_id": str(job.id), "status": job.status}))
        elif args.command == "aliases-seed":
            inserted = await seed_default_aliases(db)
            print(json.dumps({"inserted": inserted}))
        elif args.command == "conflicts-detect":
            job = await enqueue_knowledge_job(db, "detect_conflicts")
            print(json.dumps({"job_id": str(job.id), "status": job.status}))
        elif args.command == "taxonomy-import":
            report = await import_ticket_taxonomy(db, args.csv_path)
            print(json.dumps(report.__dict__, indent=2))
        elif args.command == "units":
            query = select(KnowledgeUnit).where(KnowledgeUnit.status == args.status)
            if args.unit_type:
                query = query.where(KnowledgeUnit.unit_type == args.unit_type)
            if args.source_contains:
                query = query.where(
                    KnowledgeUnit.source_uri.ilike(f"%{args.source_contains}%")
                )
            result = await db.execute(
                query.order_by(KnowledgeUnit.stable_key)
                .offset(max(0, args.offset))
                .limit(max(1, min(args.limit, 500)))
            )
            print(json.dumps([
                {
                    "id": str(item.id),
                    "stable_key": item.stable_key,
                    "type": item.unit_type,
                    "title": item.title,
                    "source": item.source_uri,
                    "status": item.status,
                }
                for item in result.scalars().all()
            ], indent=2))
        elif args.command == "unit-show":
            result = await db.execute(
                select(KnowledgeUnit).where(KnowledgeUnit.id == args.unit_id)
            )
            item = result.scalars().first()
            if item is None:
                raise ValueError("Knowledge unit not found")
            print(json.dumps({
                "id": str(item.id),
                "stable_key": item.stable_key,
                "type": item.unit_type,
                "title": item.title,
                "question": item.question,
                "answer": item.answer,
                "voice_answer": item.voice_answer,
                "retrieval_text": item.retrieval_text,
                "source": item.source_uri,
                "source_label": item.source_label,
                "authority": item.authority,
                "audience": item.audience,
                "language": item.language,
                "region": item.region,
                "metadata": item.metadata_json,
                "status": item.status,
                "review_notes": item.review_notes,
            }, indent=2))
        elif args.command == "unit-approve":
            unit = await approve_unit(
                db,
                args.unit_id,
                voice_answer=args.voice_answer,
                atomic_answer=True if args.atomic_answer else None,
                answerability_reviewed=(
                    True if args.answerability_reviewed else None
                ),
                review_notes=args.notes,
            )
            print(json.dumps({
                "id": str(unit.id),
                "status": unit.status,
                "metadata": unit.metadata_json,
                "voice_answer": unit.voice_answer,
            }))
        elif args.command == "unit-retire":
            unit = await retire_unit(db, args.unit_id, review_notes=args.notes)
            print(json.dumps({"id": str(unit.id), "status": unit.status}))
        elif args.command == "release-create":
            unit_ids = list(args.unit_ids)
            if args.all_approved:
                approved = await db.execute(
                    select(KnowledgeUnit.id).where(KnowledgeUnit.status == "approved")
                )
                unit_ids = list(approved.scalars().all())
            if not unit_ids:
                raise ValueError("Provide unit IDs or use --all-approved")
            release = await create_release(
                db,
                version=args.version,
                unit_ids=unit_ids,
                description=args.description,
            )
            print(json.dumps({"id": str(release.id), "version": release.version, "status": release.status}))
        elif args.command == "release-validate":
            result = await validate_release(db, args.release_id)
            print(json.dumps(result.__dict__, indent=2))
        elif args.command == "release-publish":
            release = await publish_release(db, args.release_id)
            print(json.dumps({"id": str(release.id), "version": release.version, "status": release.status}))
        elif args.command == "release-rollback":
            release = await rollback_release(db, args.release_id)
            print(json.dumps({"id": str(release.id), "version": release.version, "status": release.status}))
        elif args.command == "status":
            result = await db.execute(
                select(KnowledgeRelease).order_by(KnowledgeRelease.created_at.desc())
            )
            print(json.dumps([
                {
                    "id": str(item.id),
                    "version": item.version,
                    "status": item.status,
                    "units": item.unit_count,
                    "corpus_hash": item.corpus_hash,
                }
                for item in result.scalars().all()
            ], indent=2))
        elif args.command == "jobs":
            result = await db.execute(
                select(KnowledgeJob).order_by(KnowledgeJob.created_at.desc()).limit(50)
            )
            print(json.dumps([
                {
                    "id": str(item.id),
                    "type": item.job_type,
                    "status": item.status,
                    "attempts": item.attempts,
                    "error": item.error,
                    "result": (item.payload or {}).get("result"),
                }
                for item in result.scalars().all()
            ], indent=2))
        elif args.command == "corpus-audit":
            print(json.dumps(await _corpus_audit(db), indent=2))
        elif args.command == "demo-prepare":
            existing = await db.execute(
                select(KnowledgeRelease).where(KnowledgeRelease.version == args.version)
            )
            release = existing.scalars().first()
            if release is not None:
                print(json.dumps({
                    "release_id": str(release.id),
                    "version": release.version,
                    "status": release.status,
                    "units": release.unit_count,
                    "existing": True,
                }, indent=2))
                return
            result = await db.execute(
                select(KnowledgeUnit)
                .join(KnowledgeDocument, KnowledgeUnit.document_id == KnowledgeDocument.id)
                .join(KnowledgeSnapshot, KnowledgeDocument.snapshot_id == KnowledgeSnapshot.id)
                .where(
                    KnowledgeSnapshot.source_id == args.source_id,
                    KnowledgeUnit.status.in_(("draft", "approved")),
                )
                .order_by(KnowledgeUnit.stable_key)
            )
            units = [
                unit for unit in result.scalars().all()
                if _is_demo_source_uri(unit.source_uri)
            ]
            if not units:
                raise ValueError("No eligible demo units found for this source")
            now = datetime.now(timezone.utc)
            for unit in units:
                if unit.status == "draft":
                    unit.status = "approved"
                    unit.approved_at = now
                    unit.review_notes = (
                        "DEMO ONLY: auto-selected from the public Mswipe website; "
                        "not reviewed for production use"
                    )
            release = await create_release(
                db,
                version=args.version,
                unit_ids=[unit.id for unit in units],
                description=(
                    "DEMO ONLY: narrow public product/support/contact corpus; "
                    "not approved for production customer support"
                ),
            )
            print(json.dumps({
                "release_id": str(release.id),
                "version": release.version,
                "status": release.status,
                "units": release.unit_count,
                "existing": False,
                "warning": "DEMO ONLY - not production reviewed",
            }, indent=2))
        elif args.command == "search":
            response = await retrieve_knowledge(
                args.query,
                db=db,
                top_k=max(1, min(args.top_k, 10)),
            )
            print(json.dumps({
                "status": response.status,
                "route": response.route.name,
                "reason": response.reason,
                "release": response.release_version,
                "confidence": response.confidence,
                "normalized_query": response.normalized_query,
                "hits": [
                    {
                        "stable_key": hit.stable_key,
                        "title": hit.title,
                        "answer": hit.voice_answer or hit.answer,
                        "source": hit.source_uri,
                        "score": hit.score,
                        "matched_by": hit.matched_by,
                    }
                    for hit in response.hits
                ],
            }, indent=2))


async def main() -> None:
    try:
        await run(parser().parse_args())
    finally:
        await engine.dispose()
        await voice_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
