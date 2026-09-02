"""Offline website normalization into raw snapshots and draft knowledge units."""

import asyncio
import hashlib
import os
import re
import tempfile
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse
from xml.etree import ElementTree

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import or_
from sqlalchemy.future import select

from core.knowledge_config import (
    KNOWLEDGE_CRAWL_DELAY_SECONDS,
    KNOWLEDGE_MAX_CRAWL_DEPTH,
    KNOWLEDGE_MAX_CRAWL_PAGES,
)
from core.models import (
    KnowledgeDocument,
    KnowledgeSnapshot,
    KnowledgeSource,
    KnowledgeUnit,
)
from services.document_ingestion import (
    canonical_document_from_html,
    canonical_document_from_markdown,
    canonical_document_from_structured_html,
    canonical_document_from_trafilatura,
    canonical_document_to_markdown,
    chunk_canonical_document,
    merge_structured_document,
    score_document,
    select_best_document,
)
from services.knowledge.fetch import (
    SourceHTTPError,
    SourceSkipped,
    canonicalize_url,
    fetch_public_source,
)
from services.knowledge.units import UnitInput, upsert_draft_unit


@dataclass(frozen=True)
class IngestionReport:
    pages_discovered: int
    pages_ingested: int
    pages_unchanged: int
    pages_skipped: int
    pages_excluded_from_units: int
    pages_failed: int
    pages_warned: int
    units_created_or_updated: int
    duplicate_draft_units_retired: int
    errors: tuple[str, ...]
    page_outcomes: tuple[dict, ...]


@dataclass(frozen=True)
class IngestionSource:
    """Detached source settings safe to reuse after a transaction rollback."""

    id: object
    language: str
    region: str
    audience: str
    authority: int
    crawl_policy: dict


_DEFAULT_EXCLUDED_UNIT_PATH_PREFIXES = (
    "/about-us",
    "/blog",
    "/careers",
    "/knowledge",
    "/mswipe-career",
)
_EXTRACTOR_VERSION = "4"
_CHUNK_POLICY_VERSION = os.getenv(
    "MSWIPE_KNOWLEDGE_CHUNK_POLICY_VERSION", "structure-v2"
).strip()


def _unit_generation_excluded(url: str, crawl_policy: dict) -> bool:
    configured = crawl_policy.get(
        "exclude_unit_path_prefixes", _DEFAULT_EXCLUDED_UNIT_PATH_PREFIXES
    )
    path = unquote(urlparse(url).path).rstrip("/") or "/"
    return any(
        path == prefix.rstrip("/") or path.startswith(f"{prefix.rstrip('/')}/")
        for prefix in configured
        if prefix and prefix.startswith("/")
    )


async def _retire_page_draft_units(db: AsyncSession, source_uri: str) -> int:
    result = await db.execute(
        select(KnowledgeUnit).where(
            KnowledgeUnit.source_uri == source_uri,
            KnowledgeUnit.status == "draft",
        )
    )
    units = result.scalars().all()
    for unit in units:
        unit.status = "retired"
    return len(units)


def _dedupe_fingerprint(answer: str) -> str:
    normalized = re.sub(r"\s+", " ", answer).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _simhash(answer: str) -> tuple[int, int]:
    tokens = re.findall(r"[\w-]+", answer.casefold())
    features = [" ".join(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
    if not features:
        features = tokens
    vector = [0] * 64
    for feature in features:
        fingerprint = int.from_bytes(
            hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        for bit in range(64):
            vector[bit] += 1 if fingerprint & (1 << bit) else -1
    value = sum(1 << bit for bit, weight in enumerate(vector) if weight >= 0)
    return value, len(tokens)


def _near_duplicate(left: str, right: str) -> bool:
    left_hash, left_length = _simhash(left)
    right_hash, right_length = _simhash(right)
    if min(left_length, right_length) < 12:
        return False
    if min(left_length, right_length) / max(left_length, right_length) < 0.9:
        return False
    return (left_hash ^ right_hash).bit_count() <= 2


def _dedupe_preference(unit: KnowledgeUnit) -> tuple:
    parsed = urlparse(unit.source_uri)
    support_rank = 0 if parsed.path.rstrip("/") == "/support" else 1
    return (
        support_rank,
        bool(parsed.query),
        len([part for part in parsed.path.split("/") if part]),
        len(unit.source_uri),
        unit.stable_key,
    )


async def _retire_duplicate_draft_units(db: AsyncSession, source_id) -> int:
    result = await db.execute(
        select(KnowledgeUnit)
        .join(KnowledgeDocument, KnowledgeUnit.document_id == KnowledgeDocument.id)
        .join(KnowledgeSnapshot, KnowledgeDocument.snapshot_id == KnowledgeSnapshot.id)
        .where(
            KnowledgeSnapshot.source_id == source_id,
            KnowledgeUnit.status.in_(("draft", "approved")),
        )
    )
    units = result.scalars().all()
    groups: dict[str, list[KnowledgeUnit]] = {}
    for unit in units:
        groups.setdefault(_dedupe_fingerprint(unit.answer), []).append(unit)

    # Merge only extremely close, same-type records. The narrow SimHash radius
    # catches punctuation/responsive-copy drift without collapsing different
    # product claims that merely share vocabulary.
    representatives = [items[0] for items in groups.values()]
    for index, left in enumerate(representatives):
        left_key = _dedupe_fingerprint(left.answer)
        if left_key not in groups:
            continue
        for right in representatives[index + 1 :]:
            right_key = _dedupe_fingerprint(right.answer)
            if right_key not in groups or left.unit_type != right.unit_type:
                continue
            if _near_duplicate(left.answer, right.answer):
                groups[left_key].extend(groups.pop(right_key))

    retired = 0
    for duplicates in groups.values():
        if len(duplicates) < 2:
            continue
        approved = [unit for unit in duplicates if unit.status == "approved"]
        keep = min(approved or duplicates, key=_dedupe_preference)
        for unit in duplicates:
            if unit.id != keep.id and unit.status == "draft":
                unit.status = "retired"
                unit.review_notes = "Automatically retired as an exact-answer duplicate"
                retired += 1
    if retired:
        await db.commit()
    return retired


def _is_concrete_page_url(url: str) -> bool:
    """Reject framework templates such as ``/blog/[slug]`` before fetching."""

    parsed = urlparse(url)
    route = unquote(f"{parsed.path}?{parsed.query}")
    return not any(marker in route for marker in ("[", "]", "{", "}"))


def _discover_links(html: str, base_url: str) -> list[str]:
    from bs4 import BeautifulSoup

    root_host = urlparse(base_url).hostname
    found: set[str] = set()
    soup = BeautifulSoup(html, "html.parser")
    for anchor in soup.find_all("a", href=True):
        raw = str(anchor.get("href") or "").strip()
        if not raw or raw.startswith(("mailto:", "tel:", "javascript:")):
            continue
        try:
            candidate = canonicalize_url(urljoin(base_url, raw))
        except ValueError:
            continue
        parsed = urlparse(candidate)
        if parsed.hostname != root_host:
            continue
        if not _is_concrete_page_url(candidate):
            continue
        if re.search(r"\.(?:jpg|jpeg|png|gif|svg|webp|zip|mp4|mp3|css|js)$", parsed.path, re.I):
            continue
        found.add(candidate)
    return sorted(found)


async def _discover_sitemap_urls(root_url: str) -> list[str]:
    parsed = urlparse(root_url)
    queue = deque([f"{parsed.scheme}://{parsed.netloc}/sitemap.xml"])
    visited_sitemaps: set[str] = set()
    discovered: set[str] = set()
    # A recursive index can be arbitrarily large. The website page limit still
    # bounds ingestion, while this separate cap prevents an index-only crawl.
    sitemap_limit = 50
    while queue and len(visited_sitemaps) < sitemap_limit:
        sitemap_url = queue.popleft()
        if sitemap_url in visited_sitemaps:
            continue
        visited_sitemaps.add(sitemap_url)
        try:
            fetched = await fetch_public_source(sitemap_url)
            root = ElementTree.fromstring(fetched.content)
        except Exception:
            continue
        is_index = root.tag.rsplit("}", 1)[-1] == "sitemapindex"
        for element in root.iter():
            if element.tag.rsplit("}", 1)[-1] != "loc" or not element.text:
                continue
            try:
                candidate = canonicalize_url(element.text.strip())
            except ValueError:
                continue
            if urlparse(candidate).hostname != parsed.hostname:
                continue
            if is_index:
                if candidate not in visited_sitemaps:
                    queue.append(candidate)
            elif _is_concrete_page_url(candidate):
                discovered.add(candidate)
    return sorted(discovered)


def _unit_type(title: str, content: str) -> str:
    text = f"{title} {content}".lower()
    if re.search(r"\b(?:error|declined|failed|not working|troubleshoot)\b", text):
        return "troubleshooting"
    if re.search(r"\b(?:step|how to|setup|install|activate|configure)\b", text):
        return "procedure"
    if re.search(r"\b(?:fee|charge|policy|terms|refund|settlement)\b", text):
        return "policy"
    if title.strip().endswith("?"):
        return "faq"
    if re.search(r"\b(?:specification|dimension|weight|battery|connectivity)\b", text):
        return "product_spec"
    return "definition"


def _pdf_to_markdown(content: bytes) -> str:
    from docling.document_converter import DocumentConverter

    descriptor, temporary_path = tempfile.mkstemp(suffix=".pdf")
    os.close(descriptor)
    path = Path(temporary_path)
    try:
        path.write_bytes(content)
        converted = DocumentConverter().convert(source=str(path))
        markdown = converted.document.export_to_markdown()
        if not markdown or not markdown.strip():
            raise ValueError("PDF extraction returned no useful content")
        return markdown
    finally:
        path.unlink(missing_ok=True)


async def _ingest_html_page(
    db: AsyncSession,
    source: IngestionSource,
    url: str,
) -> tuple[str, int, list[str]]:
    previous_result = await db.execute(
        select(KnowledgeSnapshot)
        .where(
            KnowledgeSnapshot.source_id == source.id,
            or_(
                KnowledgeSnapshot.requested_uri == url,
                KnowledgeSnapshot.final_uri == url,
            ),
            KnowledgeSnapshot.status == "normalized",
        )
        .order_by(KnowledgeSnapshot.fetched_at.desc())
        .limit(1)
    )
    previous_snapshot = previous_result.scalars().first()
    fetched = await fetch_public_source(
        url,
        etag=previous_snapshot.etag if previous_snapshot else None,
        last_modified=(
            previous_snapshot.last_modified if previous_snapshot else None
        ),
    )
    if fetched.status == 304:
        return "unchanged", 0, []
    content_hash = hashlib.sha256(fetched.content).hexdigest()
    existing = await db.execute(
        select(KnowledgeSnapshot).where(
            KnowledgeSnapshot.source_id == source.id,
            KnowledgeSnapshot.content_hash == content_hash,
        )
    )
    existing_snapshot = existing.scalars().first()
    existing_document = None
    if existing_snapshot is not None:
        document_result = await db.execute(
            select(KnowledgeDocument).where(
                KnowledgeDocument.snapshot_id == existing_snapshot.id
            )
        )
        existing_document = document_result.scalars().first()
    if existing_snapshot and existing_snapshot.status == "normalized":
        is_pdf = "application/pdf" in (fetched.headers.get("content-type") or "").lower()
        links = [] if is_pdf else _discover_links(
            fetched.content.decode(fetched.charset, errors="replace"), fetched.final_url
        )
        if _unit_generation_excluded(fetched.final_url, source.crawl_policy):
            await _retire_page_draft_units(db, fetched.final_url)
            await db.commit()
            return "excluded", 0, links
        if (
            existing_document is not None
            and existing_document.extractor_version == _EXTRACTOR_VERSION
        ):
            return "unchanged", 0, links
    snapshot = existing_snapshot or KnowledgeSnapshot(
        source_id=source.id,
        requested_uri=fetched.requested_url,
        content_hash=content_hash,
    )
    if existing_snapshot is None:
        db.add(snapshot)
        await db.flush()
    snapshot_id = snapshot.id
    snapshot_was_normalized = snapshot.status == "normalized"
    snapshot.requested_uri = fetched.requested_url
    snapshot.final_uri = fetched.final_url
    snapshot.status = "normalized" if snapshot_was_normalized else "fetching"
    snapshot.http_status = fetched.status
    snapshot.content_type = fetched.headers.get("content-type")
    snapshot.etag = fetched.headers.get("etag")
    snapshot.last_modified = fetched.headers.get("last-modified")
    snapshot.size_bytes = len(fetched.content)
    snapshot.fetched_at = datetime.now(timezone.utc)
    snapshot.error = None
    is_pdf = "application/pdf" in (fetched.headers.get("content-type") or "").lower()
    from core.storage import storage_client

    if not snapshot.raw_storage_key:
        snapshot.raw_storage_key = await storage_client.upload_knowledge_bytes(
            fetched.content,
            f"mswipe-knowledge/raw/{snapshot_id}{'.pdf' if is_pdf else '.html'}",
            content_type=(
                "application/pdf"
                if is_pdf
                else (fetched.headers.get("content-type") or "text/html")
            ),
        )
    snapshot.status = "normalized" if snapshot_was_normalized else "fetched"
    await db.commit()
    try:
        html = ""
        if is_pdf:
            markdown_source = await asyncio.to_thread(_pdf_to_markdown, fetched.content)
            document = canonical_document_from_markdown(
                markdown_source,
                fetched.final_url,
                title=Path(urlparse(fetched.final_url).path).name or "Mswipe PDF",
                extractor="docling_markdown_v1",
            )
        else:
            html = fetched.content.decode(fetched.charset, errors="replace")
            semantic_document, signals = canonical_document_from_html(
                html, fetched.final_url, extractor="semantic_html_v1"
            )
            candidates = [semantic_document]
            trafilatura_document = await asyncio.to_thread(
                canonical_document_from_trafilatura,
                html,
                fetched.final_url,
                title=semantic_document.title,
                site_name=semantic_document.site_name,
            )
            if trafilatura_document is not None:
                candidates.append(trafilatura_document)
            document = select_best_document(candidates, signals)
            structured_document = canonical_document_from_structured_html(
                html,
                fetched.final_url,
            )
            document = merge_structured_document(document, structured_document)
        markdown = canonical_document_to_markdown(document)
        warnings = list(document.warnings)
        if not markdown.strip():
            raise ValueError("Canonical extraction returned no useful content")
        record = existing_document or KnowledgeDocument(snapshot_id=snapshot_id)
        record.canonical_uri = fetched.final_url
        record.title = document.title or fetched.final_url
        record.canonical_markdown = markdown
        record.extractor = document.extractor
        record.extractor_version = _EXTRACTOR_VERSION
        record.language = source.language
        record.quality_score = document.quality_score
        record.warnings = warnings
        record.metadata_json = {"site_name": document.site_name}
        if existing_document is None:
            db.add(record)
        await db.flush()
        if _unit_generation_excluded(fetched.final_url, source.crawl_policy):
            await _retire_page_draft_units(db, fetched.final_url)
            snapshot.status = "normalized"
            snapshot.quality_score = document.quality_score
            snapshot.warnings = warnings
            await db.commit()
            return (
                "excluded",
                0,
                _discover_links(html, fetched.final_url) if html else [],
            )
        chunks = chunk_canonical_document(
            document,
            max_tokens=int((source.crawl_policy or {}).get("unit_max_tokens", 360)),
            overlap_tokens=0,
            min_content_chars=40,
        )
        if not chunks:
            raise ValueError("Canonical extraction produced no publishable knowledge units")
        url_key = hashlib.sha256(fetched.final_url.encode("utf-8")).hexdigest()[:16]
        current_unit_ids = []
        for index, chunk in enumerate(chunks):
            title = chunk.heading_path or document.title or f"Mswipe information {index + 1}"
            structured_types = {
                record.get("structured_type")
                for record in chunk.metadata.get("source_records", [])
                if record.get("structured_type")
            }
            unit_type = (
                "faq"
                if structured_types == {"faq"}
                else "procedure"
                if structured_types == {"procedure"}
                else "product_spec"
                if structured_types == {"table_record"}
                else _unit_type(title, chunk.content)
            )
            identity_text = f"{title.casefold()}|{unit_type}"
            unit_key = hashlib.sha256(identity_text.encode("utf-8")).hexdigest()[:16]
            current_unit = await upsert_draft_unit(
                db,
                UnitInput(
                    stable_key=f"web.{url_key}.{unit_key}",
                    document_id=record.id,
                    unit_type=unit_type,
                    title=title,
                    question=title if title.endswith("?") else None,
                    answer=chunk.content,
                    retrieval_text=chunk.retrieval_text,
                    source_uri=fetched.final_url,
                    source_label=document.title or urlparse(fetched.final_url).path,
                    language=source.language,
                    region=source.region,
                    audience=source.audience,
                    authority=source.authority,
                    metadata={
                        "heading_path": chunk.heading_path,
                        "chunk_policy_version": _CHUNK_POLICY_VERSION,
                        "atomic_answer": unit_type == "faq",
                        "answerability_reviewed": False,
                        "voice_answer_approved": False,
                        **chunk.metadata,
                    },
                ),
            )
            current_unit_ids.append(current_unit.id)
        previous_units = await db.execute(
            select(KnowledgeUnit).where(
                KnowledgeUnit.source_uri == fetched.final_url,
                KnowledgeUnit.id.not_in(current_unit_ids),
                KnowledgeUnit.status.in_(("draft", "approved")),
            )
        )
        for previous in previous_units.scalars().all():
            previous.status = "retired"
        snapshot.status = "normalized"
        snapshot.quality_score = document.quality_score
        snapshot.warnings = warnings
    except Exception as exc:
        await db.rollback()
        result = await db.execute(
            select(KnowledgeSnapshot).where(KnowledgeSnapshot.id == snapshot_id)
        )
        failed_snapshot = result.scalars().first()
        if failed_snapshot:
            failed_snapshot.status = "normalized" if snapshot_was_normalized else "failed"
            failed_snapshot.error = f"{type(exc).__name__}: {exc}"[:4000]
            await db.commit()
        raise
    await db.commit()
    return (
        "ingested",
        len(chunks),
        _discover_links(html, fetched.final_url) if html else [],
    )


async def crawl_website_source(
    db: AsyncSession,
    source_id,
) -> IngestionReport:
    result = await db.execute(
        select(KnowledgeSource).where(KnowledgeSource.id == source_id)
    )
    source = result.scalars().first()
    if source is None or not source.enabled:
        raise ValueError("Enabled knowledge source not found")
    if source.source_type != "website":
        raise ValueError("crawl_website only supports website sources")
    page_limit = min(
        KNOWLEDGE_MAX_CRAWL_PAGES,
        int((source.crawl_policy or {}).get("max_pages", KNOWLEDGE_MAX_CRAWL_PAGES)),
    )
    depth_limit = min(
        KNOWLEDGE_MAX_CRAWL_DEPTH,
        int((source.crawl_policy or {}).get("max_depth", KNOWLEDGE_MAX_CRAWL_DEPTH)),
    )
    root_url = canonicalize_url(source.canonical_uri)
    seed_urls = [root_url]
    if (source.crawl_policy or {}).get("use_sitemap", True):
        seed_urls.extend(await _discover_sitemap_urls(root_url))
    source_settings = IngestionSource(
        id=source.id,
        language=source.language,
        region=source.region,
        audience=source.audience,
        authority=source.authority,
        crawl_policy=dict(source.crawl_policy or {}),
    )
    queue = deque((url, 0) for url in dict.fromkeys(seed_urls))
    seen: set[str] = set()
    discovered = set(seed_urls)
    ingested = unchanged = skipped = excluded = failed = warned = unit_count = 0
    errors: list[str] = []
    page_outcomes: list[dict] = []
    while queue and len(seen) < page_limit:
        url, depth = queue.popleft()
        if url in seen:
            continue
        seen.add(url)
        try:
            status, units, links = await _ingest_html_page(db, source_settings, url)
            if status == "ingested":
                ingested += 1
                unit_count += units
            elif status == "excluded":
                excluded += 1
            else:
                unchanged += 1
            page_outcomes.append({"url": url, "outcome": status})
            if depth < depth_limit:
                new_links = [link for link in links if link not in discovered]
                discovered.update(new_links)
                queue.extend((link, depth + 1) for link in new_links)
        except SourceSkipped as exc:
            skipped += 1
            page_outcomes.append(
                {"url": url, "outcome": "policy_skipped", "reason": str(exc)}
            )
        except SourceHTTPError as exc:
            if exc.status == 404:
                warned += 1
                page_outcomes.append(
                    {
                        "url": url,
                        "outcome": "source_warning",
                        "reason": "stale_http_404",
                    }
                )
            else:
                failed += 1
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
                page_outcomes.append(
                    {"url": url, "outcome": "failed", "reason": f"http_{exc.status}"}
                )
        except Exception as exc:
            await db.rollback()
            failed += 1
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            page_outcomes.append(
                {
                    "url": url,
                    "outcome": "failed",
                    "reason": type(exc).__name__,
                }
            )
        finally:
            if KNOWLEDGE_CRAWL_DELAY_SECONDS:
                await asyncio.sleep(KNOWLEDGE_CRAWL_DELAY_SECONDS)
    duplicate_units_retired = await _retire_duplicate_draft_units(db, source_settings.id)
    return IngestionReport(
        pages_discovered=len(discovered),
        pages_ingested=ingested,
        pages_unchanged=unchanged,
        pages_skipped=skipped,
        pages_excluded_from_units=excluded,
        pages_failed=failed,
        pages_warned=warned,
        units_created_or_updated=unit_count,
        duplicate_draft_units_retired=duplicate_units_retired,
        errors=tuple(errors[:100]),
        page_outcomes=tuple(page_outcomes[:page_limit]),
    )
