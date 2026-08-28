"""Source-neutral document structure and chunking for RAG ingestion.

Format adapters should convert their input into :class:`CanonicalDocument`.
Everything after that boundary is intentionally independent of HTML, PDF, or
any particular source layout.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse


_BLOCK_TAGS = {
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "li",
    "table",
    "pre",
    "blockquote",
    "dt",
    "dd",
}
_PRUNED_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "canvas",
    "iframe",
    "template",
    "form",
    "button",
    "input",
    "select",
    "textarea",
    "nav",
    "footer",
    "aside",
}
_TOKEN_RE = re.compile(r"[\w]+(?:[-'][\w]+)*|[^\w\s]", re.UNICODE)
_WORD_RE = re.compile(r"[\w]+(?:[-'][\w]+)*", re.UNICODE)


@dataclass(frozen=True)
class DocumentBlock:
    kind: str
    text: str
    order: int
    heading_level: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalDocument:
    source_url: str
    title: str | None
    site_name: str | None
    blocks: tuple[DocumentBlock, ...]
    extractor: str
    quality_score: float = 0.0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceSignals:
    visible_text: str
    headings: tuple[str, ...]
    numeric_anchors: tuple[str, ...]


@dataclass(frozen=True)
class DocumentChunk:
    content: str
    retrieval_text: str
    heading_path: str | None
    token_count: int
    metadata: dict[str, Any]


def token_count(value: str) -> int:
    return len(_TOKEN_RE.findall(value or ""))


def _space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _plain(value: str) -> str:
    value = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", value or "")
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"<https?://[^>]+>", " ", value)
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"[*_`#>|]", " ", value)
    return _space(value)


def _deduplicate_blocks(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    """Remove repeated responsive/hidden copies while preserving source order."""

    seen: set[tuple[str, str]] = set()
    unique: list[DocumentBlock] = []
    for block in blocks:
        fingerprint = (block.kind, _plain(block.text).casefold())
        if not fingerprint[1] or fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(replace(block, order=len(unique)))

    normalized = [_plain(block.text).casefold() for block in unique]
    compact: list[DocumentBlock] = []
    for index, block in enumerate(unique):
        text = normalized[index]
        contained_elsewhere = (
            block.kind != "heading"
            and len(text) >= 15
            and any(
                other_index != index
                and unique[other_index].kind != "heading"
                and text != other
                and text in other
                for other_index, other in enumerate(normalized)
            )
        )
        if not contained_elsewhere:
            compact.append(replace(block, order=len(compact)))
    return compact


def clean_content(value: str) -> str:
    """Remove link targets and formatting while preserving record boundaries."""
    value = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", value or "")
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"<https?://[^>]+>", " ", value)
    value = re.sub(r"https?://\S+", " ", value)
    lines = []
    for raw_line in value.splitlines():
        line = re.sub(r"[*_`#>|]+", " ", raw_line)
        line = _space(line)
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _inline_markdown(node: Any, base_url: str) -> str:
    from bs4 import NavigableString, Tag

    parts: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
            continue
        if not isinstance(child, Tag) or child.name in _PRUNED_TAGS:
            continue
        if child.name in {"ul", "ol", "table"}:
            continue
        if child.name == "br":
            parts.append("\n")
            continue
        if child.name == "a":
            label = _space(child.get_text(" ", strip=True))
            target = (child.get("href") or "").strip()
            if label and target:
                parts.append(f"[{label}]({urljoin(base_url, target)})")
            elif label:
                parts.append(label)
            continue
        parts.append(_inline_markdown(child, base_url))
    return _space(" ".join(parts))


def _table_markdown(table: Any, base_url: str) -> str:
    rows: list[list[str]] = []
    for row in table.find_all("tr"):
        cells = [
            _inline_markdown(cell, base_url)
            for cell in row.find_all(["th", "td"], recursive=False)
        ]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header = rows[0]
    rendered = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    rendered.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(rendered)


def _metadata_from_soup(soup: Any, source_url: str) -> tuple[str | None, str | None]:
    def meta_value(*selectors: tuple[str, str]) -> str | None:
        for attribute, value in selectors:
            tag = soup.find("meta", attrs={attribute: value})
            content = _space(tag.get("content", "")) if tag else ""
            if content:
                return content
        return None

    title = meta_value(("property", "og:title"), ("name", "twitter:title"))
    if not title and soup.title:
        title = _space(soup.title.get_text(" ", strip=True)) or None
    site_name = meta_value(("property", "og:site_name"))
    if not site_name:
        site_name = urlparse(source_url).hostname
    return title, site_name


def canonical_document_from_html(
    html: str,
    source_url: str,
    *,
    extractor: str = "semantic_html",
) -> tuple[CanonicalDocument, SourceSignals]:
    """Convert ordinary semantic HTML into ordered, source-neutral blocks."""
    from bs4 import BeautifulSoup, Tag

    soup = BeautifulSoup(html or "", "html.parser")
    title, site_name = _metadata_from_soup(soup, source_url)
    roots = list(soup.select("main, article, [role='main']"))
    root = max(roots, key=lambda item: len(item.get_text(" ", strip=True)), default=None)
    root = root or soup.body or soup

    for tag in list(root.find_all(_PRUNED_TAGS)):
        tag.decompose()

    visible_text = _space(root.get_text(" ", strip=True))
    headings = tuple(
        text
        for tag in root.find_all(re.compile(r"^h[1-6]$"))
        if (text := _space(tag.get_text(" ", strip=True)))
    )
    numeric_anchors = tuple(sorted(set(re.findall(r"\b\d[\w./:+-]*\b", " ".join(headings)))))

    blocks: list[DocumentBlock] = []
    emitted: set[int] = set()
    for element in root.descendants:
        if not isinstance(element, Tag) or element.name not in _BLOCK_TAGS:
            continue
        if id(element) in emitted:
            continue
        if element.find_parent("table") is not None and element.name != "table":
            continue
        if element.name == "p" and element.find_parent(["li", "blockquote"]) is not None:
            continue
        if element.name in {"blockquote", "dt", "dd"} and element.find(
            _BLOCK_TAGS - {element.name}
        ):
            # Emit the semantic child blocks, not both the container's combined
            # text and each child again.
            continue

        if element.name == "p":
            direct_links = element.find_all("a", recursive=False)
            direct_text = _space(
                " ".join(
                    str(child)
                    for child in element.children
                    if getattr(child, "name", None) is None
                )
            )
            # A paragraph made primarily from sibling anchors is semantically a
            # record list even if the page omitted ul/li markup. Preserve each
            # anchor as an atomic item so later cleaning cannot erase boundaries.
            if len(direct_links) >= 2 and not re.search(r"\w", direct_text, re.UNICODE):
                for link in direct_links:
                    label = _space(link.get_text(" ", strip=True))
                    target = (link.get("href") or "").strip()
                    text = f"[{label}]({urljoin(source_url, target)})" if label and target else label
                    if text:
                        blocks.append(
                            DocumentBlock("list_item", text, len(blocks))
                        )
                emitted.add(id(element))
                continue

        kind = element.name
        level = None
        if re.fullmatch(r"h[1-6]", kind):
            level = int(kind[1])
            text = _space(element.get_text(" ", strip=True))
            kind = "heading"
        elif kind == "table":
            text = _table_markdown(element, source_url)
        elif kind == "li":
            text = _inline_markdown(element, source_url)
            kind = "list_item"
        elif kind == "pre":
            text = element.get_text("\n", strip=True)
            kind = "code"
        else:
            text = _inline_markdown(element, source_url)
            kind = "paragraph" if kind == "p" else kind
        text = text.strip()
        if not text:
            continue
        blocks.append(
            DocumentBlock(
                kind=kind,
                text=text,
                order=len(blocks),
                heading_level=level,
            )
        )
        emitted.add(id(element))

    # Some modern pages put prose directly in leaf divs rather than semantic
    # paragraph elements. Recover those blocks without duplicating descendants.
    captured_text = _plain(" ".join(block.text for block in blocks))
    if len(captured_text) < max(80, int(len(visible_text) * 0.45)):
        for element in root.find_all(["div", "section"]):
            if element.find(_BLOCK_TAGS):
                continue
            text = _inline_markdown(element, source_url)
            if len(_plain(text)) < 30 or _plain(text) in captured_text:
                continue
            blocks.append(
                DocumentBlock(kind="paragraph", text=text, order=len(blocks))
            )

    blocks = _deduplicate_blocks(blocks)

    document = CanonicalDocument(
        source_url=source_url,
        title=title,
        site_name=site_name,
        blocks=tuple(blocks),
        extractor=extractor,
    )
    return document, SourceSignals(visible_text, headings, numeric_anchors)


def canonical_document_from_markdown(
    markdown: str,
    source_url: str,
    *,
    title: str | None = None,
    site_name: str | None = None,
    extractor: str = "markdown",
) -> CanonicalDocument:
    blocks: list[DocumentBlock] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph:
            return
        text = "\n".join(paragraph).strip()
        paragraph.clear()
        if text:
            blocks.append(DocumentBlock("paragraph", text, len(blocks)))

    for raw_line in re.sub(r"\r\n?", "\n", markdown or "").splitlines():
        line = raw_line.strip()
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            blocks.append(
                DocumentBlock(
                    "heading",
                    heading.group(2).strip(),
                    len(blocks),
                    heading_level=len(heading.group(1)),
                )
            )
        elif re.match(r"^[-*+]\s+\S", line):
            flush_paragraph()
            blocks.append(
                DocumentBlock("list_item", re.sub(r"^[-*+]\s+", "", line), len(blocks))
            )
        elif not line:
            flush_paragraph()
        else:
            paragraph.append(line)
    flush_paragraph()
    return CanonicalDocument(
        source_url=source_url,
        title=title,
        site_name=site_name,
        blocks=tuple(blocks),
        extractor=extractor,
    )


def canonical_document_to_markdown(document: CanonicalDocument) -> str:
    lines: list[str] = []
    for block in sorted(document.blocks, key=lambda item: item.order):
        if block.kind == "heading":
            level = max(1, min(block.heading_level or 2, 6))
            lines.append(f"{'#' * level} {block.text}")
        elif block.kind == "list_item":
            lines.append(f"- {block.text}")
        elif block.kind == "code":
            lines.append(f"```\n{block.text}\n```")
        else:
            lines.append(block.text)
    return "\n\n".join(line for line in lines if line.strip()).strip()


def _terms(value: str) -> set[str]:
    return {word.casefold() for word in _WORD_RE.findall(_plain(value)) if len(word) > 1}


def score_document(document: CanonicalDocument, signals: SourceSignals) -> CanonicalDocument:
    candidate_markdown = canonical_document_to_markdown(document)
    candidate_plain = _plain(candidate_markdown)
    source_terms = _terms(signals.visible_text)
    candidate_terms = _terms(candidate_plain)
    text_coverage = (
        len(source_terms & candidate_terms) / len(source_terms) if source_terms else 1.0
    )

    normalized_candidate = candidate_plain.casefold()
    heading_coverage = (
        sum(_plain(heading).casefold() in normalized_candidate for heading in signals.headings)
        / len(signals.headings)
        if signals.headings
        else 1.0
    )
    numeric_coverage = (
        sum(anchor.casefold() in normalized_candidate for anchor in signals.numeric_anchors)
        / len(signals.numeric_anchors)
        if signals.numeric_anchors
        else 1.0
    )
    candidate_headings = sum(block.kind == "heading" for block in document.blocks)
    structure_coverage = (
        min(1.0, candidate_headings / len(signals.headings))
        if signals.headings
        else (1.0 if document.blocks else 0.0)
    )
    normalized_lines = [
        _plain(line).casefold()
        for line in candidate_markdown.splitlines()
        if _plain(line)
    ]
    uniqueness = (
        len(set(normalized_lines)) / len(normalized_lines) if normalized_lines else 0.0
    )
    score = (
        0.35 * text_coverage
        + 0.30 * heading_coverage
        + 0.15 * numeric_coverage
        + 0.10 * structure_coverage
        + 0.10 * uniqueness
    )

    warnings: list[str] = []
    if heading_coverage < 0.8:
        warnings.append(
            f"heading coverage {heading_coverage:.0%} ({candidate_headings}/{len(signals.headings)})"
        )
    if numeric_coverage < 0.9:
        warnings.append(f"numeric-anchor coverage {numeric_coverage:.0%}")
    if text_coverage < 0.55:
        warnings.append(f"visible-text coverage {text_coverage:.0%}")
    if uniqueness < 0.7:
        warnings.append(f"duplicate-line ratio {1.0 - uniqueness:.0%}")
    return replace(
        document,
        quality_score=round(max(0.0, min(score, 1.0)), 4),
        warnings=tuple(warnings),
    )


def select_best_document(
    candidates: Iterable[CanonicalDocument],
    signals: SourceSignals,
) -> CanonicalDocument:
    scored = [score_document(candidate, signals) for candidate in candidates if candidate.blocks]
    if not scored:
        raise ValueError("No extractor produced structured document blocks")
    return max(
        scored,
        key=lambda item: (
            item.quality_score,
            sum(block.kind == "heading" for block in item.blocks),
            len(canonical_document_to_markdown(item)),
        ),
    )


def _split_oversized_text(value: str, max_tokens: int) -> list[str]:
    words = re.findall(r"\S+", value or "")
    if not words:
        return []
    return [" ".join(words[start : start + max_tokens]) for start in range(0, len(words), max_tokens)]


def _section_chunks(
    blocks: list[DocumentBlock],
    max_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, int, int]]:
    units: list[tuple[str, int]] = []
    for block in blocks:
        prefix = "- " if block.kind == "list_item" else ""
        text = f"{prefix}{block.text}".strip()
        if token_count(text) <= max_tokens:
            units.append((text, block.order))
        else:
            units.extend((part, block.order) for part in _split_oversized_text(text, max_tokens))

    chunks: list[tuple[str, int, int]] = []
    current: list[tuple[str, int]] = []
    current_tokens = 0
    for unit, order in units:
        unit_tokens = token_count(unit)
        if current and current_tokens + unit_tokens > max_tokens:
            chunks.append(("\n".join(text for text, _ in current), current[0][1], current[-1][1]))
            overlap: list[tuple[str, int]] = []
            overlap_count = 0
            for previous in reversed(current):
                previous_tokens = token_count(previous[0])
                if overlap and overlap_count + previous_tokens > overlap_tokens:
                    break
                overlap.insert(0, previous)
                overlap_count += previous_tokens
            current = overlap
            current_tokens = overlap_count
        current.append((unit, order))
        current_tokens += unit_tokens
    if current:
        chunks.append(("\n".join(text for text, _ in current), current[0][1], current[-1][1]))
    return chunks


def chunk_canonical_document(
    document: CanonicalDocument,
    *,
    max_tokens: int,
    overlap_tokens: int,
    min_content_chars: int,
) -> list[DocumentChunk]:
    """Chunk within heading boundaries and enrich only the retrieval view."""
    max_tokens = max(32, max_tokens)
    overlap_tokens = max(0, min(overlap_tokens, max_tokens // 3))
    heading_stack: list[str] = []
    section_blocks: list[DocumentBlock] = []
    section_heading_path: tuple[str, ...] = ()
    chunks: list[DocumentChunk] = []

    def flush() -> None:
        nonlocal section_blocks
        if not section_blocks:
            return
        for content, block_start, block_end in _section_chunks(
            section_blocks, max_tokens, overlap_tokens
        ):
            clean = clean_content(content)
            if len(clean) < min_content_chars:
                continue
            heading_bits = [bit for bit in [document.title, *section_heading_path] if bit]
            deduped_heading_bits = list(dict.fromkeys(heading_bits))
            heading_path = " > ".join(deduped_heading_bits) or None
            retrieval_parts = [
                f"Title: {document.title}" if document.title else None,
                f"Heading: {' > '.join(section_heading_path)}" if section_heading_path else None,
                clean,
            ]
            retrieval_text = "\n".join(part for part in retrieval_parts if part)
            chunks.append(
                DocumentChunk(
                    content=clean,
                    retrieval_text=retrieval_text,
                    heading_path=heading_path,
                    token_count=token_count(clean),
                    metadata={
                        "block_start": block_start,
                        "block_end": block_end,
                        "extractor": document.extractor,
                        "quality_score": document.quality_score,
                        "link_count": len(re.findall(r"\[[^\]]+\]\([^)]*\)", content)),
                        "plain_chars": len(_plain(content)),
                    },
                )
            )
        section_blocks = []

    for block in sorted(document.blocks, key=lambda item: item.order):
        if block.kind == "heading":
            flush()
            level = max(1, min(block.heading_level or 2, 6))
            heading_stack[level - 1 :] = []
            while len(heading_stack) < level - 1:
                heading_stack.append("")
            heading_stack.append(_plain(block.text))
            section_heading_path = tuple(bit for bit in heading_stack if bit)
        else:
            section_blocks.append(block)
    flush()
    return chunks


def estimated_chunk_count(document: CanonicalDocument, max_tokens: int) -> int:
    total = sum(token_count(block.text) for block in document.blocks if block.kind != "heading")
    return max(1, math.ceil(total / max(1, max_tokens)))
