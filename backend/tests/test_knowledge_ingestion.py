import asyncio

from services.knowledge.ingestion import (
    _discover_links,
    _discover_sitemap_urls,
    _is_concrete_page_url,
    _near_duplicate,
    _unit_generation_excluded,
)
from services.document_ingestion import (
    canonical_document_from_html,
    canonical_document_from_structured_html,
    chunk_canonical_document,
    merge_structured_document,
)
from services.knowledge.fetch import FetchResult


def test_placeholder_routes_are_not_concrete_pages():
    assert not _is_concrete_page_url("https://www.mswipe.com/blog/[slug]")
    assert not _is_concrete_page_url("https://www.mswipe.com/blog/%5Bslug%5D")
    assert not _is_concrete_page_url("https://www.mswipe.com/product/{product_id}")
    assert _is_concrete_page_url("https://www.mswipe.com/blog/payment-guide")


def test_discovery_keeps_only_concrete_same_site_documents():
    html = """
    <a href="/support">Support</a>
    <a href="/blog/[slug]">Template</a>
    <a href="/blog/%5Bslug%5D">Encoded template</a>
    <a href="/assets/logo.png">Image</a>
    <a href="https://example.com/elsewhere">External</a>
    <a href="mailto:help@mswipe.com">Email</a>
    """

    assert _discover_links(html, "https://www.mswipe.com/") == [
        "https://www.mswipe.com/support"
    ]


def test_default_unit_policy_archives_but_excludes_noisy_sections():
    assert _unit_generation_excluded("https://www.mswipe.com/blog/payment-guide", {})
    assert _unit_generation_excluded(
        "https://www.mswipe.com/knowledge?type=pressRelease", {}
    )
    assert not _unit_generation_excluded("https://www.mswipe.com/support", {})
    assert not _unit_generation_excluded(
        "https://www.mswipe.com/in-store-solutions/wisepos-plus", {}
    )


def test_unit_exclusion_policy_can_be_overridden_per_source():
    policy = {"exclude_unit_path_prefixes": []}
    assert not _unit_generation_excluded(
        "https://www.mswipe.com/blog/payment-guide", policy
    )


def test_html_normalization_removes_container_and_responsive_duplicates():
    html = """
    <main>
      <h1>Support</h1>
      <p>Customer Support and 1800 100 200</p>
      <dd><p>Customer Support</p><p>Customer Support</p></dd>
      <p>WisePOS Plus</p><p>WisePOS Plus</p>
    </main>
    """

    document, _signals = canonical_document_from_html(
        html, "https://www.mswipe.com/support"
    )
    assert [block.text for block in document.blocks] == [
        "Support",
        "Customer Support and 1800 100 200",
        "WisePOS Plus",
    ]


def test_json_ld_faqs_become_atomic_chunks():
    html = """
    <html><head><title>Help</title>
    <script type="application/ld+json">
    {"@type":"FAQPage","mainEntity":[
      {"@type":"Question","name":"How do I reset it?",
       "acceptedAnswer":{"@type":"Answer","text":"Hold the power key for ten seconds."}},
      {"@type":"Question","name":"How do I charge it?",
       "acceptedAnswer":{"@type":"Answer","text":"Connect the supplied power adapter."}}
    ]}
    </script></head><body><main><h1>Help</h1></main></body></html>
    """

    structured = canonical_document_from_structured_html(
        html, "https://www.mswipe.com/help"
    )
    assert structured is not None
    chunks = chunk_canonical_document(
        structured, max_tokens=100, overlap_tokens=0, min_content_chars=10
    )

    assert [chunk.content for chunk in chunks] == [
        "Hold the power key for ten seconds.",
        "Connect the supplied power adapter.",
    ]
    assert all(
        chunk.metadata["source_records"][0]["structured_type"] == "faq"
        for chunk in chunks
    )


def test_accessibility_control_relation_becomes_atomic_faq():
    html = """
    <main>
      <button aria-controls="answer-one">What is the settlement window?</button>
      <div id="answer-one">The approved source defines the settlement window here.</div>
    </main>
    """
    structured = canonical_document_from_structured_html(
        html, "https://www.mswipe.com/help"
    )

    assert structured is not None
    assert [block.text for block in structured.blocks] == [
        "What is the settlement window?",
        "The approved source defines the settlement window here.",
    ]


def test_structured_merge_removes_combined_duplicate_answer_block():
    html = """
    <main><h1>Help</h1><p>First complete answer with enough detail. Second complete answer with enough detail.</p></main>
    <details><summary>First question?</summary>First complete answer with enough detail.</details>
    <details><summary>Second question?</summary>Second complete answer with enough detail.</details>
    """
    primary, _ = canonical_document_from_html(html, "https://www.mswipe.com/help")
    structured = canonical_document_from_structured_html(
        html, "https://www.mswipe.com/help"
    )
    merged = merge_structured_document(primary, structured)

    texts = [block.text for block in merged.blocks]
    assert "First complete answer with enough detail. Second complete answer with enough detail." not in texts
    assert "First question?" in texts
    assert "Second question?" in texts


def test_sitemap_index_is_followed_and_filters_non_concrete_urls(monkeypatch):
    sitemap_index = b"""<?xml version="1.0"?>
      <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <sitemap><loc>https://www.mswipe.com/products.xml</loc></sitemap>
      </sitemapindex>"""
    url_set = b"""<?xml version="1.0"?>
      <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://www.mswipe.com/device-one</loc></url>
        <url><loc>https://www.mswipe.com/device/[slug]</loc></url>
      </urlset>"""

    async def fake_fetch(url):
        content = sitemap_index if url.endswith("sitemap.xml") else url_set
        return FetchResult(url, url, 200, {"content-type": "application/xml"}, content, "utf-8")

    monkeypatch.setattr("services.knowledge.ingestion.fetch_public_source", fake_fetch)
    assert asyncio.run(_discover_sitemap_urls("https://www.mswipe.com/")) == [
        "https://www.mswipe.com/device-one"
    ]


def test_table_rows_become_independently_answerable_schema_records():
    html = """
    <main><h2>Device specifications</h2>
      <table><tr><th>Model</th><th>Battery</th></tr>
      <tr><td>Alpha</td><td>Eight hours</td></tr>
      <tr><td>Beta</td><td>Ten hours</td></tr></table>
    </main>
    """
    document, _ = canonical_document_from_html(html, "https://www.mswipe.com/devices")
    chunks = chunk_canonical_document(
        document, max_tokens=100, overlap_tokens=0, min_content_chars=10
    )

    assert [chunk.content for chunk in chunks] == [
        "Model: Alpha; Battery: Eight hours",
        "Model: Beta; Battery: Ten hours",
    ]
    assert all(
        chunk.metadata["source_records"][0]["structured_type"] == "table_record"
        for chunk in chunks
    )


def test_near_duplicate_fingerprint_is_conservative():
    original = "The terminal accepts card payments and provides a receipt after every completed transaction for the merchant."
    cosmetic_copy = "The terminal accepts card payments, and provides a receipt after every completed transaction for the merchant."
    different_claim = "The terminal accepts card payments and sends a payment notification to the merchant application after settlement."

    assert _near_duplicate(original, cosmetic_copy)
    assert not _near_duplicate(original, different_claim)
