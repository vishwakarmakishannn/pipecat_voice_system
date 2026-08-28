from services.knowledge.ingestion import (
    _discover_links,
    _is_concrete_page_url,
    _unit_generation_excluded,
)
from services.document_ingestion import canonical_document_from_html


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
