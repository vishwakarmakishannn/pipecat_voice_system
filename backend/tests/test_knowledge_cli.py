from knowledge_cli import _is_demo_source_uri, parser


def test_demo_scope_is_narrow_and_excludes_legal_and_marketing_pages():
    assert _is_demo_source_uri("https://www.mswipe.com/support")
    assert _is_demo_source_uri(
        "https://www.mswipe.com/in-store-solutions/wisepos-plus"
    )
    assert _is_demo_source_uri("https://www.mswipe.com/online-solutions/pay-by-link")
    assert not _is_demo_source_uri("https://www.mswipe.com/")
    assert not _is_demo_source_uri("https://www.mswipe.com/merchantagreement")
    assert not _is_demo_source_uri("https://www.mswipe.com/blog/payment-guide")


def test_embed_can_be_scoped_to_release():
    args = parser().parse_args(
        ["embed", "--release-id", "11111111-1111-1111-1111-111111111111"]
    )
    assert str(args.release_id) == "11111111-1111-1111-1111-111111111111"
