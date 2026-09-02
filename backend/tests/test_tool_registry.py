import tools.registry as registry


def _names(tools):
    return [tool.__name__ for tool in tools]


def test_mswipe_runtime_surface_excludes_web_by_default(monkeypatch):
    monkeypatch.setattr(registry, "KNOWLEDGE_ENABLED", True)
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")

    assert _names(registry.configured_voice_tools()) == [
        "search_mswipe_knowledge",
        "manage_issue_draft",
        "get_current_datetime",
    ]


def test_web_search_is_added_only_by_explicit_feature_flag(monkeypatch):
    monkeypatch.setattr(registry, "KNOWLEDGE_ENABLED", True)
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")

    assert _names(registry.configured_voice_tools()) == [
        "search_mswipe_knowledge",
        "manage_issue_draft",
        "get_current_datetime",
        "tavily_search",
    ]


def test_disabled_knowledge_is_not_advertised_to_the_model(monkeypatch):
    monkeypatch.setattr(registry, "KNOWLEDGE_ENABLED", False)
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")

    assert _names(registry.configured_voice_tools()) == [
        "manage_issue_draft",
        "get_current_datetime",
    ]


def test_warmup_schemas_match_runtime_tool_order(monkeypatch):
    monkeypatch.setattr(registry, "KNOWLEDGE_ENABLED", True)
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")

    schemas = registry.configured_openai_tool_schemas()

    assert [schema["function"]["name"] for schema in schemas] == _names(
        registry.configured_voice_tools()
    )
