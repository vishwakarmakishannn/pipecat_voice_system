import json

from scripts.generate_safe_baseline import (
    SAFE_CONFIGURATION_KEYS,
    configuration_hash,
    safe_configuration,
)


def test_safe_baseline_configuration_never_reads_secret_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "must-not-appear")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-appear-either")
    monkeypatch.setenv("GROQ_MODEL", "safe-model-id")

    payload = safe_configuration()
    serialized = json.dumps(payload)

    assert "GROQ_API_KEY" not in SAFE_CONFIGURATION_KEYS
    assert "AWS_SECRET_ACCESS_KEY" not in SAFE_CONFIGURATION_KEYS
    assert "must-not-appear" not in serialized
    assert payload["GROQ_MODEL"] == "safe-model-id"


def test_configuration_hash_is_order_independent():
    assert configuration_hash({"b": "2", "a": "1"}) == configuration_hash(
        {"a": "1", "b": "2"}
    )
