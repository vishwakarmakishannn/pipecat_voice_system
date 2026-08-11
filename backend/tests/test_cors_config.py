import pytest

from core.cors_config import configure_pipecat_allowed_origins, parse_allowed_origins


def test_pipecat_cors_uses_explicit_application_allow_list(monkeypatch):
    monkeypatch.setenv(
        "ALLOWED_ORIGINS",
        "https://voice.example.com, http://localhost:5173/",
    )
    monkeypatch.setenv("PIPECAT_ALLOWED_ORIGINS", "*")

    origins = configure_pipecat_allowed_origins()

    assert origins == ["https://voice.example.com", "http://localhost:5173"]
    assert (
        __import__("os").environ["PIPECAT_ALLOWED_ORIGINS"]
        == "https://voice.example.com,http://localhost:5173"
    )
    assert "https://evil.example" not in origins


@pytest.mark.parametrize(
    "raw",
    ["*", "javascript:alert(1)", "https://example.com/path", " , "],
)
def test_invalid_cors_origins_fail_closed(raw):
    with pytest.raises(ValueError):
        parse_allowed_origins(raw)
