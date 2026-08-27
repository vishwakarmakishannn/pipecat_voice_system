from core.log_safety import safe_text_metadata


def test_safe_text_metadata_is_stable_without_exposing_content():
    secret = "Rohan rohan@example.com 9876543210"

    first = safe_text_metadata(secret)
    second = safe_text_metadata(secret)

    assert first == second
    assert "chars=" in first
    assert "words=3" in first
    assert "Rohan" not in first
    assert "example.com" not in first
    assert "9876543210" not in first
