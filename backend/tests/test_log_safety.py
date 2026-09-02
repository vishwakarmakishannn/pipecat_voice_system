from core.log_safety import redact_log_value, safe_text_metadata


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


def test_recursive_log_redaction_covers_headers_urls_bodies_and_exceptions():
    payload = {
        "headers": {
            "Authorization": "Bearer live-token",
            "x-api-key": "secret-value",
        },
        "request_url": "https://api.example.com/tickets?email=rohan@example.com#private",
        "body": {"email": "rohan@example.com", "mobile": "9876543210"},
        "exception": "failed for rohan@example.com with Bearer token-value",
    }

    safe = redact_log_value(payload)

    assert safe["headers"]["Authorization"] == "[REDACTED]"
    assert safe["headers"]["x-api-key"] == "[REDACTED]"
    assert safe["request_url"] == "https://api.example.com/tickets"
    assert safe["body"]["email"] == "[REDACTED_EMAIL]"
    assert safe["body"]["mobile"] == "[REDACTED_NUMBER]"
    assert "rohan" not in safe["exception"]
    assert "token-value" not in safe["exception"]
