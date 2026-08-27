import pytest

from core.rag_config import _validate_voice_rag_timeouts


def test_voice_rag_timeout_requires_recovery_window_after_filler():
    _validate_voice_rag_timeouts(0.9, 2.5)

    with pytest.raises(ValueError, match="strictly less"):
        _validate_voice_rag_timeouts(0.9, 0.9)
