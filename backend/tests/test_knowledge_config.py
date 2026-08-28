import os
import subprocess
import sys


def _import_config(**environment):
    values = os.environ.copy()
    values.update(environment)
    return subprocess.run(
        [sys.executable, "-c", "import core.knowledge_config"],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=values,
        text=True,
        capture_output=True,
        check=False,
    )


def test_knowledge_feature_is_disabled_by_default():
    values = os.environ.copy()
    values.pop("MSWIPE_KNOWLEDGE_ENABLED", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from core.knowledge_config import KNOWLEDGE_ENABLED; assert KNOWLEDGE_ENABLED is False",
        ],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=values,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_invalid_boolean_fails_closed_at_startup():
    result = _import_config(MSWIPE_KNOWLEDGE_ENABLED="sometimes")
    assert result.returncode != 0
    assert "must be a boolean" in result.stderr


def test_unsafe_empty_domain_allowlist_is_rejected():
    result = _import_config(MSWIPE_KNOWLEDGE_ALLOWED_DOMAINS="")
    assert result.returncode != 0
    assert "must not be empty" in result.stderr
