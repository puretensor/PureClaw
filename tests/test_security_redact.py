"""Tests for security.redact — pattern matching, env var redaction, history sanitization."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

with patch.dict("os.environ", {
    "TELEGRAM_BOT_TOKEN": "fake:token",
    "AUTHORIZED_USER_ID": "12345",
    "TEST_SECRET_KEY": "super_secret_value_1234567890",
}):
    from security.policy import _parse_policy
    from security.redact import redact_text, redact_args, redact_history, invalidate_cache, _get_compiled_patterns
    import security.policy as _pol

_REDACT_POLICY = _parse_policy({
    "version": 1,
    "credentials": {
        "redact_patterns": [
            "sk-[a-zA-Z0-9_-]{20,}",
            "xai-[a-zA-Z0-9_-]{20,}",
            "ghp_[a-zA-Z0-9]{36,}",
            "AIza[a-zA-Z0-9_-]{30,}",
        ],
        "redact_env_vars": ["TEST_SECRET_KEY"],
    },
})


@pytest.fixture(autouse=True)
def _set_policy():
    saved = _pol._current_policy
    _pol._current_policy = _REDACT_POLICY
    invalidate_cache()
    yield
    _pol._current_policy = saved
    invalidate_cache()


class TestRedactText:

    def test_openai_key(self):
        text = "My key is sk-abcdefghijklmnopqrstuvwxyz1234567890"
        result = redact_text(text)
        assert "sk-" not in result
        assert "[REDACTED]" in result

    def test_xai_key(self):
        text = "REDACTED_KEY"
        result = redact_text(text)
        assert "xai-" not in result
        assert "[REDACTED]" in result

    def test_github_token(self):
        text = "Token: ghp_abcdefghijklmnopqrstuvwxyz1234567890ABCD"
        result = redact_text(text)
        assert "ghp_" not in result
        assert "[REDACTED]" in result

    def test_google_key(self):
        # Construct at runtime to avoid triggering GitHub secret scanning
        text = "AIza" + "SyD1234567890abcdefghijklmnopqrstuv"
        result = redact_text(text)
        assert "AIza" not in result
        assert "[REDACTED]" in result

    def test_env_var_value(self):
        with patch.dict("os.environ", {"TEST_SECRET_KEY": "super_secret_value_1234567890"}):
            invalidate_cache()  # Force re-read of env values
            text = "The value is super_secret_value_1234567890"
            result = redact_text(text)
            assert "super_secret_value" not in result
            assert "[REDACTED]" in result

    def test_no_match(self):
        text = "Just a normal message with no secrets"
        result = redact_text(text)
        assert result == text

    def test_empty_string(self):
        assert redact_text("") == ""

    def test_multiple_secrets(self):
        text = "Key1: sk-aaaaaaaaaaaaaaaaaaaaaa Key2: xai-bbbbbbbbbbbbbbbbbbbbbb"
        result = redact_text(text)
        assert "sk-" not in result
        assert "xai-" not in result
        assert result.count("[REDACTED]") == 2


class TestRedactArgs:

    def test_simple_dict(self):
        args = {"url": "https://api.com?key=sk-abcdefghijklmnopqrstuvwxyz1234567890"}
        result = redact_args(args)
        assert "sk-" not in result["url"]
        # Original unchanged
        assert "sk-" in args["url"]

    def test_nested_dict(self):
        args = {"config": {"token": "REDACTED_KEY"}}
        result = redact_args(args)
        assert "xai-" not in result["config"]["token"]

    def test_list_values(self):
        args = {"items": ["safe", "sk-abcdefghijklmnopqrstuvwxyz1234567890"]}
        result = redact_args(args)
        assert result["items"][0] == "safe"
        assert "sk-" not in result["items"][1]

    def test_empty_dict(self):
        assert redact_args({}) == {}

    def test_non_string_values_preserved(self):
        args = {"count": 42, "flag": True, "name": "safe"}
        result = redact_args(args)
        assert result["count"] == 42
        assert result["flag"] is True


class TestRedactHistory:

    def test_string_content(self):
        messages = [
            {"role": "user", "content": "My key: sk-abcdefghijklmnopqrstuvwxyz1234567890"},
            {"role": "assistant", "content": "I see your key"},
        ]
        result = redact_history(messages)
        assert "sk-" not in result[0]["content"]
        assert result[1]["content"] == "I see your key"
        # Original unchanged
        assert "sk-" in messages[0]["content"]

    def test_block_content(self):
        """Anthropic-style content blocks."""
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Token: ghp_abcdefghijklmnopqrstuvwxyz1234567890ABCD"},
            ]},
        ]
        result = redact_history(messages)
        assert "ghp_" not in result[0]["content"][0]["text"]

    def test_empty_messages(self):
        assert redact_history([]) == []

    def test_preserves_non_content_fields(self):
        messages = [{"role": "user", "content": "test", "name": "alice"}]
        result = redact_history(messages)
        assert result[0]["name"] == "alice"
        assert result[0]["role"] == "user"


class TestCacheInvalidation:

    def test_invalidate_forces_recompile(self):
        # Get patterns (cached)
        from security.redact import _get_compiled_patterns
        patterns1 = _get_compiled_patterns()
        assert len(patterns1) > 0

        # Invalidate
        invalidate_cache()

        # Next call should recompile
        patterns2 = _get_compiled_patterns()
        assert len(patterns2) > 0
