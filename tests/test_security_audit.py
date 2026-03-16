"""Tests for security.audit — audit logging, redaction-before-log, fire-and-forget."""

import os
import sys
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

with patch.dict("os.environ", {
    "TELEGRAM_BOT_TOKEN": "fake:token",
    "AUTHORIZED_USER_ID": "12345",
}):
    from security.policy import _parse_policy
    from security.audit import (
        log_tool_execution,
        log_llm_call,
        log_observer_run,
        log_policy_violation,
        cleanup_old_records,
        _hash,
        _truncate,
    )
    from security.redact import invalidate_cache
    import security.policy as _pol

_AUDIT_POLICY = _parse_policy({
    "version": 1,
    "credentials": {
        "redact_patterns": ["sk-[a-zA-Z0-9_-]{20,}"],
        "redact_env_vars": [],
    },
    "audit": {
        "enabled": True,
        "log_tool_args": True,
        "log_result_preview": True,
        "retention_days": 90,
    },
})


@pytest.fixture(autouse=True)
def _set_policy():
    saved = _pol._current_policy
    _pol._current_policy = _AUDIT_POLICY
    invalidate_cache()
    yield
    _pol._current_policy = saved
    invalidate_cache()


@pytest.fixture
def audit_db(tmp_path, monkeypatch):
    """Create a temp DB with audit_log table."""
    db_path = tmp_path / "test_audit.db"
    monkeypatch.setattr("db.DB_PATH", db_path)
    from db import init_db
    init_db()
    return db_path


def _count_records(db_path, event_type=None):
    con = sqlite3.connect(str(db_path))
    if event_type:
        count = con.execute("SELECT COUNT(*) FROM audit_log WHERE event_type = ?", (event_type,)).fetchone()[0]
    else:
        count = con.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    con.close()
    return count


def _get_records(db_path, event_type=None):
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    if event_type:
        rows = con.execute("SELECT * FROM audit_log WHERE event_type = ?", (event_type,)).fetchall()
    else:
        rows = con.execute("SELECT * FROM audit_log").fetchall()
    con.close()
    return [dict(r) for r in rows]


class TestToolAudit:

    def test_log_tool_execution(self, audit_db):
        log_tool_execution("sess1", "telegram", "bash", {"command": "ls"}, "output", 100)
        assert _count_records(audit_db, "tool_execution") == 1

    def test_log_includes_args_hash(self, audit_db):
        log_tool_execution("sess1", "telegram", "read_file", {"file_path": "/data/x"}, "content", 50)
        records = _get_records(audit_db, "tool_execution")
        assert records[0]["tool_args_hash"] is not None
        assert len(records[0]["tool_args_hash"]) == 16  # truncated SHA256

    def test_result_preview_is_redacted(self, audit_db):
        secret_result = "Found key: sk-abcdefghijklmnopqrstuvwxyz1234567890 in file"
        log_tool_execution("sess1", "telegram", "read_file", {"file_path": "/x"}, secret_result, 50)
        records = _get_records(audit_db, "tool_execution")
        preview = records[0]["result_preview"]
        assert "sk-" not in preview
        assert "[REDACTED]" in preview

    def test_result_preview_truncated(self, audit_db):
        long_result = "x" * 500
        log_tool_execution("sess1", "telegram", "bash", {"command": "cat"}, long_result, 50)
        records = _get_records(audit_db, "tool_execution")
        assert len(records[0]["result_preview"]) <= 203  # 200 + "..."


class TestLLMAudit:

    def test_log_llm_call(self, audit_db):
        log_llm_call("sess1", "telegram", "anthropic_api", "claude-sonnet-4-6", 1500, 2000)
        assert _count_records(audit_db, "llm_call") == 1

    def test_llm_call_fields(self, audit_db):
        log_llm_call("sess1", "discord", "vllm", "nemotron-3-super", 3000, 5000)
        records = _get_records(audit_db, "llm_call")
        assert records[0]["backend"] == "vllm"
        assert records[0]["model"] == "nemotron-3-super"
        assert records[0]["token_count"] == 3000
        assert records[0]["duration_ms"] == 5000


class TestObserverAudit:

    def test_log_observer_success(self, audit_db):
        log_observer_run("EmailDigestObserver", True, 15000)
        records = _get_records(audit_db, "observer_run")
        assert records[0]["policy_decision"] == "success"
        assert records[0]["observer_name"] == "EmailDigestObserver"

    def test_log_observer_failure(self, audit_db):
        log_observer_run("CyberThreatFeedObserver", False, 5000, error="Connection timeout")
        records = _get_records(audit_db, "observer_run")
        assert records[0]["policy_decision"] == "error"
        assert "timeout" in records[0]["metadata_json"].lower()


class TestPolicyViolation:

    def test_log_violation(self, audit_db):
        log_policy_violation("sess1", "telegram", "web_fetch",
                             {"url": "http://192.168.1.1/"}, "block_private_ranges",
                             "IP 192.168.1.1 is in private range")
        records = _get_records(audit_db, "policy_violation")
        assert records[0]["policy_decision"] == "deny"
        assert records[0]["policy_rule"] == "block_private_ranges"


class TestFireAndForget:

    def test_db_error_does_not_raise(self):
        """Audit functions must never raise, even if DB is broken."""
        with patch("security.audit._write_audit_record", side_effect=Exception("DB broke")):
            # These should NOT raise
            log_tool_execution("s", "c", "bash", {}, "r", 0)
            log_llm_call("s", "c", "b", "m", 0, 0)
            log_observer_run("obs", True, 0)
            log_policy_violation("s", "c", "t", {}, "r", "reason")

    def test_disabled_audit_skips(self, audit_db):
        import security.policy as _pol
        saved = _pol._current_policy
        _pol._current_policy = _parse_policy({
            "version": 1,
            "audit": {"enabled": False},
        })
        log_tool_execution("sess1", "telegram", "bash", {}, "output", 100)
        assert _count_records(audit_db) == 0
        _pol._current_policy = saved


class TestHelpers:

    def test_hash_consistent(self):
        assert _hash("test") == _hash("test")
        assert _hash("a") != _hash("b")
        assert len(_hash("test")) == 16

    def test_truncate(self):
        assert _truncate("short") == "short"
        assert _truncate("x" * 300) == "x" * 200 + "..."
        assert len(_truncate("x" * 300)) == 203
