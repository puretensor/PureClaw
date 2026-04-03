"""Audit trail — structured logging of all agent actions.

ISO 27001 A.12.4 compliance: every tool execution, LLM call, and observer run
is recorded. All functions are fire-and-forget (catch exceptions, never break
the caller). Redaction is applied before logging.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("nexus.security")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(value: str) -> str:
    """SHA256 hash for forensic correlation without content leakage."""
    return hashlib.sha256(value.encode(errors="replace")).hexdigest()[:16]


def _safe_json(obj) -> str:
    """JSON-serialize, falling back to str() on failure."""
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        return str(obj)


def _truncate(text: str, limit: int = 200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


# ---------------------------------------------------------------------------
# Core audit functions
# ---------------------------------------------------------------------------

def log_tool_execution(
    session_id: Optional[str],
    channel: Optional[str],
    tool_name: str,
    args: dict,
    result: str,
    duration_ms: int,
    policy_decision: str = "allow",
    policy_rule: Optional[str] = None,
) -> None:
    """Record a tool execution in the audit log."""
    try:
        from security.policy import get_policy
        policy = get_policy()
        if not policy.audit.enabled:
            return

        from security.redact import redact_text
        result_preview = _truncate(redact_text(result)) if policy.audit.log_result_preview else None
        args_hash = _hash(_safe_json(args)) if policy.audit.log_tool_args else None

        _write_audit_record(
            event_type="tool_execution",
            session_id=session_id,
            channel=channel,
            tool_name=tool_name,
            tool_args_hash=args_hash,
            result_hash=_hash(result),
            result_preview=result_preview,
            policy_decision=policy_decision,
            policy_rule=policy_rule,
            duration_ms=duration_ms,
        )
    except Exception as e:
        log.debug("Audit log_tool_execution error (non-fatal): %s", e)


def log_llm_call(
    session_id: Optional[str],
    channel: Optional[str],
    backend: str,
    model: str,
    token_count: Optional[int],
    duration_ms: int,
) -> None:
    """Record an LLM API call in the audit log."""
    try:
        from security.policy import get_policy
        if not get_policy().audit.enabled:
            return

        _write_audit_record(
            event_type="llm_call",
            session_id=session_id,
            channel=channel,
            backend=backend,
            model=model,
            token_count=token_count,
            duration_ms=duration_ms,
        )
    except Exception as e:
        log.debug("Audit log_llm_call error (non-fatal): %s", e)


def log_observer_run(
    observer_name: str,
    success: bool,
    duration_ms: int,
    error: Optional[str] = None,
) -> None:
    """Record an observer execution in the audit log."""
    try:
        from security.policy import get_policy
        if not get_policy().audit.enabled:
            return

        _write_audit_record(
            event_type="observer_run",
            observer_name=observer_name,
            policy_decision="success" if success else "error",
            duration_ms=duration_ms,
            metadata_json=_safe_json({"error": error}) if error else None,
        )
    except Exception as e:
        log.debug("Audit log_observer_run error (non-fatal): %s", e)

    # Prometheus metrics
    try:
        from metrics import inc
        inc("nexus_observer_runs_total", {"observer_name": observer_name, "status": "success" if success else "error"})
    except Exception:
        pass


def log_policy_violation(
    session_id: Optional[str],
    channel: Optional[str],
    tool_name: str,
    args: dict,
    policy_rule: str,
    reason: str,
) -> None:
    """Record a policy violation (denied action) in the audit log."""
    try:
        from security.policy import get_policy
        if not get_policy().audit.enabled:
            return

        _write_audit_record(
            event_type="policy_violation",
            session_id=session_id,
            channel=channel,
            tool_name=tool_name,
            tool_args_hash=_hash(_safe_json(args)),
            policy_decision="deny",
            policy_rule=policy_rule,
            metadata_json=_safe_json({"reason": reason}),
        )
    except Exception as e:
        log.debug("Audit log_policy_violation error (non-fatal): %s", e)


# ---------------------------------------------------------------------------
# Database writer
# ---------------------------------------------------------------------------

def _write_audit_record(**kwargs) -> None:
    """Insert a record into the audit_log table."""
    try:
        from db import _connect
        con = _connect()
        cols = [
            "timestamp", "event_type", "session_id", "channel", "backend",
            "tool_name", "tool_args_hash", "result_hash", "result_preview",
            "model", "token_count", "observer_name",
            "policy_decision", "policy_rule", "duration_ms", "metadata_json",
        ]
        values = {c: kwargs.get(c) for c in cols}
        values["timestamp"] = _now()

        present = {k: v for k, v in values.items() if v is not None}
        col_names = ", ".join(present.keys())
        placeholders = ", ".join("?" for _ in present)
        con.execute(
            f"INSERT INTO audit_log ({col_names}) VALUES ({placeholders})",
            tuple(present.values()),
        )
        con.commit()
        con.close()
    except Exception as e:
        log.debug("Audit DB write error (non-fatal): %s", e)


# ---------------------------------------------------------------------------
# Retention cleanup
# ---------------------------------------------------------------------------

def cleanup_old_records() -> int:
    """Delete audit records older than retention_days. Returns count deleted."""
    try:
        from security.policy import get_policy
        policy = get_policy()
        if policy.audit.retention_days <= 0:
            return 0

        from db import _connect
        con = _connect()
        cutoff = f"datetime('now', '-{policy.audit.retention_days} days')"
        cur = con.execute(f"DELETE FROM audit_log WHERE timestamp < {cutoff}")
        deleted = cur.rowcount
        con.commit()
        con.close()
        if deleted > 0:
            log.info("Audit cleanup: deleted %d records older than %d days", deleted, policy.audit.retention_days)
        return deleted
    except Exception as e:
        log.debug("Audit cleanup error (non-fatal): %s", e)
        return 0
