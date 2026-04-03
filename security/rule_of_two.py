"""Rule of Two — operator approval gate for high-risk tool calls.

High-risk actions (destructive kubectl, email sends, mesh dispatches) require
explicit operator approval via Telegram inline buttons before execution.

Flow:
  1. execute_tool() calls requires_approval(tool, args)
  2. If True: request_approval() inserts pending_actions row + sends Telegram buttons
  3. Tool loop polls check_approval() every 2s (up to 120s)
  4. Callback handler resolves via resolve_approval()
  5. Tool proceeds or returns rejection message
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("nexus.security")

# ---------------------------------------------------------------------------
# High-risk tool patterns
# ---------------------------------------------------------------------------

# For "bash": list of regex patterns matched against the command string.
# For other tools: True means ALL calls require approval.
HIGH_RISK_TOOLS: dict[str, list[str] | bool] = {
    "bash": [
        r"kubectl\s+(apply|delete|drain|cordon|scale|patch|rollout\s+undo)",
        r"systemctl\s+(stop|disable|mask)",
        r"helm\s+(install|upgrade|uninstall|delete)",
        r"docker\s+(rm|rmi|kill|stop|system\s+prune)",
        r"rm\s+-rf\s+/(?!tmp)",
        r"(?i)DROP\s+(TABLE|DATABASE)",
    ],
    "send_email": True,
    "claw_dispatch": True,
    "write_file": [r"^/etc/", r"^/usr/", r"^/bin/", r"^/sbin/"],
    "edit_file": [r"^/etc/", r"^/usr/", r"^/bin/", r"^/sbin/"],
}

# Approval timeout (seconds)
APPROVAL_TIMEOUT = 120
POLL_INTERVAL = 2

# Compiled regex cache
_compiled: dict[str, list[re.Pattern]] = {}


def _get_patterns(tool_name: str) -> list[re.Pattern]:
    """Get compiled regex patterns for a tool."""
    if tool_name not in _compiled:
        raw = HIGH_RISK_TOOLS.get(tool_name)
        if isinstance(raw, list):
            _compiled[tool_name] = [re.compile(p) for p in raw]
        else:
            _compiled[tool_name] = []
    return _compiled[tool_name]


def _get_check_value(tool_name: str, args: dict) -> str:
    """Get the string to match against patterns for a given tool."""
    if tool_name == "bash":
        return args.get("command", "")
    if tool_name in ("write_file", "edit_file"):
        return args.get("file_path", "")
    return ""


def requires_approval(tool_name: str, args: dict) -> bool:
    """Check if a tool call requires operator approval."""
    spec = HIGH_RISK_TOOLS.get(tool_name)
    if spec is None:
        return False

    # All calls to this tool require approval
    if spec is True:
        return True

    # Pattern-based check
    value = _get_check_value(tool_name, args)
    for pattern in _get_patterns(tool_name):
        if pattern.search(value):
            return True

    return False


def _describe_action(tool_name: str, args: dict) -> str:
    """Generate a human-readable description of the pending action."""
    if tool_name == "bash":
        cmd = args.get("command", "")
        if len(cmd) > 200:
            cmd = cmd[:200] + "..."
        return f"bash: {cmd}"
    elif tool_name == "send_email":
        to = args.get("to", "?")
        subject = args.get("subject", "?")
        return f"send_email to {to}: {subject}"
    elif tool_name == "claw_dispatch":
        target = args.get("target", "?")
        task = args.get("task", "?")[:100]
        return f"claw_dispatch to {target}: {task}"
    elif tool_name in ("write_file", "edit_file"):
        path = args.get("file_path", "?")
        return f"{tool_name}: {path}"
    else:
        return f"{tool_name}: {json.dumps(args, default=str)[:200]}"


# ---------------------------------------------------------------------------
# Pending actions — SQLite operations
# ---------------------------------------------------------------------------

def _connect():
    from db import _connect
    return _connect()


def create_pending_action(
    session_id: Optional[str],
    channel: Optional[str],
    tool_name: str,
    args: dict,
) -> int:
    """Insert a pending action and return its ID."""
    now = datetime.now(timezone.utc).isoformat()
    description = _describe_action(tool_name, args)
    con = _connect()
    cur = con.execute(
        """INSERT INTO pending_actions
           (session_id, channel, tool_name, tool_args_json, description, status, requested_at)
           VALUES (?, ?, ?, ?, ?, 'pending', ?)""",
        (session_id, channel, tool_name, json.dumps(args, default=str), description, now),
    )
    action_id = cur.lastrowid
    con.commit()
    con.close()
    return action_id


def check_approval(action_id: int) -> Optional[str]:
    """Check the status of a pending action. Returns status or None if still pending."""
    con = _connect()
    row = con.execute(
        "SELECT status FROM pending_actions WHERE id = ?", (action_id,)
    ).fetchone()
    con.close()
    if row is None:
        return "rejected"  # Missing record = reject
    status = row[0]
    return status if status != "pending" else None


def resolve_approval(action_id: int, decision: str, operator_note: Optional[str] = None):
    """Resolve a pending action (called by Telegram callback handler)."""
    now = datetime.now(timezone.utc).isoformat()
    con = _connect()
    con.execute(
        """UPDATE pending_actions
           SET status = ?, resolved_at = ?, resolved_by = 'operator', operator_note = ?
           WHERE id = ? AND status = 'pending'""",
        (decision, now, operator_note, action_id),
    )
    con.commit()
    con.close()
    log.info("[rule_of_two] Action %d resolved: %s", action_id, decision)


def expire_stale_actions(max_age_seconds: int = 300):
    """Mark pending actions older than max_age as expired. Called periodically."""
    con = _connect()
    con.execute(
        """UPDATE pending_actions SET status = 'expired'
           WHERE status = 'pending'
           AND datetime(requested_at) < datetime('now', ?)""",
        (f"-{max_age_seconds} seconds",),
    )
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# Telegram approval request
# ---------------------------------------------------------------------------

def send_approval_request_sync(action_id: int, description: str):
    """Send a Telegram message with Approve/Reject inline buttons (sync, raw HTTP)."""
    import urllib.parse
    import urllib.request

    from config import BOT_TOKEN, AUTHORIZED_USER_ID

    text = (
        f"Rule of Two -- approval required\n\n"
        f"{description}\n\n"
        f"Action ID: {action_id}"
    )

    reply_markup = json.dumps({
        "inline_keyboard": [[
            {"text": "Approve", "callback_data": f"r2:approve:{action_id}"},
            {"text": "Reject", "callback_data": f"r2:reject:{action_id}"},
        ]]
    })

    data = urllib.parse.urlencode({
        "chat_id": str(AUTHORIZED_USER_ID),
        "text": text,
        "reply_markup": reply_markup,
    }).encode()

    req = urllib.request.Request(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", data=data
    )
    try:
        urllib.request.urlopen(req, timeout=15)
    except Exception as e:
        log.error("[rule_of_two] Failed to send approval request: %s", e)


# ---------------------------------------------------------------------------
# Blocking approval wait (called from tool execution thread)
# ---------------------------------------------------------------------------

def wait_for_approval(action_id: int, timeout: int = APPROVAL_TIMEOUT) -> str:
    """Block until the action is approved, rejected, or times out.

    Returns: "approved", "rejected", or "timeout"
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = check_approval(action_id)
        if status is not None:
            return status
        time.sleep(POLL_INTERVAL)
    # Timed out — mark as expired
    resolve_approval(action_id, "expired")
    return "timeout"
