"""Two-tier heartbeat pipeline — gather, evaluate, act.

Replaces the old single-tier HeartbeatObserver with two schedule variants:
  - HeartbeatBusinessObserver: every 30 min during business hours (Mon-Fri 8-18)
  - HeartbeatOvernightObserver: every 2 hours outside business hours

Flow: gather_all() -> evaluate_signals() -> dispatch() -> Telegram + journal
Cost: gather is free (HTTP/SSH), evaluate uses haiku, act uses sonnet only if needed.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from observers.base import Observer, ObserverContext, ObserverResult

log = logging.getLogger("nexus")

PROACTIVITY = os.environ.get("HEARTBEAT_PROACTIVITY", "advisor")
JOURNAL_DIR = Path(os.environ.get("MEMORY_DIR", "/data/memory")) / "journal"


class _HeartbeatBase(Observer):
    """Shared heartbeat logic for both schedule variants."""

    def _write_journal(self, severity: int, summary: str, gathered: dict, assessment: dict):
        """Append heartbeat run to today's journal."""
        JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        journal_file = JOURNAL_DIR / f"{today}.md"

        timestamp = datetime.now(timezone.utc).strftime("%H:%M UTC")
        entry = f"\n### Heartbeat [{timestamp}] severity={severity}\n{summary}\n"

        # Add key metrics
        for source in ("k3s", "ceph", "services", "prometheus", "gmail"):
            data = gathered.get(source, {})
            if isinstance(data, dict) and "error" not in data:
                if source == "k3s":
                    entry += f"- K3s: {data.get('healthy', '?')}/{data.get('total_pods', '?')} pods healthy\n"
                elif source == "ceph":
                    entry += f"- Ceph: {data.get('status', '?')}, {data.get('used_pct', '?')}% used\n"
                elif source == "services":
                    entry += f"- Services: {data.get('online', '?')}/{data.get('total', '?')} online\n"
                elif source == "prometheus":
                    entry += f"- Prometheus: {data.get('down_count', '?')} down, {data.get('alert_count', '?')} alerts\n"
                elif source == "gmail":
                    entry += f"- Email: {data.get('total_unread', '?')} unread\n"

        with open(journal_file, "a") as f:
            f.write(entry)

    def run(self, ctx: ObserverContext) -> ObserverResult:
        """Execute the gather -> evaluate -> act pipeline."""
        from heartbeat.gather import gather_all
        from heartbeat.evaluate import evaluate_signals
        from heartbeat.act import dispatch

        # Phase 1: Gather (free — no LLM)
        try:
            gathered = asyncio.run(gather_all())
        except Exception as e:
            log.error("[heartbeat] Gather failed: %s", e)
            return ObserverResult(success=False, error=f"Gather failed: {e}")

        # Phase 2: Evaluate (haiku — cheap)
        assessment = evaluate_signals(gathered)
        severity = assessment.get("severity", 0)
        summary = assessment.get("summary", "No summary")

        log.info("[heartbeat] Severity=%d: %s", severity, summary)

        # Prometheus gauge
        try:
            from metrics import set_gauge
            set_gauge("nexus_heartbeat_severity", value=severity)
        except Exception:
            pass

        # Journal every run
        self._write_journal(severity, summary, gathered, assessment)

        # Phase 3: Act (sonnet — only if severity warrants it)
        action_result = dispatch(gathered, assessment, proactivity=PROACTIVITY)

        if action_result.get("notified"):
            self.send_telegram(action_result["message"])

        if action_result.get("actions_taken"):
            log.info("[heartbeat] Actions taken: %s", action_result["actions_taken"])

        return ObserverResult(
            success=True,
            message=summary,
            data={
                "severity": severity,
                "gathered_ms": gathered.get("duration_ms", 0),
                "proactivity": PROACTIVITY,
                "notified": action_result.get("notified", False),
            },
        )


class HeartbeatBusinessObserver(_HeartbeatBase):
    """Every 30 min during business hours (Mon-Fri 8-18 UTC)."""
    name = "heartbeat_business"
    schedule = "*/30 8-18 * * 1-5"


class HeartbeatOvernightObserver(_HeartbeatBase):
    """Every 2 hours outside business hours."""
    name = "heartbeat_overnight"
    schedule = "0 0,2,4,6,20,22 * * *"
