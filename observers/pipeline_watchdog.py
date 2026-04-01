#!/usr/bin/env python3
"""Pipeline health watchdog observer.

Runs every 6 hours. Checks that critical pipelines are alive and producing
output. Sends a Telegram alert if any pipeline appears stalled or dead.

All freshness checks SSH to tensor-core directly, avoiding dependency on
the /sync/ mount which may be stale.

Checks:
  1. CC reports: recent session reports on TC
  2. voice-kb ingest: service running on TC, output files recent
  3. daily report: branded PDF compiled recently
  4. vLLM: responding to health checks
  5. observer state: state files present and recently updated
"""

import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from observers.base import Observer, ObserverResult

log = logging.getLogger("nexus")

# Retry config for transient network failures
MAX_RETRIES = 3
RETRY_DELAY_SECS = 5
# Skip watchdog checks for this many seconds after pod start
STARTUP_GRACE_SECS = 600  # 10 minutes

_pod_start_time = time.monotonic()

# Thresholds
CC_STALE_MINUTES = 1440        # Alert if no new CC report in 24h (sessions are intermittent)
VOICE_KB_STALE_HOURS = 24      # Alert if no new voice-kb output in this many hours
DAILY_REPORT_MAX_AGE_HOURS = 36  # Alert if last daily report PDF older than this

# Remote paths on tensor-core (checked via SSH)
TC_CC_REPORTS = "~/reports/cc"
TC_VOICE_KB = "~/voice-kb/kb"
TC_DAILY_REPORTS = "~/reports/daily"

# TC Tailscale IP for SSH checks
TC_HOST = os.environ.get("TC_SSH_HOST", "localhost")
_vllm_env = os.environ.get("VLLM_URL", "http://localhost:5000/health")
# Derive health URL: strip /v1 path if present, append /health
if _vllm_env.rstrip("/").endswith("/v1"):
    VLLM_URL = _vllm_env.rstrip("/").rsplit("/v1", 1)[0] + "/health"
else:
    VLLM_URL = _vllm_env

# Observer state directory (K8s PVC or local fallback)
OBSERVER_STATE_DIR = Path(os.environ.get(
    "OBSERVER_STATE_DIR",
    str(Path(__file__).parent / ".state"),
))


class PipelineWatchdog(Observer):
    """Monitors critical pipeline health and alerts on failures."""

    name = "pipeline_watchdog"
    schedule = "0 */6 * * *"  # Every 6 hours

    def run(self, ctx=None) -> ObserverResult:
        # Startup grace period -- skip remote checks if pod just started
        uptime = time.monotonic() - _pod_start_time
        if uptime < STARTUP_GRACE_SECS:
            log.info("Pipeline watchdog: skipping (pod uptime %.0fs < %ds grace)",
                     uptime, STARTUP_GRACE_SECS)
            return ObserverResult(
                success=True, message="",
                data={"skipped": True, "reason": "startup_grace"},
            )

        now = self.now_utc()
        alerts = []
        healthy = []

        # 1. Check CC report freshness on TC
        self._check_cc_freshness(now, alerts, healthy)

        # 2. Check voice-kb ingest service and output freshness on TC
        self._check_voice_kb(now, alerts, healthy)

        # 3. Check daily report PDF recency on TC
        self._check_daily_report(now, alerts, healthy)

        # 4. Check vLLM health
        self._check_vllm(alerts, healthy)

        # 5. Check observer state directory
        self._check_observer_health(now, alerts, healthy)

        # Build result
        if alerts:
            alert_text = (
                f"PIPELINE WATCHDOG \u2014 {len(alerts)} ALERT(S)\n\n"
                + "\n".join(f"\u26a0\ufe0f {a}" for a in alerts)
            )
            if healthy:
                alert_text += "\n\n" + "\n".join(f"\u2705 {h}" for h in healthy)
            self.send_telegram(alert_text)
            return ObserverResult(
                success=True,
                message=alert_text,
                data={"alerts": alerts, "healthy": healthy},
            )

        # All healthy -- silent success (don't spam Telegram)
        log.info("Pipeline watchdog: all %d checks healthy", len(healthy))
        return ObserverResult(
            success=True,
            message="",  # Empty = silent
            data={"alerts": [], "healthy": healthy},
        )

    # -- SSH helpers --------------------------------------------------------

    def _ssh_cmd(self, cmd: str, timeout: int = 15) -> subprocess.CompletedProcess | None:
        """Run a command on TC via SSH. Returns CompletedProcess or None on failure."""
        try:
            return subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
                 f"puretensorai@{TC_HOST}", cmd],
                capture_output=True, text=True, timeout=timeout,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None

    def _ssh_newest_file_age(self, remote_dir: str, glob_pattern: str = "*.md") -> float | None:
        """Get age in seconds of the newest file matching a pattern on TC.

        Returns age in seconds, or None on SSH/parse failure.
        """
        # ls -t sorts by mtime descending; stat gets epoch mtime of the first result
        cmd = (
            f"f=$(ls -t {remote_dir}/{glob_pattern} 2>/dev/null | head -1) && "
            f"[ -n \"$f\" ] && stat -c '%Y' \"$f\""
        )
        result = self._ssh_cmd(cmd)
        if result is None or result.returncode != 0 or not result.stdout.strip():
            return None
        try:
            mtime = int(result.stdout.strip())
            return time.time() - mtime
        except ValueError:
            return None

    def _ssh_service_active(self, service: str) -> bool | None:
        """Check if a systemd service is active on TC via SSH.

        Returns True/False for definitive results, None for transient failures.
        """
        result = self._ssh_cmd(f"systemctl is-active {service} 2>/dev/null")
        if result is None:
            return None
        return result.stdout.strip() == "active"

    # -- Individual checks --------------------------------------------------

    def _check_cc_freshness(self, now, alerts, healthy):
        """Check that CC session reports are being generated on TC."""
        age_secs = None
        for attempt in range(MAX_RETRIES):
            age_secs = self._ssh_newest_file_age(TC_CC_REPORTS, "*.md")
            if age_secs is not None:
                break
            if attempt < MAX_RETRIES - 1:
                log.info("CC freshness SSH check failed (attempt %d/%d), retrying in %ds",
                         attempt + 1, MAX_RETRIES, RETRY_DELAY_SECS)
                time.sleep(RETRY_DELAY_SECS)

        if age_secs is None:
            alerts.append(
                f"CC reports: SSH to tensor-core failed after {MAX_RETRIES} attempts "
                f"or no reports found in {TC_CC_REPORTS}/"
            )
            return

        age_min = age_secs / 60
        if age_min > CC_STALE_MINUTES:
            alerts.append(
                f"CC reports stale: newest report is {age_min:.0f} min old "
                f"(threshold: {CC_STALE_MINUTES} min)"
            )
        else:
            healthy.append(f"CC reports: fresh ({age_min:.0f} min ago)")

    def _check_voice_kb(self, now, alerts, healthy):
        """Check voice-kb ingest service and output freshness on TC (with retries)."""
        service_active = None
        for attempt in range(MAX_RETRIES):
            service_active = self._ssh_service_active("voice-kb-ingest")
            if service_active is not None:
                break
            if attempt < MAX_RETRIES - 1:
                log.info("voice-kb SSH check failed (attempt %d/%d), retrying in %ds",
                         attempt + 1, MAX_RETRIES, RETRY_DELAY_SECS)
                time.sleep(RETRY_DELAY_SECS)

        if service_active is None:
            alerts.append(
                "voice-kb check: SSH to tensor-core failed after "
                f"{MAX_RETRIES} attempts"
            )
            return

        if not service_active:
            alerts.append(
                "voice-kb-ingest service NOT running on tensor-core \u2014 "
                "pipeline is dead, new voice memos will not be processed"
            )
            return

        # Check output freshness via SSH to TC
        age_secs = self._ssh_newest_file_age(TC_VOICE_KB, "*.md")
        if age_secs is not None:
            age_hrs = age_secs / 3600
            if age_hrs > VOICE_KB_STALE_HOURS:
                alerts.append(
                    f"voice-kb output stale: newest memo is {age_hrs:.1f}h old "
                    f"(threshold: {VOICE_KB_STALE_HOURS}h) \u2014 "
                    f"service may be running but not producing output"
                )
            else:
                healthy.append(
                    f"voice-kb ingest: active, output {age_hrs:.1f}h ago"
                )
        else:
            healthy.append("voice-kb ingest: service active (could not check output freshness)")

    def _check_daily_report(self, now, alerts, healthy):
        """Check that daily branded PDF reports are being generated on TC."""
        age_secs = None
        for attempt in range(MAX_RETRIES):
            age_secs = self._ssh_newest_file_age(
                TC_DAILY_REPORTS, "PureTensor_Daily_Report_*.pdf"
            )
            if age_secs is not None:
                break
            if attempt < MAX_RETRIES - 1:
                log.info("Daily report SSH check failed (attempt %d/%d), retrying in %ds",
                         attempt + 1, MAX_RETRIES, RETRY_DELAY_SECS)
                time.sleep(RETRY_DELAY_SECS)

        if age_secs is None:
            alerts.append(
                f"Daily report: SSH to tensor-core failed after {MAX_RETRIES} attempts "
                f"or no PDFs found in {TC_DAILY_REPORTS}/"
            )
            return

        age_hrs = age_secs / 3600
        if age_hrs > DAILY_REPORT_MAX_AGE_HOURS:
            alerts.append(
                f"Daily report stale: newest PDF is {age_hrs:.0f}h old "
                f"(threshold: {DAILY_REPORT_MAX_AGE_HOURS}h)"
            )
        else:
            healthy.append(f"Daily report: PDF generated {age_hrs:.1f}h ago")

    def _curl_health(self, url: str) -> str | None:
        """Curl a health endpoint. Returns HTTP status code, or None on transient failure."""
        try:
            result = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                 "--connect-timeout", "5", url],
                capture_output=True, text=True, timeout=10,
            )
            status = result.stdout.strip()
            # HTTP 000 = connection failed -- treat as transient
            if status == "000":
                return None
            return status
        except (subprocess.TimeoutExpired, OSError):
            return None

    def _check_vllm(self, alerts, healthy):
        """Check vLLM is responding (with retries)."""
        status = None
        for attempt in range(MAX_RETRIES):
            status = self._curl_health(VLLM_URL)
            if status is not None:
                break
            if attempt < MAX_RETRIES - 1:
                log.info("vLLM health check failed (attempt %d/%d), retrying in %ds",
                         attempt + 1, MAX_RETRIES, RETRY_DELAY_SECS)
                time.sleep(RETRY_DELAY_SECS)

        if status is None:
            alerts.append(
                f"vLLM health check unreachable after {MAX_RETRIES} attempts "
                f"(HTTP 000 / connection refused)"
            )
        elif status == "200":
            port = VLLM_URL.split("://")[-1].split("/")[0].split(":")[-1]
            healthy.append(f"vLLM: healthy (port {port})")
        else:
            alerts.append(f"vLLM health check returned HTTP {status}")

    def _check_observer_health(self, now, alerts, healthy):
        """Check observer state directory for signs of life."""
        try:
            if OBSERVER_STATE_DIR.exists():
                state_files = list(OBSERVER_STATE_DIR.glob("*.json"))
                if state_files:
                    newest = max(state_files, key=lambda p: p.stat().st_mtime)
                    age_hrs = (now.timestamp() - newest.stat().st_mtime) / 3600
                    healthy.append(
                        f"Observer state: {len(state_files)} files, "
                        f"last update {age_hrs:.1f}h ago ({newest.name})"
                    )
                else:
                    alerts.append("Observer state: directory exists but no state files")
            else:
                alerts.append("Observer state directory missing")
        except Exception as e:
            alerts.append(f"Observer state check failed: {e}")
