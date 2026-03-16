#!/usr/bin/env python3
"""GitHub Activity Observer — realistic contribution graph maintenance.

Uses a research-backed human activity model to generate genuine commits across
multiple repos at realistic intervals. The timing algorithm is based on:

- Zero-inflated negative binomial commit distribution (not Poisson/uniform)
- Hidden Markov Model with sprint/cooldown states for burst patterns
- Day-of-week weights from GitClear 878K dev-year study (Tue-Wed peak)
- Hour-of-day weights from Software.com 250K developer study
- Monthly seasonality (Dec/Aug dips, Oct/Nov peaks)
- Random jitter so commits never land on round hours

Cron ticks every hour 07:00-23:00 UTC. Each tick consults the HMM state and
probability model to decide whether to commit (and how many). Achieves a
natural-looking 15-25 commits/week spread across 7+ repos.

Complements GitAutoSyncObserver which commits *dirty* changes — this observer
*generates new content* then commits it.
"""

import json
import logging
import math
import os
import random
import re
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from observers.base import Observer, ObserverResult

log = logging.getLogger("nexus")

# SSH to tensor-core (reused from git_auto_sync)
TC_SSH = "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 puretensorai@localhost"

# Prometheus on mon2 — set PROMETHEUS_URL in .env
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090") + "/api/v1/query"

# Target repos — path on TC, remote names
REPOS = {
    "tensor-scripts": {"github_remote": "github", "gitea_remote": "origin"},
    "voice-kb": {"github_remote": "origin", "gitea_remote": "gitea"},
    "nexus": {"github_remote": "origin", "gitea_remote": "gitea"},
    "ecommerce-agent": {"github_remote": "origin", "gitea_remote": "gitea"},
    "bookengine": {"github_remote": "origin", "gitea_remote": None},
    "echo-voicememo": {"github_remote": "origin", "gitea_remote": "gitea"},
    "arabic-qa": {"github_remote": "github", "gitea_remote": "gitea"},
    "autopen": {"github_remote": "origin", "gitea_remote": "gitea"},
}

# ── Human Activity Model ────────────────────────────────────────────────────
# Based on GitClear (878K dev-years), Software.com (250K devs), Apache study

# Day-of-week weights: Mon=0 .. Sun=6
# Tue-Wed peak, weekend low but nonzero (infra dev does weekend maintenance)
DAY_WEIGHTS = [1.0, 1.2, 1.3, 1.1, 0.8, 0.4, 0.3]

# Hour weights 0-23 UTC — infra dev profile with late-night tail
# Peak 10:00-15:00, taper into evening, occasional night ops
HOUR_WEIGHTS = [
    1, 1, 0, 0, 0, 0, 0,   # 00-06: near-zero
    1, 3, 7, 10, 14,        # 07-11: morning ramp
    15, 14, 13, 11,          # 12-15: afternoon peak
    9, 6, 4, 3,              # 16-19: evening taper
    2, 2, 1, 1,              # 20-23: late-night tail
]

# Monthly seasonality: Jan=0 .. Dec=11
# Dec holiday dip, Aug summer dip, Oct-Nov push
MONTH_WEIGHTS = [0.7, 0.9, 1.0, 1.0, 1.1, 1.0, 0.8, 0.7, 1.0, 1.1, 1.0, 0.5]

# HMM state transitions — sprint (active burst) vs cooldown (quiet period)
# Sprint lasts ~5-6 days on average (1/0.18 ≈ 5.6)
# Cooldown lasts ~2-3 days on average (1/0.4 = 2.5)
P_SPRINT_TO_COOLDOWN = 0.18
P_COOLDOWN_TO_SPRINT = 0.40

# Zero-inflation: base probability of committing at all on a given tick
P_ACTIVE_SPRINT = 0.30  # during sprint state
P_ACTIVE_COOLDOWN = 0.06  # during cooldown state

# Negative binomial parameters for commit count (given active)
# n=1, p=0.55 gives mean ~0.8, most ticks produce 1 commit, rare 2-3
NB_N = 1
NB_P = 0.55

# Hard caps
MAX_COMMITS_PER_DAY = 6
MAX_COMMITS_PER_WEEK = 25

# Secret patterns (from git_auto_sync)
BLOCK_PATTERNS = [
    r'sk-ant-api\d+-[A-Za-z0-9_-]{20,}',
    r'sk-proj-[A-Za-z0-9_-]{48,}',
    r'xai-[A-Za-z0-9_-]{40,}',
    r'AKIA[0-9A-Z]{16}',
    r'\d{10}:[A-Za-z0-9_-]{35}',
    r'-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----',
]


class GitHubActivityObserver(Observer):
    """Generates genuine commits with human-like timing patterns."""

    name = "github_activity"
    schedule = "0 7-23 * * *"  # Every hour 07:00-23:00 UTC (17 ticks/day)

    # ── SSH helper ───────────────────────────────────────────────────────────

    def _ssh_cmd(self, cmd: str, timeout: int = 60) -> tuple[int, str]:
        """Run a command on TC via SSH."""
        full_cmd = f'{TC_SSH} "{cmd}"'
        try:
            result = subprocess.run(
                full_cmd, shell=True, capture_output=True, text=True, timeout=timeout
            )
            return result.returncode, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return -1, "SSH command timed out"
        except Exception as e:
            return -1, str(e)

    # ── State management ─────────────────────────────────────────────────────

    def _state_file(self) -> Path:
        default = str(Path(__file__).parent / ".state")
        state_dir = Path(os.environ.get("OBSERVER_STATE_DIR", default))
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir / "github_activity_state.json"

    def _load_state(self) -> dict:
        sf = self._state_file()
        if sf.exists():
            try:
                return json.loads(sf.read_text())
            except Exception:
                pass
        return {
            "hmm_state": "sprint",
            "week_start": None,
            "commits_this_week": 0,
            "commits_today": 0,
            "today_date": None,
            "last_commit": None,
            "history": [],
            "total_commits": 0,
        }

    def _save_state(self, state: dict):
        state["history"] = state.get("history", [])[-100:]
        self._state_file().write_text(json.dumps(state, indent=2))

    def _get_week_start(self, now: datetime) -> str:
        monday = now - timedelta(days=now.weekday())
        return monday.strftime("%Y-%m-%d")

    # ── Human-like timing model ──────────────────────────────────────────────

    def _transition_hmm(self, state: dict) -> str:
        """Transition the sprint/cooldown HMM and return new state."""
        current = state.get("hmm_state", "sprint")
        if current == "sprint":
            new = "cooldown" if random.random() < P_SPRINT_TO_COOLDOWN else "sprint"
        else:
            new = "sprint" if random.random() < P_COOLDOWN_TO_SPRINT else "cooldown"
        state["hmm_state"] = new
        return new

    def _should_run(self, state: dict, now: datetime, force: bool = False) -> int:
        """Decide how many commits to make this tick. Returns 0 to skip.

        Uses zero-inflated negative binomial distribution modulated by:
        - HMM state (sprint vs cooldown)
        - Day-of-week weight
        - Monthly seasonality
        - Daily and weekly caps
        """
        if force:
            return force if isinstance(force, int) and force > 1 else 1

        today = now.strftime("%Y-%m-%d")
        week_start = self._get_week_start(now)

        # Reset daily counter
        if state.get("today_date") != today:
            state["today_date"] = today
            state["commits_today"] = 0

        # Reset weekly counter
        if state.get("week_start") != week_start:
            state["week_start"] = week_start
            state["commits_this_week"] = 0

        # Check caps
        if state["commits_today"] >= MAX_COMMITS_PER_DAY:
            return 0
        if state["commits_this_week"] >= MAX_COMMITS_PER_WEEK:
            return 0

        # HMM transition
        hmm = self._transition_hmm(state)

        # Base activity probability
        p_active = P_ACTIVE_SPRINT if hmm == "sprint" else P_ACTIVE_COOLDOWN

        # Modulate by day-of-week
        dow = now.weekday()  # Mon=0, Sun=6
        p_active *= DAY_WEIGHTS[dow]

        # Modulate by month seasonality
        p_active *= MONTH_WEIGHTS[now.month - 1]

        # Modulate by hour — weight the current hour
        hour = now.hour
        hour_w = HOUR_WEIGHTS[hour]
        hour_max = max(HOUR_WEIGHTS)
        p_active *= (hour_w / hour_max) if hour_max > 0 else 0

        # Clamp to [0, 0.85]
        p_active = max(0.0, min(p_active, 0.85))

        # Zero-inflation check
        roll = random.random()
        if roll >= p_active:
            log.debug(
                "github_activity: hmm=%s dow=%d hour=%d p=%.3f roll=%.3f → SKIP",
                hmm, dow, hour, p_active, roll,
            )
            return 0

        # Active! How many commits? Negative binomial.
        n_commits = self._neg_binomial(NB_N, NB_P)
        n_commits = max(1, n_commits)

        # Respect remaining daily/weekly budget
        daily_remaining = MAX_COMMITS_PER_DAY - state["commits_today"]
        weekly_remaining = MAX_COMMITS_PER_WEEK - state["commits_this_week"]
        n_commits = min(n_commits, daily_remaining, weekly_remaining)

        log.info(
            "github_activity: hmm=%s dow=%d hour=%d p=%.3f roll=%.3f → %d commits",
            hmm, dow, hour, p_active, roll, n_commits,
        )
        return n_commits

    @staticmethod
    def _neg_binomial(n: int, p: float) -> int:
        """Sample from negative binomial without numpy dependency."""
        # NB(n, p): number of failures before n successes
        # Uses the gamma-Poisson mixture representation
        # Gamma(n, (1-p)/p) then Poisson(gamma_sample)
        # For small n, direct geometric sum is simpler
        total = 0
        for _ in range(n):
            # Geometric: number of failures before one success
            u = random.random()
            if u == 0:
                u = 1e-10
            total += int(math.log(u) / math.log(1 - p))
        return total

    def _pick_jitter_seconds(self) -> int:
        """Random jitter 0-50 minutes, weighted toward shorter delays."""
        # Log-normal-ish: most jitters are 1-15 min, occasional long ones
        base = random.expovariate(1.0 / 600)  # mean 10 minutes
        return int(min(base, 3000))  # cap at 50 min

    # ── Content generators ───────────────────────────────────────────────────

    def _query_prometheus(self, query: str) -> list | None:
        """Query Prometheus and return the result."""
        import urllib.parse
        import urllib.request

        url = f"{PROMETHEUS_URL}?query={urllib.parse.quote(query)}"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                if data.get("status") == "success":
                    return data.get("data", {}).get("result", [])
        except Exception as e:
            log.warning("Prometheus query failed (%s): %s", query[:40], e)
        return None

    # ─── tensor-scripts generators ───

    def _generate_infra_snapshot(self, now: datetime) -> tuple[str, str, str, str] | None:
        """Fleet health snapshot from Prometheus → tensor-scripts."""
        snapshot = {"timestamp": now.isoformat(), "observer": "github_activity"}

        queries = {
            "nodes_up": 'up{job="node-exporter"}',
            "cpu_usage_pct": '100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
            "mem_usage_pct": '(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100',
            "disk_free_bytes": 'node_filesystem_avail_bytes{mountpoint="/"}',
            "ceph_health": "ceph_health_status",
            "load_avg_5m": "node_load5",
        }

        for key, query in queries.items():
            result = self._query_prometheus(query)
            if result is not None:
                snapshot[key] = [
                    {"instance": r.get("metric", {}).get("instance", "?"),
                     "value": r.get("value", [None, None])[1]}
                    for r in result
                ]

        if len(snapshot) <= 2:
            return None

        ts = now.strftime("%Y-%m-%d_%H%M%S")
        return (
            "tensor-scripts",
            f"monitoring/snapshots/{ts}.json",
            json.dumps(snapshot, indent=2),
            f"chore(monitoring): fleet health snapshot {now.strftime('%Y-%m-%d')}",
        )

    def _generate_security_trend(self, now: datetime) -> tuple[str, str, str, str] | None:
        """Security trend from .prom metrics → tensor-scripts."""
        rc, output = self._ssh_cmd(
            "cat ~/tensor-scripts/security/output/metrics/*.prom 2>/dev/null", timeout=15,
        )
        if rc != 0 or not output.strip():
            return None

        findings = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        total_metrics = 0
        sources = set()

        for line in output.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            total_metrics += 1
            parts = line.split("{")
            if parts:
                name = parts[0].strip()
                sources.add(name.split("_")[0] if "_" in name else name)
            lower = line.lower()
            for sev in findings:
                if sev in lower:
                    try:
                        findings[sev] += int(float(line.split()[-1]))
                    except (ValueError, IndexError):
                        pass

        date_str = now.strftime("%Y-%m-%d")
        md = [
            f"# Security Trend Summary — {date_str}", "",
            f"Generated: {now.isoformat()}",
            f"Sources: {', '.join(sorted(sources)) or 'N/A'}",
            f"Total metrics: {total_metrics}", "",
            "## Findings by Severity", "",
            "| Severity | Count |", "|----------|-------|",
        ]
        for sev, count in findings.items():
            md.append(f"| {sev.capitalize()} | {count} |")
        md.append("")

        return (
            "tensor-scripts",
            f"security/output/reports/trend_{date_str}.md",
            "\n".join(md) + "\n",
            f"docs(security): trend summary {date_str}",
        )

    def _generate_network_fabric_stats(self, now: datetime) -> tuple[str, str, str, str] | None:
        """Network interface throughput stats → tensor-scripts."""
        queries = {
            "rx_bytes_rate": 'rate(node_network_receive_bytes_total{device=~"enp.*|ens.*|eth.*"}[5m])',
            "tx_bytes_rate": 'rate(node_network_transmit_bytes_total{device=~"enp.*|ens.*|eth.*"}[5m])',
            "errors": 'rate(node_network_receive_errs_total[5m]) + rate(node_network_transmit_errs_total[5m])',
        }
        stats = {"timestamp": now.isoformat()}
        for key, query in queries.items():
            result = self._query_prometheus(query)
            if result:
                stats[key] = [
                    {"instance": r["metric"].get("instance", "?"),
                     "device": r["metric"].get("device", "?"),
                     "value": r["value"][1]}
                    for r in result if float(r["value"][1]) > 0
                ]
        if len(stats) <= 1:
            return None

        ts = now.strftime("%Y-%m-%d_%H%M%S")
        return (
            "tensor-scripts",
            f"monitoring/network/{ts}.json",
            json.dumps(stats, indent=2),
            f"chore(monitoring): network fabric stats {now.strftime('%Y-%m-%d')}",
        )

    def _generate_disk_io_report(self, now: datetime) -> tuple[str, str, str, str] | None:
        """Disk I/O throughput report → tensor-scripts."""
        queries = {
            "read_bytes_rate": "rate(node_disk_read_bytes_total[5m])",
            "write_bytes_rate": "rate(node_disk_written_bytes_total[5m])",
            "io_time_pct": "rate(node_disk_io_time_seconds_total[5m]) * 100",
        }
        report = {"timestamp": now.isoformat()}
        for key, query in queries.items():
            result = self._query_prometheus(query)
            if result:
                # Only include devices with meaningful activity
                report[key] = [
                    {"instance": r["metric"].get("instance", "?"),
                     "device": r["metric"].get("device", "?"),
                     "value": r["value"][1]}
                    for r in result
                    if float(r["value"][1]) > 1000  # >1KB/s
                ]
        if len(report) <= 1:
            return None

        ts = now.strftime("%Y-%m-%d_%H%M%S")
        return (
            "tensor-scripts",
            f"monitoring/disk-io/{ts}.json",
            json.dumps(report, indent=2),
            f"chore(monitoring): disk I/O report {now.strftime('%Y-%m-%d')}",
        )

    # ─── voice-kb generators ───

    def _generate_voicekb_stats(self, now: datetime) -> tuple[str, str, str, str] | None:
        """Monthly aggregate statistics → voice-kb."""
        rc, output = self._ssh_cmd("ls ~/voice-kb/kb/ 2>/dev/null | wc -l", timeout=10)
        if rc != 0:
            return None
        entry_count = int(output.strip()) if output.strip().isdigit() else 0
        if entry_count == 0:
            return None

        rc, first = self._ssh_cmd("ls ~/voice-kb/kb/ | sort | head -1", timeout=10)
        rc, last = self._ssh_cmd("ls ~/voice-kb/kb/ | sort | tail -1", timeout=10)

        rc, topics_raw = self._ssh_cmd(
            "grep -h '^topics:' ~/voice-kb/kb/*.md 2>/dev/null | head -50", timeout=15,
        )
        topic_counts: dict[str, int] = {}
        if topics_raw:
            for line in topics_raw.strip().split("\n"):
                m = re.search(r'\[(.+?)\]', line)
                if m:
                    for t in m.group(1).split(","):
                        t = t.strip().strip("'\"")
                        if t:
                            topic_counts[t] = topic_counts.get(t, 0) + 1

        month_key = now.strftime("%Y-%m")
        stats = {
            "month": month_key,
            "generated": now.isoformat(),
            "entry_count": entry_count,
            "first_entry": first.strip() if first else None,
            "last_entry": last.strip() if last else None,
            "topic_frequencies": dict(sorted(topic_counts.items(), key=lambda x: -x[1])[:30]),
        }
        return (
            "voice-kb",
            f"stats/{month_key}.json",
            json.dumps(stats, indent=2) + "\n",
            "docs(stats): voice-kb statistics update",
        )

    def _generate_voicekb_topic_trends(self, now: datetime) -> tuple[str, str, str, str] | None:
        """Weekly topic trend analysis → voice-kb."""
        # Get entries from last 7 days
        cutoff = (now - timedelta(days=7)).strftime("%Y%m%d")
        rc, output = self._ssh_cmd(
            f"ls ~/voice-kb/kb/ 2>/dev/null | awk '$0 >= \"{cutoff}\"'", timeout=10,
        )
        if rc != 0 or not output.strip():
            return None

        recent_files = [f.strip() for f in output.strip().split("\n") if f.strip()]
        if len(recent_files) < 2:
            return None

        # Sample topics from recent entries
        rc, topics_raw = self._ssh_cmd(
            f"cd ~/voice-kb/kb && grep -h '^topics:' {' '.join(recent_files[:30])} 2>/dev/null",
            timeout=15,
        )
        topics: dict[str, int] = {}
        if topics_raw:
            for line in topics_raw.strip().split("\n"):
                m = re.search(r'\[(.+?)\]', line)
                if m:
                    for t in m.group(1).split(","):
                        t = t.strip().strip("'\"")
                        if t:
                            topics[t] = topics.get(t, 0) + 1

        # Sample types
        rc, types_raw = self._ssh_cmd(
            f"cd ~/voice-kb/kb && grep -h '^type:' {' '.join(recent_files[:30])} 2>/dev/null",
            timeout=15,
        )
        types: dict[str, int] = {}
        if types_raw:
            for line in types_raw.strip().split("\n"):
                t = line.replace("type:", "").strip().strip("'\"")
                if t:
                    types[t] = types.get(t, 0) + 1

        week = now.strftime("%Y-W%W")
        report = {
            "week": week,
            "generated": now.isoformat(),
            "entries_last_7_days": len(recent_files),
            "top_topics": dict(sorted(topics.items(), key=lambda x: -x[1])[:15]),
            "type_distribution": types,
        }
        return (
            "voice-kb",
            f"stats/trends/{week}.json",
            json.dumps(report, indent=2) + "\n",
            f"docs(stats): weekly topic trends {week}",
        )

    # ─── ecommerce-agent generators ───

    def _generate_ecommerce_inventory(self, now: datetime) -> tuple[str, str, str, str] | None:
        """Listing pipeline stats from SQLite → ecommerce-agent."""
        rc, output = self._ssh_cmd(
            "sqlite3 ~/ecommerce-agent/ecommerce.db "
            "\"SELECT status, COUNT(*) FROM drafts GROUP BY status ORDER BY status\" 2>/dev/null",
            timeout=10,
        )
        if rc != 0 or not output.strip():
            return None

        status_counts = {}
        for line in output.strip().split("\n"):
            parts = line.split("|")
            if len(parts) == 2:
                status_counts[parts[0].strip()] = int(parts[1].strip())

        if not status_counts:
            return None

        rc, recent = self._ssh_cmd(
            "sqlite3 ~/ecommerce-agent/ecommerce.db "
            "\"SELECT COUNT(*) FROM audit_log WHERE timestamp > datetime('now', '-7 days')\" 2>/dev/null",
            timeout=10,
        )
        recent_actions = int(recent.strip()) if recent and recent.strip().isdigit() else 0

        report = {
            "date": now.strftime("%Y-%m-%d"),
            "generated": now.isoformat(),
            "listing_status_distribution": status_counts,
            "total_listings": sum(status_counts.values()),
            "audit_actions_last_7_days": recent_actions,
        }
        return (
            "ecommerce-agent",
            f"reports/inventory_{now.strftime('%Y-%m-%d')}.json",
            json.dumps(report, indent=2) + "\n",
            f"docs(inventory): listing pipeline stats {now.strftime('%Y-%m-%d')}",
        )

    # ─── arabic-qa generators ───

    def _generate_qa_summary(self, now: datetime) -> tuple[str, str, str, str] | None:
        """TTS model comparison stats → arabic-qa."""
        rc, output = self._ssh_cmd(
            "ls ~/arabic-qa/auto-results/*.json 2>/dev/null | tail -20", timeout=10,
        )
        if rc != 0 or not output.strip():
            return None

        files = [f.strip() for f in output.strip().split("\n") if f.strip()]
        if not files:
            return None

        # Parse recent results
        winners = {"xtts": 0, "habibi": 0, "tie": 0}
        total = 0
        for f in files[-10:]:
            rc, content = self._ssh_cmd(f"cat {f} 2>/dev/null", timeout=5)
            if rc != 0:
                continue
            try:
                data = json.loads(content)
                w = data.get("winner", "").lower()
                if w in winners:
                    winners[w] += 1
                total += 1
            except (json.JSONDecodeError, AttributeError):
                pass

        if total == 0:
            return None

        summary = {
            "date": now.strftime("%Y-%m-%d"),
            "generated": now.isoformat(),
            "recent_comparisons": total,
            "winners": winners,
            "files_analyzed": len(files),
        }
        return (
            "arabic-qa",
            f"results/summary_{now.strftime('%Y-%m-%d')}.json",
            json.dumps(summary, indent=2) + "\n",
            f"docs(qa): TTS model comparison summary {now.strftime('%Y-%m-%d')}",
        )

    # ─── nexus generators ───

    def _generate_observer_health(self, now: datetime) -> tuple[str, str, str, str] | None:
        """Observer execution health report → nexus."""
        state_dir = Path(os.environ.get(
            "OBSERVER_STATE_DIR", str(Path(__file__).parent / ".state")
        ))
        if not state_dir.exists():
            return None

        observer_states = {}
        for sf in state_dir.glob("*.json"):
            if sf.name == "github_activity_state.json":
                continue
            try:
                data = json.loads(sf.read_text())
                observer_states[sf.stem] = {
                    "last_run": data.get("last_run") or data.get("last_check"),
                    "total_runs": data.get("total_syncs") or data.get("total_runs", "?"),
                }
            except Exception:
                observer_states[sf.stem] = {"error": "unreadable"}

        if not observer_states:
            return None

        report = {
            "date": now.strftime("%Y-%m-%d"),
            "generated": now.isoformat(),
            "observers": observer_states,
            "total_tracked": len(observer_states),
        }
        return (
            "nexus",
            f"observers/.state/health_{now.strftime('%Y-%m-%d')}.json",
            json.dumps(report, indent=2) + "\n",
            f"chore(observers): health report {now.strftime('%Y-%m-%d')}",
        )

    # ─── bookengine generators ───

    def _generate_bookengine_stats(self, now: datetime) -> tuple[str, str, str, str] | None:
        """Book index statistics → bookengine."""
        rc, output = self._ssh_cmd(
            "sqlite3 ~/bookengine/bookengine.db "
            "\"SELECT COUNT(*) FROM books\" 2>/dev/null",
            timeout=10,
        )
        if rc != 0 or not output.strip():
            return None

        book_count = int(output.strip()) if output.strip().isdigit() else 0

        rc, chunks = self._ssh_cmd(
            "sqlite3 ~/bookengine/bookengine.db "
            "\"SELECT COUNT(*) FROM chunks\" 2>/dev/null",
            timeout=10,
        )
        chunk_count = int(chunks.strip()) if chunks and chunks.strip().isdigit() else 0

        if book_count == 0:
            return None

        stats = {
            "date": now.strftime("%Y-%m-%d"),
            "generated": now.isoformat(),
            "books_indexed": book_count,
            "chunks_indexed": chunk_count,
        }
        return (
            "bookengine",
            f"stats/index_{now.strftime('%Y-%m')}.json",
            json.dumps(stats, indent=2) + "\n",
            f"docs(stats): book index statistics update",
        )

    # ─── echo-voicememo generators ───

    def _generate_voicememo_stats(self, now: datetime) -> tuple[str, str, str, str] | None:
        """Voice memo pipeline stats → echo-voicememo."""
        rc, output = self._ssh_cmd(
            "ls ~/echo-voicememo/memos/ 2>/dev/null | wc -l", timeout=10,
        )
        memo_count = int(output.strip()) if output and output.strip().isdigit() else 0

        rc, enrolled = self._ssh_cmd(
            "ls ~/echo-voicememo/enrolled/ 2>/dev/null | wc -l", timeout=10,
        )
        enrolled_count = int(enrolled.strip()) if enrolled and enrolled.strip().isdigit() else 0

        if memo_count == 0 and enrolled_count == 0:
            return None

        stats = {
            "date": now.strftime("%Y-%m-%d"),
            "generated": now.isoformat(),
            "memos_processed": memo_count,
            "speakers_enrolled": enrolled_count,
        }
        return (
            "echo-voicememo",
            f"stats/pipeline_{now.strftime('%Y-%m')}.json",
            json.dumps(stats, indent=2) + "\n",
            "docs(stats): voice memo pipeline stats",
        )

    # ─── autopen generators ───

    def _generate_autopen_stats(self, now: datetime) -> tuple[str, str, str, str] | None:
        """Template processing stats → autopen."""
        rc, output = self._ssh_cmd(
            "ls ~/autopen/templates/ 2>/dev/null | wc -l", timeout=10,
        )
        template_count = int(output.strip()) if output and output.strip().isdigit() else 0

        rc, output2 = self._ssh_cmd(
            "ls ~/autopen/output/ 2>/dev/null | wc -l", timeout=10,
        )
        output_count = int(output2.strip()) if output2 and output2.strip().isdigit() else 0

        if template_count == 0 and output_count == 0:
            return None

        stats = {
            "date": now.strftime("%Y-%m-%d"),
            "generated": now.isoformat(),
            "templates_available": template_count,
            "outputs_generated": output_count,
        }
        return (
            "autopen",
            f"stats/usage_{now.strftime('%Y-%m')}.json",
            json.dumps(stats, indent=2) + "\n",
            "docs(stats): autopen usage statistics",
        )

    # ── Git operations ───────────────────────────────────────────────────────

    def _security_scan(self, content: str) -> list[str]:
        violations = []
        for pattern in BLOCK_PATTERNS:
            for match in re.findall(pattern, content):
                if any(p in match.lower() for p in ["replace_me", "your-key", "example", "dummy"]):
                    continue
                violations.append(f"Pattern matched: {match[:50]}")
        return violations

    def _write_and_commit(
        self, repo_name: str, file_path: str, content: str, commit_msg: str,
        dry_run: bool = False,
    ) -> dict:
        """Write a file to a repo on TC, commit, and push."""
        result = {
            "repo": repo_name, "file": file_path, "committed": False,
            "pushed_github": False, "pushed_gitea": False,
            "message": commit_msg, "error": "",
        }

        repo = REPOS.get(repo_name)
        if not repo:
            result["error"] = f"Unknown repo: {repo_name}"
            return result

        violations = self._security_scan(content)
        if violations:
            result["error"] = f"BLOCKED — secrets: {'; '.join(violations[:3])}"
            return result

        if dry_run:
            result["message"] = f"[DRY RUN] {file_path} → {commit_msg}"
            return result

        # Ensure directory exists
        dir_path = os.path.dirname(file_path)
        if dir_path:
            self._ssh_cmd(f"mkdir -p ~/{repo_name}/{dir_path}", timeout=10)

        # Write content to TC via stdin piped to tee
        full_path = f"$HOME/{repo_name}/{file_path}"
        try:
            proc = subprocess.run(
                [
                    "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                    "puretensorai@localhost",
                    f"tee {full_path} > /dev/null",
                ],
                input=content, capture_output=True, text=True, timeout=15,
            )
            if proc.returncode != 0:
                result["error"] = f"Write failed: {proc.stderr[:200]}"
                return result
        except Exception as e:
            result["error"] = f"Write failed: {e}"
            return result

        # Commit (use -f in case path is gitignored)
        escaped_msg = commit_msg.replace("'", "'\\''")
        rc, output = self._ssh_cmd(
            f"cd ~/{repo_name} && git add -f {file_path} && "
            f"git commit -m '{escaped_msg}' 2>&1",
            timeout=30,
        )
        if rc != 0:
            if "nothing to commit" in output:
                result["message"] = "nothing to commit (unchanged)"
                return result
            result["error"] = f"Commit failed: {output[:200]}"
            return result

        result["committed"] = True

        # Push to Gitea
        if repo.get("gitea_remote"):
            rc, _ = self._ssh_cmd(
                f"cd ~/{repo_name} && git push {repo['gitea_remote']} HEAD 2>&1", timeout=60,
            )
            result["pushed_gitea"] = (rc == 0)

        # Push to GitHub
        if repo.get("github_remote"):
            rc, _ = self._ssh_cmd(
                f"cd ~/{repo_name} && git push {repo['github_remote']} HEAD 2>&1", timeout=60,
            )
            result["pushed_github"] = (rc == 0)

        return result

    # ── Main observer interface ──────────────────────────────────────────────

    def _get_all_generators(self):
        """Return all content generators grouped by repo for variety."""
        return [
            # tensor-scripts (4 generators — most infra content)
            self._generate_infra_snapshot,
            self._generate_security_trend,
            self._generate_network_fabric_stats,
            self._generate_disk_io_report,
            # voice-kb (2 generators)
            self._generate_voicekb_stats,
            self._generate_voicekb_topic_trends,
            # ecommerce-agent
            self._generate_ecommerce_inventory,
            # arabic-qa
            self._generate_qa_summary,
            # nexus
            self._generate_observer_health,
            # bookengine
            self._generate_bookengine_stats,
            # echo-voicememo
            self._generate_voicememo_stats,
            # autopen
            self._generate_autopen_stats,
        ]

    def run(self, ctx=None, force: bool = False, dry_run: bool = False) -> ObserverResult:
        """Execute the GitHub activity observer."""
        now = self.now_utc()
        state = self._load_state()

        n_commits = self._should_run(state, now, force=force)
        if n_commits == 0:
            return ObserverResult(success=True)  # Silent skip

        # Jitter (skip in dry-run/force)
        if not dry_run and not force:
            jitter = self._pick_jitter_seconds()
            log.info("github_activity: jitter %d seconds", jitter)
            time.sleep(jitter)

        # Test SSH
        rc, output = self._ssh_cmd("echo OK", timeout=10)
        if rc != 0 or "OK" not in output:
            return ObserverResult(success=False, error="SSH to TC failed")

        # Shuffle generators and try to produce n_commits
        generators = self._get_all_generators()
        random.shuffle(generators)

        results = []
        committed = 0

        for gen in generators:
            if committed >= n_commits:
                break

            try:
                gen_result = gen(now)
            except Exception as e:
                log.warning("github_activity: %s failed: %s", gen.__name__, e)
                continue

            if gen_result is None:
                continue

            repo_name, file_path, content, commit_msg = gen_result
            log.info("github_activity: %s → %s/%s", gen.__name__, repo_name, file_path)

            result = self._write_and_commit(repo_name, file_path, content, commit_msg, dry_run=dry_run)
            results.append(result)

            if result["committed"]:
                committed += 1
                # Small delay between commits for realism
                if committed < n_commits:
                    time.sleep(random.randint(5, 30))

        # Update state
        if committed > 0:
            state["commits_today"] = state.get("commits_today", 0) + committed
            state["commits_this_week"] = state.get("commits_this_week", 0) + committed
            state["total_commits"] = state.get("total_commits", 0) + committed
            state["last_commit"] = now.isoformat()
            for r in results:
                if r["committed"]:
                    state.setdefault("history", []).append({
                        "time": now.isoformat(),
                        "repo": r["repo"],
                        "file": r["file"],
                        "msg": r["message"],
                    })

        self._save_state(state)

        # Telegram summary (only if commits were made)
        if committed > 0 and not dry_run:
            lines = [f"GITHUB ACTIVITY — {now.strftime('%H:%M UTC')}"]
            for r in results:
                if r["committed"]:
                    push = []
                    if r["pushed_github"]:
                        push.append("GH")
                    if r["pushed_gitea"]:
                        push.append("Gitea")
                    lines.append(f"  {r['repo']}: {r['message']} → {','.join(push) or '?'}")
            lines.append(
                f"  Today: {state['commits_today']} | "
                f"Week: {state['commits_this_week']} | "
                f"Total: {state['total_commits']}"
            )
            self.send_telegram("\n".join(lines))

        errors = [r for r in results if r.get("error")]
        return ObserverResult(
            success=len(errors) == 0,
            message=f"{committed} commits across {len(set(r['repo'] for r in results if r['committed']))} repos",
            data={"committed": committed, "results": results},
        )


# Standalone execution for testing
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import log as _  # noqa: F401

    observer = GitHubActivityObserver()

    dry_run = "--dry-run" in sys.argv
    force: bool | int = "--force" in sys.argv or dry_run
    # Support --force N for multi-commit testing
    if "--force" in sys.argv:
        idx = sys.argv.index("--force")
        if idx + 1 < len(sys.argv) and sys.argv[idx + 1].isdigit():
            force = int(sys.argv[idx + 1])

    if "--simulate" in sys.argv:
        # Simulate 4 weeks of activity to preview the pattern
        print("SIMULATING 4 WEEKS OF ACTIVITY\n")
        state = observer._load_state()
        now = datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)
        day_commits: dict[str, int] = {}

        for day in range(28):
            current_day = now + timedelta(days=day)
            date_str = current_day.strftime("%Y-%m-%d %a")
            daily = 0
            for hour in range(7, 24):
                tick = current_day.replace(hour=hour)
                n = observer._should_run(state, tick)
                if n > 0:
                    daily += n
                    state["commits_today"] = state.get("commits_today", 0) + n
                    state["commits_this_week"] = state.get("commits_this_week", 0) + n
            day_commits[date_str] = daily
            # Reset daily
            state["commits_today"] = 0
            # Reset weekly on Monday
            if (current_day + timedelta(days=1)).weekday() == 0:
                week_total = sum(
                    day_commits.get((current_day - timedelta(days=i)).strftime("%Y-%m-%d %a"), 0)
                    for i in range(7)
                )
                state["commits_this_week"] = 0

        for date_str, count in day_commits.items():
            bar = "█" * count + "░" * (8 - count)
            print(f"  {date_str}  {bar} {count}")

        total = sum(day_commits.values())
        active_days = sum(1 for c in day_commits.values() if c > 0)
        print(f"\n  Total: {total} commits over {active_days}/{len(day_commits)} active days")
        print(f"  Average: {total/4:.1f}/week, {total/28:.1f}/day")
        sys.exit(0)

    if dry_run:
        print("DRY RUN — will generate content but not commit")
    elif force:
        print("FORCE — bypassing probability check")

    result = observer.run(force=force, dry_run=dry_run)

    if result.data:
        for r in result.data.get("results", []):
            print(f"\n  Repo: {r.get('repo')}")
            print(f"  File: {r.get('file')}")
            print(f"  Msg:  {r.get('message')}")
            if r.get("committed"):
                print(f"  Push: GH={r.get('pushed_github')} Gitea={r.get('pushed_gitea')}")
            if r.get("error"):
                print(f"  Error: {r['error']}", file=sys.stderr)
    else:
        print(result.message or "Skipped (probability check)")
