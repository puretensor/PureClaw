"""Async orchestrator — runs all data collectors in parallel.

Usage:
    import asyncio
    from heartbeat.gather import gather_all
    data = asyncio.run(gather_all())
"""

import asyncio
import logging
import time
from datetime import datetime, timezone

from heartbeat.sources import gmail as src_gmail
from heartbeat.sources import github as src_github
from heartbeat.sources import k3s as src_k3s
from heartbeat.sources import ceph as src_ceph
from heartbeat.sources import services as src_services
from heartbeat.sources import prometheus as src_prometheus
from heartbeat.sources import gitea as src_gitea

log = logging.getLogger("nexus")

COLLECTOR_TIMEOUT = 10  # seconds per collector


async def _run_collector(name: str, coro) -> tuple[str, dict]:
    """Run a single collector with timeout and error isolation."""
    try:
        result = await asyncio.wait_for(coro, timeout=COLLECTOR_TIMEOUT)
        return name, result
    except asyncio.TimeoutError:
        log.warning("[gather] %s timed out after %ds", name, COLLECTOR_TIMEOUT)
        return name, {"error": "timeout", "timeout_seconds": COLLECTOR_TIMEOUT}
    except Exception as e:
        log.warning("[gather] %s failed: %s", name, e)
        return name, {"error": str(e)}


async def gather_all() -> dict:
    """Collect data from all sources in parallel.

    Returns a dict keyed by source name. Each value is either the collector's
    result dict or {"error": "..."} on failure. Top-level keys:
        gathered_at, duration_ms, gmail, github, k3s, ceph, services,
        prometheus, gitea
    """
    start = time.monotonic()

    collectors = [
        ("gmail", src_gmail.collect()),
        ("github", src_github.collect()),
        ("k3s", src_k3s.collect()),
        ("ceph", src_ceph.collect()),
        ("services", src_services.collect()),
        ("prometheus", src_prometheus.collect()),
        ("gitea", src_gitea.collect()),
    ]

    tasks = [_run_collector(name, coro) for name, coro in collectors]
    results = await asyncio.gather(*tasks)

    data = {
        "gathered_at": datetime.now(timezone.utc).isoformat(),
        "duration_ms": int((time.monotonic() - start) * 1000),
    }
    for name, result in results:
        data[name] = result

    log.info("[gather] Completed in %dms (%d sources)", data["duration_ms"], len(collectors))
    return data
