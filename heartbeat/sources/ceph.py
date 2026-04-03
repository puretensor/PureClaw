"""Ceph collector — cluster health and capacity."""

import asyncio
import json
import os

TC_SSH_HOST = os.environ.get("TC_SSH_HOST", "")


async def collect() -> dict:
    """Get Ceph cluster status via SSH to tensor-core."""
    if not TC_SSH_HOST:
        return {"error": "TC_SSH_HOST not configured"}

    proc = await asyncio.create_subprocess_exec(
        "ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
        f"root@{TC_SSH_HOST}",
        "ceph", "-s", "--format", "json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        return {"error": stderr.decode().strip() or f"exit code {proc.returncode}"}

    data = json.loads(stdout)
    health = data.get("health", {})
    pgmap = data.get("pgmap", {})
    osdmap = data.get("osdmap", {})

    # Capacity
    total_bytes = pgmap.get("bytes_total", 0)
    used_bytes = pgmap.get("bytes_used", 0)
    avail_bytes = pgmap.get("bytes_avail", 0)

    return {
        "status": health.get("status", "UNKNOWN"),
        "checks": list(health.get("checks", {}).keys()),
        "osds_up": osdmap.get("num_up_osds", 0),
        "osds_total": osdmap.get("num_osds", 0),
        "capacity_tb": round(total_bytes / (1024**4), 2),
        "used_tb": round(used_bytes / (1024**4), 2),
        "avail_tb": round(avail_bytes / (1024**4), 2),
        "used_pct": round(used_bytes / total_bytes * 100, 1) if total_bytes else 0,
    }
