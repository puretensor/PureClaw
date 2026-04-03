"""K3s collector — pod status across all namespaces.

Uses the Kubernetes API via in-cluster service account token when available,
falls back to kubectl CLI when running outside the cluster.
"""

import asyncio
import json
import os
import ssl
from pathlib import Path

import httpx

# In-cluster paths (auto-mounted by K8s)
_SA_TOKEN = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
_SA_CA = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
_K8S_HOST = os.environ.get("KUBERNETES_SERVICE_HOST", "")
_K8S_PORT = os.environ.get("KUBERNETES_SERVICE_PORT", "443")


def _parse_pods(pods: list) -> dict:
    """Parse pod list into summary."""
    summary = {"running": 0, "pending": 0, "failed": 0, "crash_loop": 0, "namespaces": {}}
    unhealthy = []

    for pod in pods:
        ns = pod["metadata"]["namespace"]
        name = pod["metadata"]["name"]
        phase = pod.get("status", {}).get("phase", "Unknown")

        summary["namespaces"].setdefault(ns, {"running": 0, "not_ready": 0})

        if phase == "Running":
            containers = pod.get("status", {}).get("containerStatuses", [])
            crash = any(
                cs.get("state", {}).get("waiting", {}).get("reason") == "CrashLoopBackOff"
                for cs in containers
            )
            if crash:
                summary["crash_loop"] += 1
                summary["namespaces"][ns]["not_ready"] += 1
                unhealthy.append({"namespace": ns, "pod": name, "issue": "CrashLoopBackOff"})
            else:
                ready = all(cs.get("ready", False) for cs in containers) if containers else True
                if ready:
                    summary["running"] += 1
                    summary["namespaces"][ns]["running"] += 1
                else:
                    summary["namespaces"][ns]["not_ready"] += 1
                    unhealthy.append({"namespace": ns, "pod": name, "issue": "not_ready"})
        elif phase == "Pending":
            summary["pending"] += 1
            summary["namespaces"][ns]["not_ready"] += 1
            unhealthy.append({"namespace": ns, "pod": name, "issue": "pending"})
        elif phase == "Failed":
            summary["failed"] += 1
            summary["namespaces"][ns]["not_ready"] += 1
            unhealthy.append({"namespace": ns, "pod": name, "issue": "failed"})

    total = summary["running"] + summary["pending"] + summary["failed"] + summary["crash_loop"]
    return {
        "total_pods": total,
        "healthy": summary["running"],
        "unhealthy_count": len(unhealthy),
        "unhealthy": unhealthy[:10],
        "namespaces": summary["namespaces"],
    }


async def _collect_api() -> dict:
    """Collect via in-cluster Kubernetes API."""
    token = _SA_TOKEN.read_text().strip()
    ca_path = str(_SA_CA)
    base = f"https://{_K8S_HOST}:{_K8S_PORT}"

    ssl_ctx = ssl.create_default_context(cafile=ca_path)
    async with httpx.AsyncClient(
        base_url=base,
        headers={"Authorization": f"Bearer {token}"},
        verify=ssl_ctx,
        timeout=8.0,
    ) as client:
        r = await client.get("/api/v1/pods", params={
            "fieldSelector": "status.phase!=Succeeded",
        })
        r.raise_for_status()
        pods = r.json().get("items", [])
    return _parse_pods(pods)


async def _collect_kubectl() -> dict:
    """Collect via kubectl CLI (outside cluster)."""
    proc = await asyncio.create_subprocess_exec(
        "kubectl", "get", "pods", "-A",
        "-o", "json",
        "--field-selector=status.phase!=Succeeded",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        return {"error": stderr.decode().strip() or f"exit code {proc.returncode}"}
    pods = json.loads(stdout).get("items", [])
    return _parse_pods(pods)


async def collect() -> dict:
    """Get pod status — uses K8s API in-cluster, kubectl outside."""
    if _SA_TOKEN.exists() and _K8S_HOST:
        try:
            return await _collect_api()
        except Exception as e:
            # Fall back to kubectl if API fails
            try:
                return await _collect_kubectl()
            except Exception:
                return {"error": f"k8s API: {e}"}
    else:
        return await _collect_kubectl()
