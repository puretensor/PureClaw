"""Prometheus collector — active alerts and down targets."""

import json
import os
import urllib.parse
import urllib.request


PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "")
ALERTMANAGER_URL = os.environ.get("ALERTMANAGER_URL", "")


def _query_prom(query: str) -> list:
    """Run an instant PromQL query synchronously."""
    url = f"{PROMETHEUS_URL}/api/v1/query?query={urllib.parse.quote(query)}"
    resp = urllib.request.urlopen(url, timeout=8)
    data = json.loads(resp.read())
    return data.get("data", {}).get("result", [])


def _get_alerts() -> list:
    """Fetch active alerts from Alertmanager."""
    url = f"{ALERTMANAGER_URL}/api/v2/alerts?active=true&silenced=false&inhibited=false"
    resp = urllib.request.urlopen(url, timeout=8)
    return json.loads(resp.read())


async def collect() -> dict:
    """Gather Prometheus down targets and active alerts."""
    import asyncio

    if not PROMETHEUS_URL:
        return {"error": "PROMETHEUS_URL not configured"}

    loop = asyncio.get_event_loop()

    # Down targets
    try:
        down_results = await loop.run_in_executor(None, _query_prom, "up == 0")
        down_targets = [
            {"instance": r["metric"].get("instance", "?"), "job": r["metric"].get("job", "?")}
            for r in down_results
        ]
    except Exception as e:
        down_targets = [{"error": str(e)}]

    # Active alerts
    alerts = []
    if ALERTMANAGER_URL:
        try:
            raw = await loop.run_in_executor(None, _get_alerts)
            alerts = [
                {
                    "name": a.get("labels", {}).get("alertname", "?"),
                    "severity": a.get("labels", {}).get("severity", "?"),
                    "instance": a.get("labels", {}).get("instance", ""),
                }
                for a in raw
            ]
        except Exception as e:
            alerts = [{"error": str(e)}]

    return {
        "down_targets": down_targets,
        "down_count": len([d for d in down_targets if "error" not in d]),
        "active_alerts": alerts,
        "alert_count": len([a for a in alerts if "error" not in a]),
    }
