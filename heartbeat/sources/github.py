"""GitHub collector — recent push activity and notifications."""

import os

import httpx

GH_TOKEN = os.environ.get("GH_TOKEN", "")
GH_USER = os.environ.get("GH_USER", "puretensor")


async def collect() -> dict:
    """Fetch recent GitHub activity."""
    if not GH_TOKEN:
        return {"error": "GH_TOKEN not configured"}

    headers = {
        "Authorization": f"Bearer {GH_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient(headers=headers, timeout=8.0, base_url="https://api.github.com") as client:
        # Recent events (pushes, PRs, issues)
        try:
            r = await client.get(f"/users/{GH_USER}/events", params={"per_page": 10})
            r.raise_for_status()
            events = [
                {
                    "type": ev["type"],
                    "repo": ev["repo"]["name"],
                    "created": ev["created_at"],
                }
                for ev in r.json()[:10]
            ]
        except Exception as e:
            events = [{"error": str(e)}]

        # Unread notifications
        try:
            r = await client.get("/notifications", params={"per_page": 5})
            r.raise_for_status()
            notif_count = len(r.json())
        except Exception:
            notif_count = -1

    return {
        "recent_events": events[:5],
        "unread_notifications": notif_count,
    }
