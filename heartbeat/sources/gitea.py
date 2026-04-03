"""Gitea collector — recent activity and repo stats."""

import os

import httpx

GITEA_URL = os.environ.get("GITEA_URL", "")
GITEA_TOKEN = os.environ.get("GITEA_TOKEN", "")


async def collect() -> dict:
    """Fetch recent Gitea activity."""
    if not GITEA_URL or not GITEA_TOKEN:
        return {"error": "GITEA_URL or GITEA_TOKEN not configured"}

    headers = {"Authorization": f"token {GITEA_TOKEN}"}
    base = GITEA_URL.rstrip("/")

    async with httpx.AsyncClient(headers=headers, timeout=8.0) as client:
        # Recent repos (sorted by last updated)
        try:
            r = await client.get(f"{base}/api/v1/repos/search", params={
                "sort": "updated", "order": "desc", "limit": 5,
            })
            r.raise_for_status()
            repos = [
                {"name": repo["full_name"], "updated": repo["updated_at"]}
                for repo in r.json().get("data", [])
            ]
        except Exception as e:
            repos = [{"error": str(e)}]

        # Org/user repo count
        try:
            r = await client.get(f"{base}/api/v1/repos/search", params={"limit": 1})
            r.raise_for_status()
            # Gitea returns X-Total-Count header
            total_repos = int(r.headers.get("X-Total-Count", 0))
        except Exception:
            total_repos = -1

    return {
        "total_repos": total_repos,
        "recently_updated": repos,
    }
