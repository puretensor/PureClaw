"""Gmail collector — unread email counts per account.

Reuses the existing tools/gmail.py OAuth infrastructure.
Runs synchronously via executor since google-api-python-client is sync.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("nexus")

# Add project root to path for tools.gmail import
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

GMAIL_IDENTITY = os.environ.get("GMAIL_IDENTITY", "")


def _count_unread() -> dict:
    """Synchronous: count unread emails per configured account."""
    try:
        from tools.gmail import ACCOUNTS, get_service
    except ImportError as e:
        return {"error": f"gmail import failed: {e}"}

    results = {}
    for key, acct in ACCOUNTS.items():
        try:
            service = get_service(key)
            resp = service.users().messages().list(
                userId="me", q="is:unread", maxResults=1
            ).execute()
            # resultSizeEstimate gives approximate unread count
            results[key] = {
                "email": acct.get("name", key),
                "unread": resp.get("resultSizeEstimate", 0),
            }
        except Exception as e:
            results[key] = {"email": acct.get("name", key), "error": str(e)}

    return results


async def collect() -> dict:
    """Count unread emails across all configured Gmail accounts."""
    loop = asyncio.get_event_loop()
    try:
        accounts = await loop.run_in_executor(None, _count_unread)
    except Exception as e:
        return {"error": str(e)}

    total_unread = sum(
        a.get("unread", 0) for a in accounts.values() if "error" not in a
    )
    return {
        "total_unread": total_unread,
        "accounts": accounts,
    }
