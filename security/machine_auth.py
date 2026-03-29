"""Shared HMAC auth for machine-to-machine ingress."""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Mapping

SIGNATURE_HEADER = "X-Nexus-Signature"
TIMESTAMP_HEADER = "X-Nexus-Timestamp"
DEFAULT_MAX_SKEW_SECS = 300


def sign_payload(body: bytes, secret: str, timestamp: str | None = None) -> dict[str, str]:
    """Return auth headers for a request body."""
    if not secret:
        raise ValueError("secret is required")
    timestamp = timestamp or str(int(time.time()))
    mac = hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256)
    return {
        TIMESTAMP_HEADER: timestamp,
        SIGNATURE_HEADER: f"sha256={mac.hexdigest()}",
    }


def verify_headers(
    body: bytes,
    headers: Mapping[str, str],
    secret: str,
    *,
    max_skew_secs: int = DEFAULT_MAX_SKEW_SECS,
) -> tuple[bool, str]:
    """Verify timestamped HMAC headers for a request body."""
    if not secret:
        return False, "shared secret not configured"

    timestamp = headers.get(TIMESTAMP_HEADER, "")
    signature = headers.get(SIGNATURE_HEADER, "")
    if not timestamp or not signature:
        return False, "missing signature headers"

    if signature.startswith("sha256="):
        signature = signature.split("=", 1)[1]

    try:
        ts = int(timestamp)
    except ValueError:
        return False, "invalid timestamp"

    now = int(time.time())
    if abs(now - ts) > max_skew_secs:
        return False, "stale timestamp"

    expected = hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        return False, "invalid signature"
    return True, "ok"
