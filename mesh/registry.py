"""ClawRegistry — peer discovery and health tracking for the PureClaw mesh."""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from dataclasses import dataclass, field

log = logging.getLogger("nexus.mesh")


@dataclass
class PeerState:
    """Tracked state of a mesh peer."""
    url: str
    last_seen: float = 0.0
    consecutive_failures: int = 0
    online: bool = False
    claw_id: str = ""

    # 3 consecutive failures = offline, 1 success = online
    FAILURE_THRESHOLD = 3

    def mark_success(self) -> bool:
        """Mark peer as reachable. Returns True if state changed (was offline)."""
        was_offline = not self.online
        self.online = True
        self.consecutive_failures = 0
        self.last_seen = time.monotonic()
        return was_offline

    def mark_failure(self) -> bool:
        """Mark peer as unreachable. Returns True if state changed (went offline)."""
        self.consecutive_failures += 1
        if self.online and self.consecutive_failures >= self.FAILURE_THRESHOLD:
            self.online = False
            return True
        return False


class ClawRegistry:
    """Registry of mesh peers, loaded from CLAW_MESH_PEERS env var.

    CLAW_MESH_PEERS format: JSON dict of claw_id -> base URL.
    Example: {"prime":"http://<TS_FOX_N1>:30876","infra":"http://<TS_TENSOR_CORE>:9880"}
    """

    def __init__(self, peers_json: str = "", self_id: str = ""):
        self._self_id = self_id
        self._peers: dict[str, PeerState] = {}

        if peers_json:
            try:
                peers = json.loads(peers_json)
                for claw_id, url in peers.items():
                    if claw_id == self_id:
                        continue  # don't track self
                    self._peers[claw_id] = PeerState(url=url.rstrip("/"), claw_id=claw_id)
            except (json.JSONDecodeError, AttributeError) as e:
                log.warning("Failed to parse CLAW_MESH_PEERS: %s", e)

        if self._peers:
            log.info("Mesh registry: %d peers (%s)", len(self._peers), ", ".join(self._peers))

    def get_peer_url(self, claw_id: str) -> str | None:
        """Get the base URL for a peer, or None if unknown."""
        peer = self._peers.get(claw_id)
        return peer.url if peer else None

    def get_online_peers(self) -> list[str]:
        """Return IDs of peers currently considered online."""
        return [cid for cid, p in self._peers.items() if p.online]

    def get_all_peers(self) -> dict[str, PeerState]:
        """Return all tracked peers."""
        return dict(self._peers)

    def health_check(self, claw_id: str, timeout: float = 3.0) -> bool:
        """Probe a single peer's /health endpoint. Updates state. Returns reachable."""
        peer = self._peers.get(claw_id)
        if not peer:
            return False

        try:
            req = urllib.request.Request(f"{peer.url}/health", method="GET")
            urllib.request.urlopen(req, timeout=timeout)
            changed = peer.mark_success()
            if changed:
                log.info("Mesh peer %s is now ONLINE", claw_id)
            return True
        except Exception:
            changed = peer.mark_failure()
            if changed:
                log.warning("Mesh peer %s is now OFFLINE", claw_id)
            return False

    def health_check_all(self, timeout: float = 3.0) -> dict[str, bool]:
        """Probe all peers. Returns dict of claw_id -> reachable."""
        results = {}
        for claw_id in self._peers:
            results[claw_id] = self.health_check(claw_id, timeout)
        return results

    def status_summary(self) -> str:
        """Human-readable status of all peers."""
        if not self._peers:
            return "No mesh peers configured"
        lines = []
        for cid, peer in sorted(self._peers.items()):
            status = "ONLINE" if peer.online else "OFFLINE"
            age = ""
            if peer.last_seen:
                ago = int(time.monotonic() - peer.last_seen)
                age = f" (last seen {ago}s ago)"
            lines.append(f"  {cid}: {status}{age} @ {peer.url}")
        return "\n".join(lines)
