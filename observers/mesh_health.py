"""MeshHealthObserver — monitors health of all Claw mesh peers.

Reports status changes (online/offline transitions) via Telegram.
"""

from __future__ import annotations

import logging

from observers.base import Observer, ObserverContext, ObserverResult

log = logging.getLogger("nexus.mesh")


class MeshHealthObserver(Observer):
    name = "mesh_health"
    schedule = "*/5 * * * *"  # every 5 minutes

    def __init__(self):
        self._registry = None
        self._initialized = False

    def _ensure_registry(self):
        if self._initialized:
            return
        self._initialized = True
        try:
            from config import CLAW_MESH_PEERS, CLAW_ID
            from mesh.registry import ClawRegistry
            self._registry = ClawRegistry(CLAW_MESH_PEERS, CLAW_ID)
        except Exception as e:
            log.warning("MeshHealthObserver: failed to init registry: %s", e)

    def run(self, ctx: ObserverContext) -> ObserverResult:
        self._ensure_registry()
        if not self._registry or not self._registry.get_all_peers():
            return ObserverResult(success=True, message="No mesh peers configured")

        results = self._registry.health_check_all(timeout=5.0)

        # Build status report
        changes = []
        for claw_id, reachable in results.items():
            peer = self._registry.get_all_peers().get(claw_id)
            if not peer:
                continue
            # Detect transitions (mark_success/mark_failure already update state)
            # We just report the current state
            status = "ONLINE" if peer.online else "OFFLINE"
            if not reachable and peer.consecutive_failures == peer.FAILURE_THRESHOLD:
                changes.append(f"Claw-{claw_id} went OFFLINE")
            elif reachable and peer.consecutive_failures == 0 and peer.last_seen > 0:
                # First success — only report if it was previously tracked
                pass

        # Send Telegram alert for state changes
        if changes:
            msg = "*Mesh Status Change*\n" + "\n".join(f"- {c}" for c in changes)
            try:
                self.send_telegram(msg)
            except Exception as e:
                log.error("Failed to send mesh status alert: %s", e)

        online = [cid for cid, r in results.items() if r]
        offline = [cid for cid, r in results.items() if not r]

        summary = f"Online: {', '.join(online) or 'none'}"
        if offline:
            summary += f" | Offline: {', '.join(offline)}"

        return ObserverResult(
            success=True,
            message=summary if changes else None,
            data={"online": online, "offline": offline, "changes": changes},
        )
