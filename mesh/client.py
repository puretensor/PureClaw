"""ClawClient — send messages to mesh peers."""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error

from .message import ClawMessage
from .registry import ClawRegistry
from security.machine_auth import sign_payload

log = logging.getLogger("nexus.mesh")


class ClawClient:
    """HTTP client for sending ClawMessages to mesh peers.

    Follows the einherjar_dispatch pattern: POST JSON, get JSON response.
    """

    def __init__(self, registry: ClawRegistry, self_id: str = ""):
        self._registry = registry
        self._self_id = self_id

    def send(
        self,
        to_claw: str,
        msg_type: str,
        payload: dict,
        priority: int = 1,
        reply_to: str | None = None,
        timeout: int = 120,
        wait: bool = True,
    ) -> dict:
        """Send a message to a peer Claw.

        Args:
            to_claw: Target Claw ID (e.g. "infra", "ops", "sentinel", "prime")
            msg_type: Message type (alert, task, escalation, report, query, ack)
            payload: Type-specific data dict
            priority: 0=routine, 1=normal, 2=high, 3=critical
            reply_to: Message ID this replies to
            timeout: HTTP timeout in seconds
            wait: If True, wait for response. If False, fire-and-forget.

        Returns:
            Response dict from the peer, or error dict.
        """
        url = self._registry.get_peer_url(to_claw)
        if not url:
            return {"error": f"Unknown peer: {to_claw}"}

        msg = ClawMessage(
            msg_type=msg_type,
            payload=payload,
            from_claw=self._self_id,
            to_claw=to_claw,
            priority=priority,
            reply_to=reply_to,
        )

        errors = msg.validate()
        if errors:
            return {"error": f"Invalid message: {'; '.join(errors)}"}

        endpoint = f"{url}/mesh/message"
        data = msg.to_json().encode()
        mesh_secret = os.environ.get("MESH_SHARED_SECRET", "")
        if not mesh_secret:
            return {"error": "MESH_SHARED_SECRET not configured"}
        auth_headers = sign_payload(data, mesh_secret)

        log.info(
            "Mesh send: %s -> %s [%s] priority=%d",
            self._self_id, to_claw, msg_type, priority,
        )

        try:
            req = urllib.request.Request(
                endpoint,
                data=data,
                headers={"Content-Type": "application/json", **auth_headers},
                method="POST",
            )
            effective_timeout = timeout if wait else 5
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                result = json.loads(resp.read().decode())
                return result
        except urllib.error.URLError as e:
            log.error("Mesh send to %s failed: %s", to_claw, e)
            return {"error": f"Peer {to_claw} unreachable: {e}"}
        except Exception as e:
            log.error("Mesh send to %s error: %s", to_claw, e)
            return {"error": f"Mesh error: {e}"}

    def broadcast(
        self,
        msg_type: str,
        payload: dict,
        priority: int = 1,
        exclude: list[str] | None = None,
    ) -> dict[str, dict]:
        """Send a message to all online peers.

        Returns dict of claw_id -> response.
        """
        exclude = set(exclude or [])
        results = {}
        for claw_id in self._registry.get_online_peers():
            if claw_id in exclude:
                continue
            results[claw_id] = self.send(
                to_claw=claw_id,
                msg_type=msg_type,
                payload=payload,
                priority=priority,
                wait=False,
            )
        return results
