"""ClawRunner — headless entry point for non-Prime Claw agents.

Replaces nexus.py for Claws that don't need Telegram/Discord/WhatsApp channels.
Runs the mesh HTTP server + observer registry + engine, processes incoming
messages via the LLM backend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys

# Ensure parent package is importable when run as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SYSTEM_PROMPT

log = logging.getLogger("nexus.mesh")


class ClawRunner:
    """Headless Claw agent -- mesh server + LLM message processing."""

    def __init__(self):
        from config import ENGINE_BACKEND

        self._claw_id = os.environ.get("CLAW_ID", "unknown")
        self._claw_port = int(os.environ.get("CLAW_PORT", "9880"))
        self._peers_json = os.environ.get("CLAW_MESH_PEERS", "{}")
        self._authority_str = os.environ.get("CLAW_AUTHORITY_LEVEL", "escalate")

        log.info(
            "ClawRunner init: id=%s port=%d backend=%s authority=%s",
            self._claw_id, self._claw_port, ENGINE_BACKEND, self._authority_str,
        )

        # Mesh components
        from .registry import ClawRegistry
        from .client import ClawClient
        from .server import MeshServer
        from .authority import AuthorityLevel

        self._registry = ClawRegistry(self._peers_json, self._claw_id)
        self._client = ClawClient(self._registry, self._claw_id)
        self._authority = AuthorityLevel.from_str(self._authority_str)
        self._server = MeshServer(
            claw_id=self._claw_id,
            port=self._claw_port,
            on_message=self._handle_message,
            on_alert=self._handle_alert,
        )

        # Message processing queue
        self._queue: asyncio.Queue | None = None

    def _handle_message(self, msg) -> dict:
        """Synchronous callback from mesh server -- enqueue for async processing."""
        from .message import ClawMessage

        if msg.msg_type == "query" or msg.msg_type == "task":
            # Process via LLM
            result = self._process_with_llm(msg)
            return {"response": result}
        elif msg.msg_type == "ack":
            log.info("Received ack from %s for message %s", msg.from_claw, msg.reply_to)
            return {"acknowledged": True}
        elif msg.msg_type == "alert":
            return self._handle_alert_message(msg)
        else:
            log.info("Unhandled message type: %s", msg.msg_type)
            return {"received": True}

    def _handle_alert(self, data: dict) -> dict:
        """Handle Alertmanager webhook -- classify and route."""
        alerts = data.get("alerts", [])
        results = []

        for alert in alerts:
            labels = alert.get("labels", {})
            status = alert.get("status", "firing")
            alertname = labels.get("alertname", "unknown")
            instance = labels.get("instance", "")

            log.info("Alert: %s [%s] instance=%s", alertname, status, instance)

            # Route based on label matching
            routed = self._route_alert(alert)
            results.append({"alertname": alertname, "status": status, "routed_to": routed})

        return {"processed": results}

    def _handle_alert_message(self, msg) -> dict:
        """Handle an alert forwarded from another Claw."""
        alert_data = msg.payload
        prompt = (
            f"ALERT from {msg.from_claw} (priority {msg.priority}):\n"
            f"{json.dumps(alert_data, indent=2)}\n\n"
            f"Investigate this alert. Use your tools to check the relevant systems. "
            f"Report your findings and any actions taken."
        )
        result = self._process_with_llm_prompt(prompt)
        return {"response": result, "action_taken": True}

    def _route_alert(self, alert: dict) -> str | None:
        """Route an alert to the appropriate Claw based on labels."""
        labels = alert.get("labels", {})
        alertname = labels.get("alertname", "")
        job = labels.get("job", "")
        instance = labels.get("instance", "")

        # Domain routing rules
        infra_patterns = ("gpu", "ipmi", "fan", "temp", "vllm", "network", "node_")
        ops_patterns = ("ceph", "osd", "disk", "storage", "backup", "zfs")

        alertname_lower = alertname.lower()
        job_lower = job.lower()

        target = None
        if any(p in alertname_lower or p in job_lower for p in infra_patterns):
            target = "infra"
        elif any(p in alertname_lower or p in job_lower for p in ops_patterns):
            target = "ops"
        else:
            target = "prime"  # escalate unknown to user

        if target and target != self._claw_id:
            self._client.send(
                to_claw=target,
                msg_type="alert",
                payload={"alert": alert},
                priority=2 if alert.get("status") == "firing" else 1,
                wait=False,
            )
            log.info("Routed alert %s to %s", alertname, target)

        return target

    def _process_with_llm(self, msg) -> str:
        """Process a message through the LLM backend."""
        prompt = (
            f"Message from {msg.from_claw} ({msg.msg_type}, priority {msg.priority}):\n"
            f"{json.dumps(msg.payload, indent=2)}"
        )
        return self._process_with_llm_prompt(prompt)

    def _process_with_llm_prompt(self, prompt: str) -> str:
        """Send a prompt to the LLM backend and return the result text."""
        try:
            from engine import call_sync
            result = call_sync(prompt, timeout=300)
            return result.get("result", "No response from LLM")
        except Exception as e:
            log.error("LLM call failed: %s", e)
            return f"Error: {e}"

    async def run(self):
        """Main async entry point -- start server and run forever."""
        # Start mesh HTTP server
        await self._server.start()

        # Peer health check on startup
        self._registry.health_check_all(timeout=3.0)
        log.info("Mesh peers:\n%s", self._registry.status_summary())

        # Signal handling
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self._shutdown()))

        log.info("Claw-%s running. Waiting for messages.", self._claw_id)

        # Run periodic health checks
        while True:
            await asyncio.sleep(300)  # 5 minutes
            self._registry.health_check_all(timeout=3.0)

    async def _shutdown(self):
        """Graceful shutdown."""
        log.info("Claw-%s shutting down...", self._claw_id)
        await self._server.stop()
        asyncio.get_event_loop().stop()


def main():
    """Entry point for non-Prime Claw agents."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    runner = ClawRunner()
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
