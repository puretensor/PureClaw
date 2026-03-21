"""Mesh HTTP server — receives inter-Claw messages and webhooks.

Follows the git_push.py persistent HTTP server pattern but uses aiohttp
for native async support within the Nexus event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Callable

from aiohttp import web

from .message import ClawMessage

log = logging.getLogger("nexus.mesh")


class MeshServer:
    """HTTP server for the PureClaw mesh.

    Endpoints:
        GET  /health              - Liveness probe
        GET  /mesh/status         - Claw identity, uptime, queue depth
        POST /mesh/message        - Receive a ClawMessage
        POST /webhook/alertmanager - Receive Alertmanager webhooks
    """

    def __init__(
        self,
        claw_id: str,
        port: int = 9880,
        on_message: Callable[[ClawMessage], dict] | None = None,
        on_alert: Callable[[dict], dict] | None = None,
    ):
        self._claw_id = claw_id
        self._port = port
        self._on_message = on_message
        self._on_alert = on_alert
        self._start_time = time.monotonic()
        self._messages_received = 0
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None

    async def start(self):
        """Start the mesh HTTP server."""
        self._app = web.Application()
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/mesh/status", self._handle_status)
        self._app.router.add_post("/mesh/message", self._handle_message)
        self._app.router.add_post("/webhook/alertmanager", self._handle_alertmanager)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self._port)
        await site.start()
        log.info("Mesh server started: %s on port %d", self._claw_id, self._port)

    async def stop(self):
        """Stop the mesh HTTP server."""
        if self._runner:
            await self._runner.cleanup()
            log.info("Mesh server stopped: %s", self._claw_id)

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "claw_id": self._claw_id})

    async def _handle_status(self, request: web.Request) -> web.Response:
        uptime = int(time.monotonic() - self._start_time)
        return web.json_response({
            "claw_id": self._claw_id,
            "uptime_seconds": uptime,
            "messages_received": self._messages_received,
        })

    async def _handle_message(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        try:
            msg = ClawMessage.from_dict(data)
        except Exception as e:
            return web.json_response({"error": f"Invalid message: {e}"}, status=400)

        errors = msg.validate()
        if errors:
            return web.json_response({"error": "; ".join(errors)}, status=400)

        if msg.is_expired():
            return web.json_response({"error": "Message expired"}, status=410)

        self._messages_received += 1
        log.info(
            "Mesh recv: %s -> %s [%s] priority=%d id=%s",
            msg.from_claw, self._claw_id, msg.msg_type, msg.priority, msg.id,
        )

        if self._on_message:
            try:
                result = self._on_message(msg)
                return web.json_response({"status": "ok", "result": result})
            except Exception as e:
                log.error("Message handler error: %s", e)
                return web.json_response({"error": str(e)}, status=500)

        return web.json_response({"status": "received", "id": msg.id})

    async def _handle_alertmanager(self, request: web.Request) -> web.Response:
        """Receive Alertmanager webhook payload."""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        alerts = data.get("alerts", [])
        log.info("Alertmanager webhook: %d alerts", len(alerts))

        if self._on_alert:
            try:
                result = self._on_alert(data)
                return web.json_response({"status": "ok", "processed": len(alerts), "result": result})
            except Exception as e:
                log.error("Alert handler error: %s", e)
                return web.json_response({"error": str(e)}, status=500)

        return web.json_response({"status": "received", "alerts": len(alerts)})
