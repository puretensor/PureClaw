"""Health endpoint for hal-mail K8s probes."""

import logging
import os
import time

from aiohttp import web

log = logging.getLogger("hal-mail")

HEALTH_PORT = int(os.environ.get("HEALTH_PORT", "9877"))


class HealthServer:
    """Lightweight HTTP health server for K8s liveness/readiness probes."""

    def __init__(self):
        self.last_poll_time: float = 0
        self.emails_processed: int = 0
        self.poll_count: int = 0
        self.errors: int = 0
        self._app = web.Application()
        self._app.router.add_get("/healthz", self._healthz)
        self._runner: web.AppRunner | None = None

    async def start(self):
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", HEALTH_PORT)
        await site.start()
        log.info("Health server listening on port %d", HEALTH_PORT)

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()

    async def _healthz(self, request: web.Request) -> web.Response:
        now = time.time()
        # Unhealthy if no successful poll in 5 minutes (> 2x poll interval)
        healthy = (now - self.last_poll_time) < 300 if self.last_poll_time else True
        body = {
            "status": "ok" if healthy else "degraded",
            "last_poll_time": self.last_poll_time,
            "last_poll_ago_s": round(now - self.last_poll_time, 1) if self.last_poll_time else None,
            "emails_processed": self.emails_processed,
            "poll_count": self.poll_count,
            "errors": self.errors,
        }
        return web.json_response(body, status=200 if healthy else 503)

    def record_poll(self):
        self.last_poll_time = time.time()
        self.poll_count += 1

    def record_email(self):
        self.emails_processed += 1

    def record_error(self):
        self.errors += 1
