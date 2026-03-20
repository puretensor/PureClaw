"""Terminal WebSocket StreamingEditor — sends deltas over WebSocket as JSON events.

Same interface as DiscordStreamingEditor (channels/discord/streaming.py) but
with no rate limiting or message splitting — the terminal handles everything.
"""

import json
import logging

log = logging.getLogger("nexus")


class TerminalStreamingEditor:
    """Streams LLM output over a WebSocket connection as JSON events.

    Events sent:
        {"type": "text_delta", "text": "..."}
        {"type": "tool_status", "status": "..."}
        {"type": "stream_end"}
    """

    def __init__(self, ws):
        self.ws = ws
        self.text = ""
        self._closed = False

    async def _send(self, event: dict):
        if self._closed:
            return
        try:
            await self.ws.send(json.dumps(event))
        except Exception:
            self._closed = True

    async def add_text(self, delta: str):
        """Send a text delta immediately — no rate limiting needed."""
        self.text += delta
        await self._send({"type": "text_delta", "text": delta})

    async def add_tool_status(self, status: str):
        """Send tool-use progress to the terminal."""
        await self._send({"type": "tool_status", "status": status})

    async def finalize(self):
        """Signal end of stream."""
        await self._send({"type": "stream_end"})
        return []
