"""Terminal WebSocket channel — direct CLI access to Nexus over Tailscale.

Starts a websockets server on TERMINAL_WS_PORT (default 9877). Auth via
bearer token on handshake. Fixed chat_id offset 700B (same pattern as
WhatsApp 800B / Email 900B).
"""

import asyncio
import json
import logging

import websockets

from channels.base import Channel
from channels.terminal.streaming import TerminalStreamingEditor
from db import get_session, upsert_session, update_model, delete_session, get_lock
from engine import call_streaming, get_model_display
from config import (
    AGENT_NAME,
    TIMEOUT,
    log,
)

from config import (
    TERMINAL_WS_PORT,
    TERMINAL_API_KEY,
    TERMINAL_CHAT_ID_OFFSET,
)

# Fixed chat ID for terminal sessions (single user)
TERMINAL_CHAT_ID = TERMINAL_CHAT_ID_OFFSET + 1

TERMINAL_SYSTEM_PROMPT = (
    f"IMPORTANT IDENTITY OVERRIDE: You are {AGENT_NAME}, NOT Claude Code. "
    f"You are {AGENT_NAME}, responding via terminal. "
    f"Full infrastructure, email, calendar, and tool access available. "
    f"Formatting: Plain text with markdown. Terminal supports ANSI rendering."
)


async def _send_json(ws, event: dict):
    """Send a JSON event, swallowing errors on closed connections."""
    try:
        await ws.send(json.dumps(event))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Train helpers — same data path as Telegram dispatcher (router.py)
# ---------------------------------------------------------------------------

def _fmt_departures(departures: list[dict]) -> str:
    """Format a departure list into a text table."""
    lines = [f"{'Time':<6} {'Exp':<8} {'Plat':<5} {'Status'}"]
    lines.append("-" * 32)
    for d in departures:
        lines.append(
            f"{d['scheduled']:<6} {d.get('expected', '-'):<8} "
            f"{d.get('platform', '-'):<5} {d['status']}"
        )
    if not departures:
        lines.append("  No departures found.")
    return "\n".join(lines)


async def _fetch_departures(from_crs: str, to_crs: str, count: int = 6) -> dict:
    """Fetch departures — same chain as Telegram's handle_trains().

    1. Try Darwin live state (Kafka push port)
    2. Fall back to Huxley2 REST API (free, no token)

    Returns {"origin": str, "destination": str, "departures": [...]}.
    """
    from dispatcher.apis.trains import STATION_NAMES

    # Try Darwin live state first (same as dispatcher/router.py:163-168)
    try:
        from dispatcher.apis.darwin import fetch_darwin_departures
        return await fetch_darwin_departures(from_crs, to_crs, count=count)
    except Exception:
        pass

    # Fallback: Huxley2 free endpoint (same as dispatcher/apis/commute.py)
    from dispatcher.apis.commute import _fetch_huxley
    departures = await _fetch_huxley(from_crs, to_crs, count=count)

    return {
        "origin": STATION_NAMES.get(from_crs, from_crs),
        "destination": STATION_NAMES.get(to_crs, to_crs),
        "departures": departures,
    }


async def _handle_trains_route(ws, from_crs: str, to_crs: str):
    """Fetch and display departures for a single A->B route."""
    try:
        data = await _fetch_departures(from_crs, to_crs)
        text = f"{data['origin']} -> {data['destination']}\n{_fmt_departures(data['departures'])}"
        await _send_json(ws, {"type": "command_result", "text": text})
    except Exception as e:
        await _send_json(ws, {"type": "error", "message": f"Train lookup failed: {e}"})


async def _handle_trains_central(ws):
    """Full Windsor Central commute — same as Telegram /commute central.

    Shuttle (WNC->SLO) + Elizabeth line (SLO->PAD XR) + GWR (SLO->PAD GW).
    """
    try:
        lines = ["Windsor Central -> Paddington"]

        # Shuttle: Windsor Central -> Slough
        shuttle = await _fetch_departures("WNC", "SLO", count=3)
        lines.append(f"\nWindsor Central -> Slough (shuttle)")
        lines.append(_fmt_departures(shuttle["departures"]))

        # Slough -> Paddington: try Darwin state for TOC split, fall back to Huxley2
        slo_data = await _fetch_departures("SLO", "PAD", count=12)
        all_deps = slo_data["departures"]
        xr = [d for d in all_deps if d.get("toc") == "XR"][:3]
        gw = [d for d in all_deps if d.get("toc") == "GW"][:3]

        lines.append(f"\nSlough -> Paddington (Elizabeth line)")
        lines.append(_fmt_departures(xr))

        lines.append(f"\nSlough -> Paddington (GWR)")
        lines.append(_fmt_departures(gw))

        await _send_json(ws, {"type": "command_result", "text": "\n".join(lines)})
    except Exception as e:
        await _send_json(ws, {"type": "error", "message": f"Train lookup failed: {e}"})


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------

async def _handle_command(ws, cmd: str, args: str):
    """Handle a terminal command."""
    chat_id = TERMINAL_CHAT_ID

    if cmd == "new":
        session = get_session(chat_id)
        model = session["model"] if session else "sonnet"
        delete_session(chat_id)
        update_model(chat_id, model)
        await _send_json(ws, {
            "type": "command_result",
            "text": "Session cleared. Next message starts fresh.",
        })

    # --- Model switching: Bedrock for Claude, vLLM for Nemotron ---

    elif cmd == "opus":
        import config
        from backends import reset_backend
        old_backend = config.ENGINE_BACKEND
        if old_backend != "bedrock_api":
            config.ENGINE_BACKEND = "bedrock_api"
            reset_backend()
            delete_session(chat_id)  # clear history on backend switch
        update_model(chat_id, "opus")
        await _send_json(ws, {
            "type": "command_result",
            "text": "Switched to Claude Opus (Bedrock).",
        })

    elif cmd == "sonnet":
        import config
        from backends import reset_backend
        old_backend = config.ENGINE_BACKEND
        if old_backend != "bedrock_api":
            config.ENGINE_BACKEND = "bedrock_api"
            reset_backend()
            delete_session(chat_id)
        update_model(chat_id, "sonnet")
        await _send_json(ws, {
            "type": "command_result",
            "text": "Switched to Claude Sonnet (Bedrock).",
        })

    elif cmd in ("nemotron", "vllm"):
        import config
        from backends import reset_backend
        old_backend = config.ENGINE_BACKEND
        if old_backend != "vllm":
            config.ENGINE_BACKEND = "vllm"
            reset_backend()
            delete_session(chat_id)
        update_model(chat_id, "sonnet")
        await _send_json(ws, {
            "type": "command_result",
            "text": "Switched to Nemotron Super (vLLM, local GPUs).",
        })

    elif cmd == "ollama":
        import config
        from backends import reset_backend
        old_backend = config.ENGINE_BACKEND
        if old_backend != "ollama":
            config.ENGINE_BACKEND = "ollama"
            reset_backend()
            delete_session(chat_id)
        update_model(chat_id, "sonnet")
        await _send_json(ws, {
            "type": "command_result",
            "text": f"Switched to {get_model_display('sonnet')} (local Ollama).",
        })

    elif cmd == "model":
        if args and args in ("opus", "sonnet", "ollama", "nemotron", "vllm"):
            await _handle_command(ws, args, "")
        else:
            import config
            session = get_session(chat_id)
            model = session["model"] if session else "sonnet"
            backend = config.ENGINE_BACKEND
            await _send_json(ws, {
                "type": "command_result",
                "text": f"Backend: {backend}, Model: {model}",
            })

    # --- Train commands (Darwin state -> Huxley2 fallback) ---

    elif cmd == "central":
        await _handle_trains_central(ws)

    elif cmd == "riverside":
        await _handle_trains_route(ws, "WNR", "WAT")

    elif cmd in ("waterloo-riverside", "waterloo"):
        await _handle_trains_route(ws, "WAT", "WNR")

    elif cmd == "train":
        from dispatcher.apis.trains import resolve_station
        if not args:
            await _send_json(ws, {"type": "error", "message": "Usage: /train <from> to <to>"})
            return
        if " to " in args.lower():
            fr, _, to = args.partition(" to ")
        else:
            parts = args.split(None, 1)
            fr = parts[0]
            to = parts[1] if len(parts) > 1 else ""
        from_crs = resolve_station(fr.strip()) or fr.strip().upper()
        to_crs = (resolve_station(to.strip()) or to.strip().upper()) if to.strip() else None
        if not to_crs:
            await _send_json(ws, {"type": "error", "message": "Usage: /train <from> to <to>"})
        else:
            await _handle_trains_route(ws, from_crs, to_crs)

    # --- Session info ---

    elif cmd == "status":
        session = get_session(chat_id)
        if session is None or session["session_id"] is None:
            text = "No active session. Send a message to start one."
        else:
            import config
            name = session.get("name", "default")
            summary = session.get("summary")
            text = (
                f"Session: {session['session_id'][:12]}... (name: {name})\n"
                f"Backend: {config.ENGINE_BACKEND}, Model: {session['model']}\n"
                f"Messages: {session['message_count']}\n"
            )
            if summary:
                text += f"Summary: {summary}\n"
            text += f"Started: {session['created_at']}"
        await _send_json(ws, {"type": "command_result", "text": text})

    elif cmd == "sessions":
        from db import list_sessions
        sessions = list_sessions()
        if not sessions:
            text = "No saved sessions."
        else:
            lines = []
            for s in sessions[:10]:
                label = s.get("name") or s["session_id"][:12]
                lines.append(f"  {label} ({s['model']}, {s['message_count']} msgs)")
            text = "Recent sessions:\n" + "\n".join(lines)
        await _send_json(ws, {"type": "command_result", "text": text})

    elif cmd == "help":
        await _send_json(ws, {
            "type": "command_result",
            "text": (
                "Trains:\n"
                "  /central  - Windsor Central -> Slough -> Paddington\n"
                "  /riverside - Windsor Riverside -> Waterloo\n"
                "  /waterloo - Waterloo -> Windsor Riverside\n"
                "  /train <from> to <to> - Any route\n"
                "\n"
                "Models:\n"
                "  /nemotron - Nemotron Super (local GPUs)\n"
                "  /opus     - Claude Opus (Bedrock)\n"
                "  /sonnet   - Claude Sonnet (Bedrock)\n"
                "  /ollama   - Local Ollama\n"
                "  /model    - Show current backend/model\n"
                "\n"
                "Session:\n"
                "  /new      - Start fresh session\n"
                "  /status   - Current session info\n"
                "  /sessions - List recent sessions\n"
                "  /help     - This message\n"
                "  Ctrl-C    - Exit"
            ),
        })

    else:
        await _send_json(ws, {
            "type": "error",
            "message": f"Unknown command: /{cmd}. Type /help for available commands.",
        })


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------

async def _handle_message(ws, text: str):
    """Handle a regular message — send to engine with streaming."""
    chat_id = TERMINAL_CHAT_ID

    lock = get_lock(chat_id)
    if lock.locked():
        await _send_json(ws, {
            "type": "error",
            "message": "Still processing previous message -- please wait.",
        })
        return

    async with lock:
        session = get_session(chat_id)
        model = session["model"] if session else "sonnet"
        session_id = session["session_id"] if session else None
        msg_count = session["message_count"] if session else 0

        for attempt in range(2):
            try:
                editor = TerminalStreamingEditor(ws)
                data = await call_streaming(
                    text,
                    session_id,
                    model,
                    streaming_editor=editor,
                    extra_system_prompt=TERMINAL_SYSTEM_PROMPT,
                    chat_id=chat_id,
                )

                # Check if primary failed and fell over (stale history)
                if data.get("_failover") and session_id and attempt == 0:
                    log.info("Terminal: failover detected with existing session, flushing and retrying on clean session")
                    delete_session(chat_id)
                    update_model(chat_id, model)
                    session_id = None
                    msg_count = 0
                    continue

                result_text = data.get("result", "")
                new_session_id = data.get("session_id", session_id)

                upsert_session(chat_id, new_session_id, model, msg_count + 1)

                await editor.finalize()

                await _send_json(ws, {
                    "type": "result",
                    "text": result_text if not editor.text else "",
                    "session_id": new_session_id or "",
                })
                break

            except TimeoutError as e:
                log.error("Terminal timeout: %s", e)
                await _send_json(ws, {
                    "type": "error",
                    "message": f"Timed out after {TIMEOUT}s. Try a simpler query or /new.",
                })
                break
            except Exception as e:
                log.exception("Terminal handler error")
                await _send_json(ws, {
                    "type": "error",
                    "message": f"Error: {e}",
                })
                break


# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------

async def _ws_handler(ws):
    """Handle a single WebSocket connection."""
    log.info("Terminal client connected from %s", ws.remote_address)
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send_json(ws, {"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "command":
                cmd = msg.get("cmd", "").strip().lower()
                args = msg.get("args", "").strip()
                await _handle_command(ws, cmd, args)

            elif msg_type == "message":
                text = msg.get("text", "").strip()
                if text:
                    await _handle_message(ws, text)

            elif msg_type == "ping":
                await _send_json(ws, {"type": "pong"})

            else:
                await _send_json(ws, {
                    "type": "error",
                    "message": f"Unknown event type: {msg_type}",
                })

    except websockets.ConnectionClosed:
        pass
    except Exception as e:
        log.warning("Terminal WebSocket error: %s", e)
    finally:
        log.info("Terminal client disconnected")


class TerminalChannel(Channel):
    """WebSocket server for direct terminal access to Nexus."""

    def __init__(self):
        self._server = None
        self._task = None

    async def start(self):
        """Start the WebSocket server."""

        async def auth_handler(ws):
            """Authenticate then delegate to _ws_handler."""
            if TERMINAL_API_KEY:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    auth = json.loads(raw)
                    if auth.get("type") != "auth" or auth.get("token") != TERMINAL_API_KEY:
                        await _send_json(ws, {"type": "error", "message": "Unauthorized"})
                        await ws.close(4001, "Unauthorized")
                        return
                    import config as _cfg
                    await _send_json(ws, {
                        "type": "auth_ok",
                        "backend": _cfg.ENGINE_BACKEND,
                        "model": get_session(TERMINAL_CHAT_ID)["model"] if get_session(TERMINAL_CHAT_ID) else "sonnet",
                        "version": "1.0.0",
                    })
                except (asyncio.TimeoutError, json.JSONDecodeError, Exception):
                    await ws.close(4001, "Auth timeout")
                    return

            await _ws_handler(ws)

        self._server = await websockets.serve(
            auth_handler,
            "0.0.0.0",
            TERMINAL_WS_PORT,
            ping_interval=30,
            ping_timeout=10,
        )
        log.info("Terminal WebSocket server listening on port %d", TERMINAL_WS_PORT)

    async def stop(self):
        """Stop the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        log.info("Terminal channel stopped.")
