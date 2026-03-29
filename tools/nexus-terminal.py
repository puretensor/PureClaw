#!/usr/bin/env python3
"""PureClaw Cli — terminal interface to PureClaw over WebSocket.

Lightweight single-file client for the PureClaw agent platform. Connects to a
Nexus server over WebSocket and provides a streaming terminal interface with
slash command autocomplete, model switching, and live train departures.

Dependencies:
    pip install websockets
    pip install prompt_toolkit  # optional, for slash command autocomplete

Config: ~/.config/pureclaw/cli.conf (JSON)
    {"host": "localhost", "port": 9877, "token": "YOUR_TOKEN"}

Usage:
    python3 pureclaw-cli.py
    NEXUS_HOST=myhost NEXUS_PORT=9877 NEXUS_TOKEN=xyz python3 pureclaw-cli.py
"""

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Missing dependency: pip install websockets")
    sys.exit(1)

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import ANSI as PTK_ANSI
    from prompt_toolkit.patch_stdout import patch_stdout as ptk_patch_stdout
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VERSION = "1.2.5"
CONFIG_PATH = Path.home() / ".config" / "puretensor" / "nexus_terminal.conf"
DEFAULT_HOST = os.environ.get("NEXUS_HOST", "localhost")
DEFAULT_PORT = 30879


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception as e:
            print(f"Warning: failed to read config: {e}")
    return {}


CONFIG = load_config()
HOST = CONFIG.get("host", os.environ.get("NEXUS_HOST", DEFAULT_HOST))
PORT = int(CONFIG.get("port", os.environ.get("NEXUS_PORT", DEFAULT_PORT)))
TOKEN = CONFIG.get("token", os.environ.get("NEXUS_TOKEN", ""))

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

RESET = "\033[0m"
DIM = "\033[2m"
ITALIC = "\033[3m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
BOLD = "\033[1m"
CLEAR_LINE = "\033[2K\r"
BG_DARK = "\033[48;5;236m"  # dark grey background for separator


def _term_width() -> int:
    return shutil.get_terminal_size((80, 24)).columns


def _separator() -> str:
    """Full-width thin separator line."""
    return f"{DIM}{'─' * _term_width()}{RESET}"


def _model_label(backend: str, model: str) -> str:
    """Human-readable model label."""
    labels = {
        ("vllm", "sonnet"): "Nemotron Super (local GPUs)",
        ("bedrock_api", "opus"): "Claude Opus 4.6 (Bedrock)",
        ("bedrock_api", "sonnet"): "Claude Sonnet 4.6 (Bedrock)",
        ("ollama", "sonnet"): "Ollama (local)",
    }
    return labels.get((backend, model), f"{backend} / {model}")


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

LOGO_LINES = [
    f"    /\\___/\\ ",
    f"   ( {RED}o{RESET}{WHITE}   {RED}o{RESET}{WHITE} )",
    f"   (  =^=  )",
    f"    )     ( ",
    f"   (  |||  )",
    f"  ( ||| ||| )",
]

def print_banner(backend: str = "vllm", model: str = "sonnet", server_version: str = ""):
    """Print branded startup banner like Claude Code."""
    ver = server_version or VERSION
    model_text = _model_label(backend, model)
    w = _term_width()

    # Logo + title side by side
    logo_lines = LOGO_LINES
    title_lines = [
        f"{BOLD}{CYAN}PureClaw Cli{RESET} {DIM}v{ver}{RESET}",
        f"{model_text}",
        f"{DIM}{Path.home()}{RESET}",
    ]

    # Print logo lines with title alongside
    # Strip ANSI for width calculation
    import re
    _ansi_re = re.compile(r'\033\[[0-9;]*m')
    max_logo_w = max(len(_ansi_re.sub('', l)) for l in logo_lines) if logo_lines else 0
    for i, logo_line in enumerate(logo_lines):
        visible_len = len(_ansi_re.sub('', logo_line))
        pad = ' ' * (max_logo_w - visible_len)
        if i < len(title_lines):
            print(f"  {WHITE}{logo_line}{pad}{RESET}  {title_lines[i]}")
        else:
            print(f"  {WHITE}{logo_line}{pad}{RESET}")

    # Print any remaining title lines
    for i in range(len(logo_lines), len(title_lines)):
        print(f"  {''.ljust(max_logo_w)}  {title_lines[i]}")

    print(_separator())
    print()


# ---------------------------------------------------------------------------
# Slash command completion (requires prompt_toolkit)
# ---------------------------------------------------------------------------

COMMANDS = [
    ("/central", "Windsor Central \u2192 Slough \u2192 Paddington"),
    ("/riverside", "Windsor Riverside \u2192 Waterloo"),
    ("/waterloo", "Waterloo \u2192 Windsor Riverside"),
    ("/train", "Any route: /train <from> to <to>"),
    ("/nemotron", "Nemotron Super (local GPUs)"),
    ("/opus", "Claude Opus (Bedrock)"),
    ("/sonnet", "Claude Sonnet (Bedrock)"),
    ("/model", "Show current backend/model"),
    ("/new", "Start fresh session"),
    ("/status", "Current session info"),
    ("/sessions", "List recent sessions"),
    ("/help", "Show all commands"),
    ("/exit", "Disconnect"),
]

if HAS_PROMPT_TOOLKIT:
    class SlashCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/"):
                return
            for cmd, desc in COMMANDS:
                if cmd.startswith(text):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display=cmd,
                        display_meta=desc,
                    )


# ---------------------------------------------------------------------------
# WebSocket client
# ---------------------------------------------------------------------------

class NexusTerminal:
    def __init__(self):
        self.ws = None
        self._streaming = False
        self._last_tool_status = ""
        self._reconnect_delay = 1
        self._backend = "vllm"
        self._model = "sonnet"
        self._command_pending = False

    async def connect(self):
        uri = f"ws://{HOST}:{PORT}"
        self.ws = await websockets.connect(uri, ping_interval=30, ping_timeout=10)

        # Authenticate
        if TOKEN:
            await self.ws.send(json.dumps({"type": "auth", "token": TOKEN}))
            raw = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
            resp = json.loads(raw)
            if resp.get("type") == "error":
                print(f"{RED}Auth failed: {resp.get('message', 'unknown')}{RESET}")
                await self.ws.close()
                sys.exit(1)
            # Extract model info from auth_ok
            if resp.get("type") == "auth_ok":
                self._backend = resp.get("backend", self._backend)
                self._model = resp.get("model", self._model)

        self._reconnect_delay = 1
        return self.ws

    async def send_message(self, text: str):
        await self.ws.send(json.dumps({"type": "message", "text": text}))
        self._streaming = True

    async def send_command(self, cmd: str, args: str = ""):
        self._command_pending = True
        await self.ws.send(json.dumps({"type": "command", "cmd": cmd, "args": args}))

    def _print_separator(self):
        """Print separator after response, before next prompt."""
        print(f"\n{_separator()}")

    async def receive_loop(self):
        """Process incoming WebSocket events."""
        try:
            async for raw in self.ws:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type", "")

                if etype == "text_delta":
                    self._last_tool_status = ""
                    sys.stdout.write(event.get("text", ""))
                    sys.stdout.flush()

                elif etype == "tool_status":
                    status = event.get("status", "")
                    self._last_tool_status = status
                    print(f"  {DIM}{ITALIC}{status}{RESET}")

                elif etype == "stream_end":
                    self._last_tool_status = ""
                    self._streaming = False
                    self._print_separator()

                elif etype == "result":
                    text = event.get("text", "")
                    if text:
                        print(text)
                    self._streaming = False
                    self._print_separator()

                elif etype == "command_result":
                    text = event.get("text", "")
                    print(f"\n{CYAN}{text}{RESET}")
                    self._command_pending = False
                    # Update local model state if it looks like a switch
                    if "Switched to" in text:
                        if "Opus" in text:
                            self._backend, self._model = "bedrock_api", "opus"
                        elif "Sonnet" in text:
                            self._backend, self._model = "bedrock_api", "sonnet"
                        elif "Nemotron" in text:
                            self._backend, self._model = "vllm", "sonnet"
                        elif "Ollama" in text:
                            self._backend, self._model = "ollama", "sonnet"
                    self._print_separator()

                elif etype == "error":
                    msg = event.get("message", "Unknown error")
                    print(f"\n{RED}Error: {msg}{RESET}")
                    self._streaming = False
                    self._command_pending = False
                    self._print_separator()

                elif etype in ("pong", "auth_ok"):
                    pass

        except websockets.ConnectionClosed:
            if self._streaming:
                print(f"{RED}\nConnection lost during streaming{RESET}")
            raise

    async def _prompt_loop(self, pt_session, prompt_str):
        """Inner prompt loop — runs inside patch_stdout context."""
        while True:
            try:
                line = await pt_session.prompt_async(PTK_ANSI(prompt_str))
            except EOFError:
                return
            except KeyboardInterrupt:
                if self._streaming:
                    print(f"\n{DIM}(interrupt){RESET}")
                    continue
                return

            line = line.strip()
            if not line:
                continue

            if line.startswith("/"):
                parts = line[1:].split(None, 1)
                cmd = parts[0].lower() if parts else ""
                args = parts[1] if len(parts) > 1 else ""

                if cmd in ("quit", "exit"):
                    return

                await self.send_command(cmd, args)
                while self._command_pending:
                    await asyncio.sleep(0.05)
            else:
                await self.send_message(line)
                while self._streaming:
                    await asyncio.sleep(0.05)

    async def input_loop(self):
        """Read user input and send to server."""
        prompt_str = f"{BOLD}{GREEN}\u276f{RESET} "

        if HAS_PROMPT_TOOLKIT:
            pt_session = PromptSession(
                completer=SlashCompleter(),
                complete_while_typing=True,
            )
            # patch_stdout keeps the prompt fixed at the bottom —
            # all stdout writes from the receive_loop scroll above it
            with ptk_patch_stdout(raw=True):
                await self._prompt_loop(pt_session, prompt_str)
        else:
            while True:
                try:
                    loop = asyncio.get_event_loop()
                    line = await loop.run_in_executor(None, lambda: input(prompt_str))
                except EOFError:
                    break
                except KeyboardInterrupt:
                    if self._streaming:
                        print(f"\n{DIM}(interrupt){RESET}")
                        continue
                    break

                line = line.strip()
                if not line:
                    continue

                if line.startswith("/"):
                    parts = line[1:].split(None, 1)
                    cmd = parts[0].lower() if parts else ""
                    args = parts[1] if len(parts) > 1 else ""

                    if cmd in ("quit", "exit"):
                        break

                    await self.send_command(cmd, args)
                    while self._command_pending:
                        await asyncio.sleep(0.05)
                else:
                    await self.send_message(line)
                    while self._streaming:
                        await asyncio.sleep(0.05)

    async def run(self):
        """Main loop with auto-reconnect."""
        while True:
            try:
                await self.connect()
                print_banner(self._backend, self._model)

                recv_task = asyncio.create_task(self.receive_loop())
                input_task = asyncio.create_task(self.input_loop())

                done, pending = await asyncio.wait(
                    {recv_task, input_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

                if input_task in done:
                    break

                raise websockets.ConnectionClosed(None, None)

            except (websockets.ConnectionClosed, ConnectionRefusedError, OSError):
                if self.ws:
                    try:
                        await self.ws.close()
                    except Exception:
                        pass
                    self.ws = None

                print(f"\n{DIM}Reconnecting in {self._reconnect_delay}s...{RESET}")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30)

            except KeyboardInterrupt:
                break

        if self.ws:
            await self.ws.close()
        print(f"\n{DIM}Disconnected.{RESET}")


def main():
    if not TOKEN:
        print(f"{RED}No token configured.{RESET}")
        print(f"Set NEXUS_TOKEN env var or create {CONFIG_PATH}:")
        print(f'  {{"host": "{DEFAULT_HOST}", "port": {DEFAULT_PORT}, "token": "YOUR_TOKEN"}}')
        sys.exit(1)

    terminal = NexusTerminal()
    try:
        asyncio.run(terminal.run())
    except KeyboardInterrupt:
        print(f"\n{DIM}Bye.{RESET}")


if __name__ == "__main__":
    main()
