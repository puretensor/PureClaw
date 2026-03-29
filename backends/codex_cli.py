"""OpenAI Codex CLI backend — shells out to the Codex CLI binary.

Supports session resume via `codex exec resume <thread_id> <prompt>`.
First message uses `codex exec <prompt>`, subsequent messages in the same
session use `codex exec resume <thread_id> <prompt>` to maintain context.

CLI flags verified against `codex exec --help` / `codex exec resume --help`.
"""

import asyncio
import json
import logging
import os
import subprocess

log = logging.getLogger("nexus")


class CodexCLIBackend:
    """Backend that shells out to the OpenAI Codex CLI for full agentic tool use."""

    def __init__(self):
        from config import CODEX_BIN, CODEX_MODEL, CODEX_CWD
        self._bin = CODEX_BIN
        self._model = CODEX_MODEL
        self._cwd = CODEX_CWD

    def get_model_display(self, model: str) -> str:
        return "Codex"

    @staticmethod
    def _write_instructions(
        system_prompt: str | None = None,
        memory_context: str | None = None,
        extra_system_prompt: str | None = None,
    ) -> str:
        """Write instructions to AGENTS.md in CWD before each Codex call."""
        parts = []
        if system_prompt:
            parts.append(system_prompt)
        if memory_context:
            parts.append(memory_context)
        if extra_system_prompt:
            parts.append(extra_system_prompt)

        from config import CODEX_CWD
        cwd = CODEX_CWD or "/app"
        agents_path = os.path.join(cwd, "AGENTS.md")
        content = "\n\n".join(parts) if parts else ""
        with open(agents_path, "w") as f:
            f.write(content)
        return agents_path

    def _run_with_instructions(
        self,
        runner,
        system_prompt: str | None = None,
        memory_context: str | None = None,
        extra_system_prompt: str | None = None,
    ):
        """Temporarily write AGENTS.md for a single Codex invocation."""
        from config import CODEX_CWD

        cwd = CODEX_CWD or "/app"
        agents_path = os.path.join(cwd, "AGENTS.md")
        had_existing = os.path.exists(agents_path)
        previous = ""
        if had_existing:
            try:
                with open(agents_path, "r") as f:
                    previous = f.read()
            except Exception:
                previous = ""

        if extra_system_prompt is None:
            self._write_instructions(system_prompt, memory_context)
        else:
            if extra_system_prompt is None:
                self._write_instructions(system_prompt, memory_context)
            else:
                self._write_instructions(system_prompt, memory_context, extra_system_prompt)
        try:
            return runner()
        finally:
            try:
                if had_existing:
                    with open(agents_path, "w") as f:
                        f.write(previous)
                else:
                    os.unlink(agents_path)
            except FileNotFoundError:
                pass

    def _build_cmd(
        self,
        message: str,
        *,
        session_id: str | None = None,
    ) -> list[str]:
        """Build the CLI command.

        New session:    codex exec <prompt> [flags]
        Resume session: codex exec resume <thread_id> <prompt> [flags]

        Note: `codex exec resume` does NOT accept -C/--cd (inherits from
        the original session) or --skip-git-repo-check.
        """
        if session_id:
            # Resume existing session — limited flag set
            cmd = [self._bin, "exec", "resume", session_id, message,
                   "--json", "--dangerously-bypass-approvals-and-sandbox"]
            if self._model:
                cmd.extend(["-m", self._model])
        else:
            # New session — full flag set
            cmd = [self._bin, "exec", message,
                   "--json", "--dangerously-bypass-approvals-and-sandbox",
                   "--skip-git-repo-check"]
            if self._model:
                cmd.extend(["-m", self._model])
            if self._cwd:
                cmd.extend(["-C", self._cwd])
        return cmd

    @property
    def name(self) -> str:
        return "codex_cli"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_tools(self) -> bool:
        return True

    @property
    def supports_sessions(self) -> bool:
        return True

    def call_sync(
        self,
        prompt: str,
        *,
        model: str = "sonnet",
        session_id: str | None = None,
        timeout: int = 300,
        system_prompt: str | None = None,
        memory_context: str | None = None,
        tool_context=None,
    ) -> dict:
        """Synchronous Codex CLI call.

        Returns {"result": str, "session_id": str | None}
        """
        cmd = self._build_cmd(prompt, session_id=session_id)

        mode = "resume" if session_id else "new"
        log.info("Codex CLI (sync, %s): %s", mode, " ".join(cmd[:6]) + " ...")

        try:
            result = self._run_with_instructions(
                lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=timeout),
                system_prompt,
                memory_context,
            )
        except FileNotFoundError:
            return {
                "result": f"Codex CLI not found at {self._bin}. Install it first.",
                "session_id": None,
            }
        except subprocess.TimeoutExpired:
            return {"result": f"Codex CLI timed out after {timeout}s", "session_id": session_id}

        if result.returncode != 0:
            return {
                "result": f"Codex CLI error (exit {result.returncode}): {result.stderr[:500]}",
                "session_id": session_id,
            }

        return _parse_codex_jsonl(result.stdout, fallback_session_id=session_id)

    async def call_streaming(
        self,
        message: str,
        *,
        session_id: str | None = None,
        model: str = "sonnet",
        on_progress=None,
        streaming_editor=None,
        system_prompt: str | None = None,
        memory_context: str | None = None,
        extra_system_prompt: str | None = None,
        tool_context=None,
    ) -> dict:
        """Async streaming Codex CLI call.

        Returns {"result": str, "session_id": str | None, "written_files": list}
        """
        cmd = self._build_cmd(message, session_id=session_id)

        mode = "resume" if session_id else "new"
        log.info("Codex CLI (streaming, %s): %s", mode, " ".join(cmd[:6]) + " ...")

        from config import CODEX_CWD

        cwd = CODEX_CWD or "/app"
        agents_path = os.path.join(cwd, "AGENTS.md")
        had_existing = os.path.exists(agents_path)
        previous = ""
        if had_existing:
            try:
                with open(agents_path, "r") as f:
                    previous = f.read()
            except Exception:
                previous = ""
        try:
            self._write_instructions(system_prompt, memory_context, extra_system_prompt)

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    limit=10 * 1024 * 1024,
                )
            except FileNotFoundError:
                return {
                    "result": f"Codex CLI not found at {self._bin}. Install it first.",
                    "session_id": None,
                    "written_files": [],
                }

            try:
                data = await asyncio.wait_for(
                    _read_codex_stream(
                        proc,
                        on_progress=on_progress,
                        streaming_editor=streaming_editor,
                        fallback_session_id=session_id,
                    ),
                    timeout=1800,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                raise TimeoutError("Codex CLI timed out after 1800s")
            except RuntimeError as e:
                await proc.wait()
                stderr_bytes = await proc.stderr.read() if proc.stderr else b""
                err = stderr_bytes.decode().strip()
                log.warning("Codex CLI stream empty, stderr: %s", err[:500])
                if "401" in err or "Unauthorized" in err:
                    return {
                        "result": "Codex CLI auth failed (401). Run `codex login` to re-authenticate.",
                        "session_id": session_id,
                        "written_files": [],
                    }
                return {
                    "result": f"Codex CLI error: {err[:500] or str(e)}",
                    "session_id": session_id,
                    "written_files": [],
                }

            await proc.wait()

            if proc.returncode != 0:
                stderr_bytes = await proc.stderr.read() if proc.stderr else b""
                err = stderr_bytes.decode().strip()
                log.warning("Codex CLI exited %d (stream mode), stderr: %s", proc.returncode, err[:500])
                if data and data.get("result"):
                    return data
                raise RuntimeError(f"Codex CLI exited {proc.returncode}: {err}")

            return data
        finally:
            try:
                if had_existing:
                    with open(agents_path, "w") as f:
                        f.write(previous)
                else:
                    os.unlink(agents_path)
            except FileNotFoundError:
                pass


def _extract_thread_id(event: dict, fallback: str | None = None) -> str | None:
    """Extract thread_id from a thread.started event."""
    if event.get("type") == "thread.started":
        tid = event.get("thread_id")
        if tid:
            return tid
    return fallback


def _parse_codex_jsonl(stdout: str, fallback_session_id: str | None = None) -> dict:
    """Parse JSONL output from `codex exec --json` and extract result + thread_id."""
    last_text = ""
    session_id = fallback_session_id

    for line in stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type", "")

        # Capture thread_id for session continuity
        if event_type == "thread.started":
            tid = event.get("thread_id")
            if tid:
                session_id = tid

        # v0.101+ format
        elif event_type == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                text = item.get("text", "")
                if text:
                    last_text = text

        # Legacy format
        elif event_type == "message" and event.get("role") == "assistant":
            content = event.get("content", "")
            if isinstance(content, str) and content:
                last_text = content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                        last_text = part.get("text", last_text)
        elif event_type in ("output_text", "text"):
            last_text = event.get("text", last_text)

    return {
        "result": last_text.strip()[:4000] if last_text else stdout.strip()[:4000] or "(no output)",
        "session_id": session_id,
    }


async def _read_codex_stream(
    proc,
    on_progress=None,
    streaming_editor=None,
    fallback_session_id: str | None = None,
) -> dict:
    """Read JSONL output from Codex CLI line by line.

    Captures thread_id from thread.started events for session persistence.
    """
    written_files = []
    streamed_text = ""
    session_id = fallback_session_id

    while True:
        try:
            raw_line = await proc.stdout.readline()
        except (ValueError, asyncio.LimitOverrunError) as e:
            log.warning("Codex stream line too large, skipping: %s", e)
            try:
                proc.stdout._buffer.clear()
            except Exception:
                pass
            continue

        if not raw_line:
            break
        line = raw_line.decode().strip()
        if not line:
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            log.debug("Non-JSON codex stream line: %s", line[:200])
            continue

        event_type = event.get("type", "")

        # Capture thread_id for session continuity
        if event_type == "thread.started":
            tid = event.get("thread_id")
            if tid:
                session_id = tid
                log.info("Codex thread_id captured: %s", tid)

        # --- v0.101+ format: item.completed / item.started ---
        elif event_type == "item.completed":
            item = event.get("item", {})
            item_type = item.get("type", "")
            if item_type == "agent_message":
                text = item.get("text", "")
                if text:
                    streamed_text += text
                    if streaming_editor:
                        await streaming_editor.add_text(text)
            elif item_type == "command_execution":
                cmd_str = item.get("command", "")
                status = f"Ran: {cmd_str}" if cmd_str else "Command completed"
                if streaming_editor:
                    await streaming_editor.add_tool_status(status)
                elif on_progress:
                    await on_progress(status)

        elif event_type == "item.started":
            item = event.get("item", {})
            item_type = item.get("type", "")
            if item_type == "command_execution":
                cmd_str = item.get("command", "")
                status = f"Running: {cmd_str}" if cmd_str else "Running command..."
                if streaming_editor:
                    await streaming_editor.add_tool_status(status)
                elif on_progress:
                    await on_progress(status)

        # --- Legacy format (pre-v0.101) ---
        elif event_type == "message" and event.get("role") == "assistant":
            content = event.get("content", "")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                        text = part.get("text", "")
            if text:
                streamed_text += text
                if streaming_editor:
                    await streaming_editor.add_text(text)

        elif event_type in ("output_text", "text"):
            text = event.get("text", "")
            if text:
                streamed_text += text
                if streaming_editor:
                    await streaming_editor.add_text(text)

        elif event_type in ("function_call", "tool_call", "function_call_output"):
            tool_name = event.get("name", event.get("function", ""))
            status = f"Using tool: {tool_name}" if tool_name else "Running tool..."
            if streaming_editor:
                await streaming_editor.add_tool_status(status)
            elif on_progress:
                await on_progress(status)

    if not streamed_text.strip():
        raise RuntimeError("No output from Codex CLI stream")

    return {
        "result": streamed_text,
        "session_id": session_id,
        "written_files": written_files,
    }
