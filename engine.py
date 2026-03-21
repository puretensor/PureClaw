"""Engine — LLM backend facade, stream reader, and message splitting.

Public API: call_sync(), call_streaming(), split_message()
Backend selection: ENGINE_BACKEND env var (default: claude_code)

Utilities (_read_stream, _format_tool_status, split_message) remain here
for backward compatibility and are used by the claude_code backend.
"""

import asyncio
import json
import logging

from config import SYSTEM_PROMPT, CLAUDE_MODEL

try:
    from memory import get_memories_for_injection, get_shared_context
except ImportError:
    get_memories_for_injection = None
    get_shared_context = None

log = logging.getLogger("nexus")


# ---------------------------------------------------------------------------
# Message splitting for Telegram's 4096-char limit
# ---------------------------------------------------------------------------


def split_message(text: str, limit: int = 4000) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Find a newline near the limit to split cleanly
        idx = text.rfind("\n", 0, limit)
        if idx == -1:
            # No newline — split at limit
            idx = limit
        chunks.append(text[:idx])
        text = text[idx:].lstrip("\n")
    return chunks


# ---------------------------------------------------------------------------
# Tool status formatting
# ---------------------------------------------------------------------------


def _format_tool_status(tool_name: str, tool_input: dict) -> str:
    """Map a tool_use event to a human-readable status line."""
    if tool_name in ("Bash", "bash"):
        cmd = tool_input.get("command", "")
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        return f"Running: {cmd}"
    elif tool_name in ("Read", "read_file"):
        return f"Reading: {tool_input.get('file_path', '?')}"
    elif tool_name in ("Edit", "edit_file"):
        return f"Editing: {tool_input.get('file_path', '?')}"
    elif tool_name in ("Write", "write_file"):
        return f"Writing: {tool_input.get('file_path', '?')}"
    elif tool_name in ("Glob", "glob"):
        return f"Searching files: {tool_input.get('pattern', '?')}"
    elif tool_name in ("Grep", "grep"):
        return f"Searching content: {tool_input.get('pattern', '?')}"
    elif tool_name == "WebFetch":
        url = tool_input.get("url", "?")
        if len(url) > 60:
            url = url[:57] + "..."
        return f"Fetching: {url}"
    elif tool_name == "WebSearch":
        return f"Searching web: {tool_input.get('query', '?')}"
    elif tool_name == "Task":
        desc = tool_input.get("description", "")
        if desc:
            return f"Spawning agent: {desc}"
        return "Spawning agent..."
    else:
        return f"Using tool: {tool_name}"


# ---------------------------------------------------------------------------
# Stream reader (used by claude_code backend)
# ---------------------------------------------------------------------------


async def _read_stream(proc, on_progress=None, streaming_editor=None) -> dict:
    """Read stream-json output line by line.

    If streaming_editor is provided, streams text deltas to it in real-time.
    Otherwise falls back to on_progress for tool events only.

    Returns the final result dict with 'result', 'session_id', and 'written_files'.
    """
    result = None
    written_files = []
    streamed_text = ""
    while True:
        try:
            raw_line = await proc.stdout.readline()
        except (ValueError, asyncio.LimitOverrunError) as e:
            log.warning("Stream line too large, skipping: %s", e)
            try:
                proc.stdout._buffer.clear()
            except Exception:
                pass
            continue
        if not raw_line:
            break  # EOF
        line = raw_line.decode().strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            log.debug("Non-JSON stream line: %s", line[:200])
            continue

        event_type = event.get("type")

        # Streaming text deltas (requires --include-partial-messages)
        if event_type == "stream_event" and streaming_editor:
            inner = event.get("event", {})
            inner_type = inner.get("type")
            if inner_type == "content_block_delta":
                delta = inner.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        streamed_text += text
                        await streaming_editor.add_text(text)

        # Tool-use events from full assistant messages
        elif event_type == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    tool_name = block.get("name", "")
                    tool_input = block.get("input", {})
                    status = _format_tool_status(tool_name, tool_input)
                    if streaming_editor:
                        await streaming_editor.add_tool_status(status)
                    elif on_progress:
                        await on_progress(status)
                    # Track files written by Claude
                    if tool_name == "Write":
                        fpath = tool_input.get("file_path", "")
                        if fpath:
                            written_files.append(fpath)

        elif event_type == "result":
            result = {
                "result": event.get("result", ""),
                "session_id": event.get("session_id"),
                "written_files": written_files,
            }

    if result is None:
        # Try to read stderr for context on why the stream ended without a result
        stderr_text = ""
        try:
            stderr_bytes = await asyncio.wait_for(proc.stderr.read(), timeout=5)
            stderr_text = stderr_bytes.decode().strip() if stderr_bytes else ""
        except Exception:
            pass

        # If we already streamed text to the user, synthesize a result
        if streamed_text.strip():
            log.warning(
                "No result event but %d chars already streamed. stderr: %s",
                len(streamed_text), stderr_text[:500] or "(empty)",
            )
            return {
                "result": streamed_text,
                "session_id": None,
                "written_files": written_files,
            }

        # No text was streamed — this is a real failure
        if stderr_text:
            log.error("No result event in stream. stderr: %s", stderr_text[:1000])
            raise RuntimeError(f"Claude stream ended without result: {stderr_text[:300]}")
        raise RuntimeError("No result event in stream output (claude may have crashed)")
    return result


# ---------------------------------------------------------------------------
# Public API — delegates to configured backend with automatic failover
# ---------------------------------------------------------------------------

_failover_chain: list | None = None

_CHAIN_REGISTRY = {
    "bedrock_api": ("backends.bedrock_api", "BedrockAPIBackend"),
    "anthropic_api": ("backends.anthropic_api", "AnthropicAPIBackend"),
    "gemini_api": ("backends.gemini_api", "GeminiAPIBackend"),
    "vllm": ("backends.vllm", "VLLMBackend"),
}


def _health_check_url(url: str, timeout: float = 2.0) -> bool:
    """Quick HTTP connectivity check. Returns True if reachable."""
    import urllib.request
    try:
        base = url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        req = urllib.request.Request(base + "/health", method="GET")
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except Exception:
        return False


async def _health_check_url_async(url: str, timeout: float = 2.0) -> bool:
    """Async HTTP connectivity check."""
    import aiohttp
    try:
        base = url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        async with aiohttp.ClientSession() as session:
            async with session.get(
                base + "/health",
                timeout=aiohttp.ClientTimeout(total=timeout),
            ):
                return True
    except Exception:
        return False


def _get_failover_chain() -> list:
    """Build and cache the ordered failover chain.

    Returns list of dicts: {backend_name, url_override, instance}.
    Supports FAILOVER_CHAIN env (new) or FAILOVER_BACKEND (legacy).
    """
    global _failover_chain
    if _failover_chain is not None:
        return _failover_chain

    from config import (
        FAILOVER_ENABLED, FAILOVER_BACKEND, FAILOVER_CHAIN,
        VLLM_FALLBACK_URL, ENGINE_BACKEND,
    )

    if not FAILOVER_ENABLED:
        _failover_chain = []
        return _failover_chain

    chain_specs = []

    if FAILOVER_CHAIN:
        # New format: "vllm:http://10.200.0.3:5000/v1,bedrock_api,gemini_api"
        for spec in FAILOVER_CHAIN.split(","):
            spec = spec.strip()
            if not spec:
                continue
            if ":" in spec and not spec.startswith("http"):
                name, url = spec.split(":", 1)
                chain_specs.append({"backend_name": name.strip(), "url_override": url.strip(), "instance": None})
            else:
                chain_specs.append({"backend_name": spec, "url_override": None, "instance": None})
    else:
        # Legacy: single FAILOVER_BACKEND. Inject VLLM_FALLBACK_URL before it if applicable.
        if ENGINE_BACKEND == "vllm" and VLLM_FALLBACK_URL:
            chain_specs.append({"backend_name": "vllm", "url_override": VLLM_FALLBACK_URL, "instance": None})
        if FAILOVER_BACKEND:
            chain_specs.append({"backend_name": FAILOVER_BACKEND, "url_override": None, "instance": None})

    _failover_chain = chain_specs
    if chain_specs:
        chain_desc = " -> ".join(
            f"{s['backend_name']}({s['url_override']})" if s["url_override"] else s["backend_name"]
            for s in chain_specs
        )
        log.info("Failover chain: %s", chain_desc)
    return _failover_chain


def _get_chain_backend(spec: dict):
    """Lazy-initialize a backend instance for a failover chain entry."""
    if spec["instance"] is not None:
        return spec["instance"]

    import importlib

    name = spec["backend_name"]
    if name not in _CHAIN_REGISTRY:
        log.warning("Unknown failover backend in chain: %s", name)
        return None

    try:
        module_path, class_name = _CHAIN_REGISTRY[name]
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        instance = cls()

        # Apply URL override for vLLM backends pointing to alternate endpoints
        if spec["url_override"] and name == "vllm":
            from openai import OpenAI, AsyncOpenAI
            instance._url = spec["url_override"]
            instance._client = OpenAI(base_url=spec["url_override"], api_key="dummy")
            instance._aclient = AsyncOpenAI(base_url=spec["url_override"], api_key="dummy")

        spec["instance"] = instance
        log.info(
            "Failover chain backend initialized: %s%s",
            instance.name,
            f" (url={spec['url_override']})" if spec["url_override"] else "",
        )
    except Exception as e:
        log.warning("Failed to initialize failover chain backend %s: %s", name, e)
        return None

    return spec["instance"]


def get_model_display(model: str = CLAUDE_MODEL) -> str:
    """Return a human-readable label for the current backend + model.

    E.g. 'Claude Sonnet' for claude_code, 'qwen3:30b-a3b' for ollama.
    """
    from backends import get_backend

    backend = get_backend()
    if hasattr(backend, "get_model_display"):
        return backend.get_model_display(model)
    return f"{backend.name}:{model}"


def call_sync(
    prompt: str,
    model: str = CLAUDE_MODEL,
    session_id: str | None = None,
    timeout: int = 300,
) -> dict:
    """Synchronous LLM call (for observers running in thread pool).

    Returns dict with 'result', 'session_id' keys.
    Automatically fails over to Bedrock if primary backend errors.
    """
    import time as _time
    from backends import get_backend

    backend = get_backend()
    t0 = _time.monotonic()

    try:
        result = backend.call_sync(
            prompt,
            model=model,
            session_id=session_id,
            timeout=timeout,
            system_prompt=SYSTEM_PROMPT,
        )
        # Check for error results from backends that return errors as dicts
        if result.get("error"):
            raise RuntimeError(result.get("result", "Backend returned error"))
    except Exception as primary_err:
        log.error("Primary backend (%s) failed: %s", backend.name, primary_err)

        from config import FAILOVER_HEALTH_TIMEOUT
        chain = _get_failover_chain()

        for i, spec in enumerate(chain):
            fb = _get_chain_backend(spec)
            if fb is None:
                continue

            # Health-check preflight for URL-based backends
            url_to_check = spec.get("url_override") or getattr(fb, "_url", None)
            if url_to_check:
                if not _health_check_url(url_to_check, FAILOVER_HEALTH_TIMEOUT):
                    log.info("Failover[%d] %s health check failed, skipping", i, spec["backend_name"])
                    continue

            fb_model = model if spec["backend_name"] == "vllm" else "sonnet"
            log.info("Failing over to chain[%d]: %s (model=%s)", i, fb.name, fb_model)
            try:
                result = fb.call_sync(
                    prompt,
                    model=fb_model,
                    session_id=None,
                    timeout=timeout,
                    system_prompt=SYSTEM_PROMPT,
                )
                result["_failover"] = True
                result["_failover_backend"] = fb.name
                result["_failover_chain_index"] = i
                duration = int((_time.monotonic() - t0) * 1000)
                try:
                    from security.audit import log_llm_call
                    log_llm_call(None, None, fb.name, fb_model, None, duration)
                except Exception:
                    pass
                return result
            except Exception as chain_err:
                log.error("Failover[%d] %s failed: %s", i, spec["backend_name"], chain_err)
                continue

        return {"result": f"All backends failed. Primary: {primary_err}", "session_id": session_id}

    duration = int((_time.monotonic() - t0) * 1000)

    try:
        from security.audit import log_llm_call
        log_llm_call(session_id, None, backend.name, model, None, duration)
    except Exception:
        pass

    return result


def _build_memory_context(chat_id: int | None = None) -> str | None:
    """Assemble memory context for injection (shared by streaming + failover)."""
    parts = []
    if get_memories_for_injection:
        mem = get_memories_for_injection()
        if mem:
            parts.append(mem)
    if get_shared_context:
        shared = get_shared_context()
        if shared:
            parts.append(shared)
    # Inject user profile if chat_id provided
    if chat_id:
        try:
            from db import get_user_profile
            profile = get_user_profile(chat_id)
            if profile and profile.get("display_name"):
                profile_lines = ["[User Profile]"]
                profile_lines.append(f"Name: {profile['display_name']}")
                if profile.get("timezone") and profile["timezone"] != "UTC":
                    profile_lines.append(f"Timezone: {profile['timezone']}")
                parts.append("\n".join(profile_lines))
        except Exception:
            pass

    memory_ctx = "\n\n".join(parts) if parts else None

    # Redact memory context before injection
    try:
        from security.redact import redact_text
        if memory_ctx:
            memory_ctx = redact_text(memory_ctx)
    except Exception:
        pass

    return memory_ctx


async def call_streaming(
    message: str,
    session_id: str | None,
    model: str,
    on_progress=None,
    streaming_editor=None,
    extra_system_prompt: str | None = None,
    chat_id: int | None = None,
) -> dict:
    """Async streaming LLM call with real-time progress.

    Returns dict with 'result', 'session_id', 'written_files' keys.
    Automatically fails over to Bedrock Sonnet if primary backend errors.
    """
    import time as _time
    from backends import get_backend

    # Inference guard: model allowlist check
    try:
        from security.inference import get_inference_guard
        guard = get_inference_guard()
        allowed, reason = guard.check_model(model)
        if not allowed:
            log.warning("Inference guard blocked model '%s': %s", model, reason)
            return {"result": f"Error: {reason}", "session_id": session_id, "written_files": []}
    except Exception:
        pass

    memory_ctx = _build_memory_context(chat_id)

    backend = get_backend()
    t0 = _time.monotonic()

    try:
        result = await backend.call_streaming(
            message,
            session_id=session_id,
            model=model,
            on_progress=on_progress,
            streaming_editor=streaming_editor,
            system_prompt=SYSTEM_PROMPT,
            memory_context=memory_ctx,
            extra_system_prompt=extra_system_prompt,
        )
    except Exception as primary_err:
        log.error("Primary backend (%s) streaming failed: %s", backend.name, primary_err)

        from config import FAILOVER_HEALTH_TIMEOUT
        chain = _get_failover_chain()

        for i, spec in enumerate(chain):
            fb = _get_chain_backend(spec)
            if fb is None:
                continue

            # Async health-check preflight for URL-based backends
            url_to_check = spec.get("url_override") or getattr(fb, "_url", None)
            if url_to_check:
                if not await _health_check_url_async(url_to_check, FAILOVER_HEALTH_TIMEOUT):
                    log.info("Failover[%d] %s health check failed, skipping", i, spec["backend_name"])
                    continue

            # Notify user
            if streaming_editor:
                try:
                    fb_label = spec["backend_name"]
                    if spec.get("url_override"):
                        fb_label += " (200G)"
                    await streaming_editor.add_tool_status(
                        f"Primary failed. Trying {fb_label}..."
                    )
                except Exception:
                    pass

            fb_model = model if spec["backend_name"] == "vllm" else "sonnet"
            log.info("Failing over to chain[%d]: %s (model=%s)", i, fb.name, fb_model)
            try:
                result = await fb.call_streaming(
                    message,
                    session_id=None,
                    model=fb_model,
                    on_progress=on_progress,
                    streaming_editor=streaming_editor,
                    system_prompt=SYSTEM_PROMPT,
                    memory_context=memory_ctx,
                    extra_system_prompt=extra_system_prompt,
                )
                result["_failover"] = True
                result["_failover_backend"] = fb.name
                result["_failover_chain_index"] = i
                duration = int((_time.monotonic() - t0) * 1000)
                try:
                    from security.audit import log_llm_call
                    log_llm_call(None, None, fb.name, fb_model, None, duration)
                except Exception:
                    pass
                return result
            except Exception as chain_err:
                log.error("Failover[%d] %s streaming failed: %s", i, spec["backend_name"], chain_err)
                continue

        return {
            "result": f"All backends failed. Primary: {primary_err}",
            "session_id": session_id,
            "written_files": [],
        }

    duration = int((_time.monotonic() - t0) * 1000)

    try:
        from security.audit import log_llm_call
        log_llm_call(session_id, None, backend.name, model, None, duration)
    except Exception:
        pass

    return result
