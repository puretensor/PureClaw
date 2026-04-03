"""Azure OpenAI backend — via openai SDK with tool support.

Uses AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT from environment.
Mirrors the BedrockAPIBackend interface: tool loop, history sanitization,
streaming progress callbacks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid

from openai import AzureOpenAI

from db import get_conversation_history, save_conversation_history
from backends.anthropic_api import _sanitize_history, _build_system_blocks
from backends.tools import TOOL_SCHEMAS, ToolCall, run_tool_loop_sync, run_tool_loop_async

log = logging.getLogger("nexus")

# Model map — short names to Azure OpenAI deployment names
_MODEL_MAP = {
    "sonnet": "gpt-5-1-chat",
    "opus": "gpt-5-1",
    "haiku": "gpt-5-1-chat",
    # Legacy Bedrock IDs
    "us.anthropic.claude-sonnet-4-6": "gpt-5-1-chat",
    "us.anthropic.claude-opus-4-6": "gpt-5-1",
    "us.anthropic.claude-haiku-4-5-20251001": "gpt-5-1-chat",
}

# Vision model for image content
_VISION_MODEL = "gpt-4o"

# Pricing per million tokens (USD) for cost logging
_PRICING = {
    "gpt-5-1-chat": (2.00, 8.00),
    "gpt-5-1": (10.00, 30.00),
    "gpt-4o": (2.50, 10.00),
}


def _openai_tools() -> list[dict]:
    """Return tool schemas in OpenAI function-calling format.

    TOOL_SCHEMAS are already in OpenAI format: {"type": "function", "function": {...}}.
    """
    return list(TOOL_SCHEMAS)


def _convert_history_to_openai(
    messages: list[dict],
    system_text: str | None = None,
) -> list[dict]:
    """Convert Anthropic-format history to OpenAI chat messages.

    Anthropic uses:
      {"role": "user", "content": "text"} or
      {"role": "user", "content": [{"type": "text", ...}, ...]}
      {"role": "assistant", "content": [{"type": "text", ...}, {"type": "tool_use", ...}]}
      {"role": "user", "content": [{"type": "tool_result", ...}]}

    OpenAI uses:
      {"role": "system", "content": "..."}
      {"role": "user", "content": "..."}
      {"role": "assistant", "content": "...", "tool_calls": [...]}
      {"role": "tool", "tool_call_id": "...", "content": "..."}
    """
    result: list[dict] = []

    if system_text:
        result.append({"role": "system", "content": system_text})

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        # Simple string content
        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            result.append({"role": role, "content": str(content)})
            continue

        # Complex content blocks — need to decompose
        text_parts: list[str] = []
        tool_calls_out: list[dict] = []
        tool_results: list[dict] = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif not isinstance(block, dict):
                continue
            elif block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    text_parts.append(text)
            elif block.get("type") == "thinking":
                # Skip thinking blocks — OpenAI doesn't have an equivalent
                pass
            elif block.get("type") == "tool_use":
                tool_calls_out.append({
                    "id": block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })
            elif block.get("type") == "tool_result":
                result_content = block.get("content", "")
                if isinstance(result_content, str):
                    result_text = result_content
                elif isinstance(result_content, list):
                    result_text = " ".join(
                        rb.get("text", "") if isinstance(rb, dict) else str(rb)
                        for rb in result_content
                    )
                else:
                    result_text = str(result_content)
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": result_text[:8000],
                })

        # Emit assistant message with optional tool_calls
        if role == "assistant":
            assistant_msg: dict = {"role": "assistant"}
            combined_text = "\n".join(text_parts).strip()
            assistant_msg["content"] = combined_text or None
            if tool_calls_out:
                assistant_msg["tool_calls"] = tool_calls_out
            result.append(assistant_msg)
        elif tool_results:
            # tool_result blocks come as role=user in Anthropic format
            # but need to be role=tool in OpenAI format
            result.extend(tool_results)
        else:
            combined_text = "\n".join(text_parts).strip()
            if combined_text:
                result.append({"role": role, "content": combined_text})

    return result


def _log_usage(usage, model_id: str, label: str = "") -> None:
    """Log token usage and estimated cost from OpenAI response."""
    if not usage:
        return

    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0

    prefix = f"[{label}] " if label else ""

    input_price, output_price = _PRICING.get(model_id, (2.00, 8.00))
    cost_in = input_tokens * input_price / 1_000_000
    cost_out = output_tokens * output_price / 1_000_000
    total_cost = cost_in + cost_out

    log.info(
        "%sAzure OpenAI usage: in=%d out=%d cost=$%.4f",
        prefix, input_tokens, output_tokens, total_cost,
    )

    # Prometheus metrics
    try:
        from metrics import inc
        inc("nexus_llm_calls_total", {"backend": "gemini", "model": model_id})
        inc("nexus_llm_tokens_total", {"backend": "gemini", "model": model_id, "direction": "input"}, input_tokens)
        inc("nexus_llm_tokens_total", {"backend": "gemini", "model": model_id, "direction": "output"}, output_tokens)
        inc("nexus_llm_cost_usd_total", {"backend": "gemini", "model": model_id}, total_cost)
    except Exception:
        pass


class GeminiAPIBackend:
    """Backend that calls Azure OpenAI with tool loop.

    Class name kept as GeminiAPIBackend for caller compatibility.
    """

    def __init__(self):
        from config import (
            ANTHROPIC_TOOLS_ENABLED,
            ANTHROPIC_TOOL_MAX_ITER,
            ANTHROPIC_TOOL_TIMEOUT,
            ANTHROPIC_TOTAL_TIMEOUT,
            CLAUDE_CWD,
        )

        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

        if not api_key or not endpoint:
            raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set")

        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self._default_model = os.environ.get("AZURE_OPENAI_MODEL", "gpt-5-1-chat")
        self._max_tokens = int(os.environ.get("AZURE_OPENAI_MAX_TOKENS", "65536"))
        self._tools_enabled = ANTHROPIC_TOOLS_ENABLED
        self._max_iterations = ANTHROPIC_TOOL_MAX_ITER
        self._tool_timeout = ANTHROPIC_TOOL_TIMEOUT
        self._total_timeout = ANTHROPIC_TOTAL_TIMEOUT
        self._cwd = CLAUDE_CWD

        self._tools = _openai_tools() if self._tools_enabled else None

    @property
    def name(self) -> str:
        return "azure_openai"

    def get_model_display(self, model: str) -> str:
        return self._resolve_model(model)

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_tools(self) -> bool:
        return self._tools_enabled

    @property
    def supports_sessions(self) -> bool:
        return False

    def _resolve_model(self, model: str) -> str:
        if not model:
            return self._default_model
        return _MODEL_MAP.get(model, model)

    # ------------------------------------------------------------------
    # Response parsing helpers
    # ------------------------------------------------------------------

    def _parse_response(self, response) -> tuple[str, list[ToolCall], dict]:
        """Parse Azure OpenAI response into (text, tool_calls, assistant_msg).

        Returns the assistant message in Anthropic-compatible format so it can
        be appended to the shared conversation history without conversion.
        """
        _log_usage(
            getattr(response, "usage", None),
            self._default_model,
            "azure-openai",
        )

        tool_calls: list[ToolCall] = []
        anthropic_blocks: list[dict] = []

        if not response.choices:
            return ("", [], {"role": "assistant", "content": []})

        message = response.choices[0].message
        text = message.content or ""

        if text:
            anthropic_blocks.append({"type": "text", "text": text})

        if message.tool_calls:
            for tc in message.tool_calls:
                call_id = tc.id or f"call_{uuid.uuid4().hex[:24]}"
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(
                    id=call_id,
                    name=fn_name,
                    arguments=args,
                ))
                anthropic_blocks.append({
                    "type": "tool_use",
                    "id": call_id,
                    "name": fn_name,
                    "input": args,
                })

        assistant_msg = {"role": "assistant", "content": anthropic_blocks}
        return (text.strip(), tool_calls, assistant_msg)

    @staticmethod
    def _format_tool_result(tool_name: str, call_id: str, result_str: str) -> dict:
        """Format tool result in Anthropic-compatible format for history."""
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "_tool_name": tool_name,  # Extra field for history conversion
                    "content": result_str,
                }
            ],
        }

    # ------------------------------------------------------------------
    # Core API call
    # ------------------------------------------------------------------

    def _build_system_text(
        self,
        system_prompt: str | None,
        memory_context: str | None,
        extra_system_prompt: str | None = None,
    ) -> str | None:
        """Build system text from components."""
        system_blocks = _build_system_blocks(system_prompt, memory_context, extra_system_prompt)
        system_text = "\n\n".join(b.get("text", "") for b in system_blocks if b.get("text"))
        return system_text or None

    def _build_request_kwargs(
        self,
        model_id: str,
        messages: list[dict],
        system_prompt: str | None = None,
        memory_context: str | None = None,
        extra_system_prompt: str | None = None,
        stream: bool = False,
    ) -> dict:
        """Build kwargs for client.chat.completions.create()."""
        system_text = self._build_system_text(system_prompt, memory_context, extra_system_prompt)
        openai_messages = _convert_history_to_openai(messages, system_text)

        kwargs = {
            "model": model_id,
            "messages": openai_messages,
            "max_completion_tokens": self._max_tokens,
            "stream": stream,
        }

        if self._tools_enabled and self._tools:
            kwargs["tools"] = self._tools

        return kwargs

    def _generate(self, model_id: str, messages: list[dict], **system_kw):
        """Synchronous generate call."""
        kwargs = self._build_request_kwargs(model_id, messages, **system_kw)
        return self._client.chat.completions.create(**kwargs)

    async def _generate_async(self, model_id: str, messages: list[dict], **system_kw):
        """Async generate call (runs sync client in thread pool)."""
        kwargs = self._build_request_kwargs(model_id, messages, **system_kw)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._client.chat.completions.create(**kwargs),
        )

    async def _stream_request(
        self,
        messages: list[dict],
        model_id: str,
        streaming_editor=None,
        **system_kw,
    ) -> tuple[str, list, dict]:
        """Send streaming request and consume chunks.

        Returns (text, tool_calls, assistant_msg) -- same as _parse_response().
        """
        kwargs = self._build_request_kwargs(model_id, messages, stream=True, **system_kw)
        loop = asyncio.get_event_loop()

        def _sync_stream():
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            anthropic_blocks: list[dict] = []

            # Accumulate tool call deltas keyed by index
            tc_accum: dict[int, dict] = {}

            response_stream = self._client.chat.completions.create(**kwargs)

            for chunk in response_stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Text content
                if delta and delta.content:
                    text_parts.append(delta.content)
                    if streaming_editor and loop:
                        try:
                            future = asyncio.run_coroutine_threadsafe(
                                streaming_editor.add_text(delta.content), loop
                            )
                            future.result(timeout=2)
                        except Exception:
                            pass

                # Tool call deltas — streamed incrementally
                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tc_accum:
                            tc_accum[idx] = {
                                "id": tc_delta.id or "",
                                "name": "",
                                "arguments": "",
                            }
                        if tc_delta.id:
                            tc_accum[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tc_accum[idx]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tc_accum[idx]["arguments"] += tc_delta.function.arguments

            # Assemble final text
            full_text = "".join(text_parts).strip()
            if full_text:
                anthropic_blocks.append({"type": "text", "text": full_text})

            # Assemble tool calls from accumulated deltas
            for _idx in sorted(tc_accum):
                acc = tc_accum[_idx]
                call_id = acc["id"] or f"call_{uuid.uuid4().hex[:24]}"
                fn_name = acc["name"]
                try:
                    args = json.loads(acc["arguments"]) if acc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(
                    id=call_id,
                    name=fn_name,
                    arguments=args,
                ))
                anthropic_blocks.append({
                    "type": "tool_use",
                    "id": call_id,
                    "name": fn_name,
                    "input": args,
                })

            # Log usage from final chunk if available
            if chunk and hasattr(chunk, "usage") and chunk.usage:
                _log_usage(chunk.usage, model_id, "azure-openai-stream")

            assistant_msg = {"role": "assistant", "content": anthropic_blocks}
            return (full_text, tool_calls, assistant_msg)

        return await loop.run_in_executor(None, _sync_stream)

    # ------------------------------------------------------------------
    # Sync call (for observers)
    # ------------------------------------------------------------------

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
        session_id = session_id or str(uuid.uuid4())
        model_id = self._resolve_model(model)

        return self._call_sync_inner(
            prompt, model_id=model_id, session_id=session_id,
            timeout=timeout, system_prompt=system_prompt,
            memory_context=memory_context,
            tool_context=tool_context,
        )

    def _call_sync_inner(
        self,
        prompt: str,
        *,
        model_id: str,
        session_id: str,
        timeout: int,
        system_prompt: str | None,
        memory_context: str | None,
        tool_context=None,
    ) -> dict:
        from context_compression import compress_tool_results, compress_history
        history = _sanitize_history(get_conversation_history(session_id))
        history = compress_tool_results(history)
        history = compress_history(history)
        messages = history + [{"role": "user", "content": prompt}]

        system_kw = dict(
            system_prompt=system_prompt,
            memory_context=memory_context,
        )

        def send_request(msgs):
            try:
                from security.redact import redact_history

                msgs = redact_history(msgs)
            except Exception:
                pass
            return self._generate(model_id, msgs, **system_kw)

        if self._tools_enabled:
            try:
                result = run_tool_loop_sync(
                    messages,
                    send_request,
                    self._parse_response,
                    self._format_tool_result,
                    max_iterations=self._max_iterations,
                    tool_timeout=self._tool_timeout,
                    total_timeout=min(timeout, self._total_timeout),
                    cwd=self._cwd,
                    tool_context=tool_context,
                )
            except Exception as e:
                log.error("Azure OpenAI tool loop error (sync): %s", e)
                return {"result": f"Azure OpenAI error: {e}", "session_id": session_id}
            result_text = result.get("result", "")
            if result_text:
                save_conversation_history(session_id, messages)
            result["session_id"] = session_id
            return result

        # No tools — single request
        try:
            resp = send_request(messages)
        except Exception as e:
            return {"result": f"Azure OpenAI error: {e}", "session_id": session_id}

        text, _tool_calls, _assistant_msg = self._parse_response(resp)
        if text:
            save_conversation_history(session_id, messages + [
                {"role": "assistant", "content": text}
            ])
        return {"result": text or "(empty response)", "session_id": session_id}

    # ------------------------------------------------------------------
    # Async call (for Telegram with progress)
    # ------------------------------------------------------------------

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
        session_id = session_id or str(uuid.uuid4())
        model_id = self._resolve_model(model)

        from context_compression import compress_tool_results, compress_history
        history = _sanitize_history(get_conversation_history(session_id))
        history = compress_tool_results(history)
        history = compress_history(history)
        messages = history + [{"role": "user", "content": message}]

        system_kw = dict(
            system_prompt=system_prompt,
            memory_context=memory_context,
            extra_system_prompt=extra_system_prompt,
        )

        async def send_and_parse_stream(msgs, editor):
            try:
                from security.redact import redact_history

                msgs = redact_history(msgs)
            except Exception:
                pass
            return await self._stream_request(
                msgs, model_id, streaming_editor=editor, **system_kw,
            )

        async def send_request(msgs):
            try:
                from security.redact import redact_history

                msgs = redact_history(msgs)
            except Exception:
                pass
            return await self._generate_async(model_id, msgs, **system_kw)

        if self._tools_enabled:
            try:
                result = await run_tool_loop_async(
                    messages,
                    send_request,
                    self._parse_response,
                    self._format_tool_result,
                    max_iterations=self._max_iterations,
                    tool_timeout=self._tool_timeout,
                    total_timeout=self._total_timeout,
                    cwd=self._cwd,
                    streaming_editor=streaming_editor,
                    on_progress=on_progress,
                    send_and_parse_stream=send_and_parse_stream,
                    tool_context=tool_context,
                )
            except Exception as e:
                log.error("Azure OpenAI tool loop error (async): %s", e)
                return {"result": f"Azure OpenAI error: {e}", "session_id": session_id, "written_files": []}
            result_text = result.get("result", "")
            if result_text:
                save_conversation_history(session_id, messages)
            result["session_id"] = session_id
            return result

        # No tools — single streaming request
        try:
            text, _tool_calls, _assistant_msg = await send_and_parse_stream(
                messages, streaming_editor,
            )
        except Exception as e:
            return {"result": f"Azure OpenAI error: {e}", "session_id": session_id, "written_files": []}

        if text:
            save_conversation_history(session_id, messages + [
                {"role": "assistant", "content": text}
            ])
        return {"result": text or "(empty response)", "session_id": session_id, "written_files": []}
