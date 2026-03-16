"""Credential redaction — prevents secret leakage into LLM context and logs.

Applies regex patterns from policy + environment variable values. Patterns
are compiled once at first use and cached. Thread-safe (patterns are frozen
after compilation).
"""

from __future__ import annotations

import copy
import logging
import os
import re
from typing import Optional

log = logging.getLogger("nexus.security")

_REDACTED = "[REDACTED]"

# Compiled patterns cache (lazily initialized)
_compiled_patterns: Optional[list[re.Pattern]] = None
_env_values: Optional[dict[str, str]] = None


def _get_compiled_patterns() -> list[re.Pattern]:
    """Lazily compile regex patterns from policy."""
    global _compiled_patterns
    if _compiled_patterns is not None:
        return _compiled_patterns

    from security.policy import get_policy
    policy = get_policy().credentials

    patterns = []
    for raw in policy.redact_patterns:
        try:
            patterns.append(re.compile(raw))
        except re.error as e:
            log.warning("Invalid redact pattern '%s': %s", raw, e)
    _compiled_patterns = patterns
    return patterns


def _get_env_values() -> dict[str, str]:
    """Get environment variable values to redact."""
    global _env_values
    if _env_values is not None:
        return _env_values

    from security.policy import get_policy
    policy = get_policy().credentials

    values = {}
    for var_name in policy.redact_env_vars:
        val = os.environ.get(var_name, "")
        if val and len(val) >= 8:  # Only redact values long enough to be meaningful
            values[var_name] = val
    _env_values = values
    return values


def invalidate_cache() -> None:
    """Clear compiled pattern cache (called on policy reload)."""
    global _compiled_patterns, _env_values
    _compiled_patterns = None
    _env_values = None


def redact_text(text: str) -> str:
    """Apply all redaction rules to a text string.

    1. Regex patterns from policy
    2. Literal env var values
    """
    if not text:
        return text

    # Apply regex patterns
    for pattern in _get_compiled_patterns():
        text = pattern.sub(_REDACTED, text)

    # Apply env var value redaction (literal string replacement)
    for var_name, var_value in _get_env_values().items():
        if var_value in text:
            text = text.replace(var_value, _REDACTED)

    return text


def redact_args(args: dict) -> dict:
    """Deep-copy and redact all string values in an args dict."""
    if not args:
        return args
    result = copy.deepcopy(args)
    _redact_dict(result)
    return result


def _redact_dict(d: dict) -> None:
    """Recursively redact string values in a dict (in-place)."""
    for key, value in d.items():
        if isinstance(value, str):
            d[key] = redact_text(value)
        elif isinstance(value, dict):
            _redact_dict(value)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, str):
                    value[i] = redact_text(item)
                elif isinstance(item, dict):
                    _redact_dict(item)


def redact_history(messages: list[dict]) -> list[dict]:
    """Redact credential patterns from conversation history messages.

    Deep-copies the list and redacts all 'content' fields (string or list).
    Safe to call on any message format (Anthropic, OpenAI, Bedrock).
    """
    if not messages:
        return messages

    result = copy.deepcopy(messages)
    for msg in result:
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = redact_text(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        block["text"] = redact_text(text)
    return result
