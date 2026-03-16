"""Inference guard — model allowlist, token budget, system prompt integrity.

Validates model selection, tracks cumulative token usage per session,
and optionally locks system prompts against mutation.
"""

from __future__ import annotations

import fnmatch
import logging
import time
from collections import defaultdict
from typing import Optional

from security.policy import get_policy

log = logging.getLogger("nexus.security")


class InferenceGuard:
    """Enforces inference policy on LLM calls."""

    def __init__(self):
        # Token tracking: session_id -> (total_tokens, last_update_timestamp)
        self._token_usage: dict[str, tuple[int, float]] = defaultdict(lambda: (0, 0.0))
        self._ttl = 86400  # 24h session TTL for token tracking

    def check_model(self, model: str) -> tuple[bool, str]:
        """Validate model against allowlist.

        Returns (allowed: bool, reason: str).
        """
        policy = get_policy().inference
        for pattern in policy.model_allowlist:
            if pattern == "*" or fnmatch.fnmatch(model, pattern):
                return True, "allowed"
        return False, f"Model '{model}' not in inference model_allowlist"

    def check_system_prompt(self, current: Optional[str], original: Optional[str]) -> tuple[bool, str]:
        """Check if system prompt has been mutated (if immutability is enabled).

        Returns (allowed: bool, reason: str).
        """
        policy = get_policy().inference
        if not policy.system_prompt_immutable:
            return True, "system_prompt_immutable disabled"

        if current != original:
            return False, "System prompt mutation detected (system_prompt_immutable=true)"
        return True, "allowed"

    def check_token_budget(self, session_id: Optional[str], new_tokens: int) -> tuple[bool, str]:
        """Check if adding new_tokens exceeds the session budget.

        Returns (allowed: bool, reason: str).
        """
        policy = get_policy().inference
        if policy.max_tokens_per_session <= 0 or not session_id:
            return True, "no budget limit"

        # Clean expired entries
        now = time.time()
        self._cleanup(now)

        current, _ = self._token_usage[session_id]
        projected = current + new_tokens

        if projected > policy.max_tokens_per_session:
            return False, (
                f"Token budget exceeded: {projected} > {policy.max_tokens_per_session} "
                f"(session {session_id})"
            )

        # Update usage
        self._token_usage[session_id] = (projected, now)
        return True, "within budget"

    def record_tokens(self, session_id: Optional[str], tokens: int) -> None:
        """Record token usage without budget check (for post-hoc tracking)."""
        if not session_id:
            return
        current, _ = self._token_usage[session_id]
        self._token_usage[session_id] = (current + tokens, time.time())

    def _cleanup(self, now: float) -> None:
        """Remove expired session entries."""
        expired = [
            sid for sid, (_, ts) in self._token_usage.items()
            if now - ts > self._ttl
        ]
        for sid in expired:
            del self._token_usage[sid]


# Module-level singleton
_guard: Optional[InferenceGuard] = None


def get_inference_guard() -> InferenceGuard:
    """Get or create the singleton InferenceGuard."""
    global _guard
    if _guard is None:
        _guard = InferenceGuard()
    return _guard
