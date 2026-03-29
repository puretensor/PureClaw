"""Filesystem access control — path-based ACLs for tool operations.

Resolves symlinks, matches against policy read/write allow/deny lists.
Deny rules take precedence over allow. Defense-in-depth for bash commands
via heuristic static analysis (not a security boundary).
"""

from __future__ import annotations

import logging
import os
import re
import shlex

from security.policy import get_policy, matches_glob

log = logging.getLogger("nexus.security")


def check_path_access(path: str, mode: str) -> tuple[bool, str]:
    """Check if a path is allowed for the given mode ('read' or 'write').

    Resolves symlinks before matching. Deny rules take precedence.
    Returns (allowed: bool, reason: str).
    """
    policy = get_policy().filesystem

    # Resolve symlinks to get real path
    try:
        real_path = os.path.realpath(path)
    except (OSError, ValueError):
        real_path = path

    if mode == "read":
        deny_patterns = policy.read_deny
        allow_patterns = policy.read_allow
    elif mode == "write":
        deny_patterns = policy.write_deny
        allow_patterns = policy.write_allow
    else:
        return False, f"Unknown mode: {mode}"

    # Check deny first (takes precedence)
    if matches_glob(real_path, deny_patterns):
        return False, f"Path '{real_path}' matches {mode}_deny policy"

    # Also check the original path (before symlink resolution)
    if path != real_path and matches_glob(path, deny_patterns):
        return False, f"Path '{path}' matches {mode}_deny policy"

    # Check allow
    if matches_glob(real_path, allow_patterns):
        return True, "allowed"

    if path != real_path and matches_glob(path, allow_patterns):
        return True, "allowed"

    return False, f"Path '{real_path}' not in {mode}_allow policy"


# ---------------------------------------------------------------------------
# Bash command heuristic analysis (defense-in-depth, NOT a security boundary)
# ---------------------------------------------------------------------------

# Patterns that suggest dangerous write operations
_DANGEROUS_PATTERNS = [
    (re.compile(r"\brm\s+(-[rRf]+\s+)?/(?!tmp)"), "rm on system path"),
    (re.compile(r"\bchmod\b"), "chmod"),
    (re.compile(r"\bchown\b"), "chown"),
    (re.compile(r"\bmkfs\b"), "mkfs"),
    (re.compile(r"\bdd\s+"), "dd"),
    (re.compile(r"\b(curl|wget)\b.*\|\s*(ba)?sh"), "pipe-to-shell"),
]

_NETWORK_PATTERNS = [
    (re.compile(r"\b(curl|wget|lynx|httpie)\b"), "network egress"),
    (re.compile(r"\b(nc|ncat|telnet|ssh|scp|rsync)\b"), "remote network access"),
]


def _extract_write_targets(command: str) -> list[str]:
    """Extract potential file write targets from a bash command."""
    targets = []
    # Output redirects: > /path, >> /path
    for m in re.finditer(r">{1,2}\s*(\S+)", command):
        targets.append(m.group(1))
    # tee
    for m in re.finditer(r"\btee\s+(?:-a\s+)?(\S+)", command):
        targets.append(m.group(1))
    # cp/mv destination (last arg)
    for m in re.finditer(r"\b(cp|mv)\s+.*\s+(\S+)\s*$", command):
        targets.append(m.group(2))
    return targets


def _extract_read_targets(command: str) -> list[str]:
    """Extract likely filesystem read targets from simple bash commands."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return []

    if not tokens:
        return []

    verb = tokens[0]
    read_verbs = {"cat", "head", "tail", "less", "more", "nl", "wc"}
    regex_verbs = {"grep", "rg", "sed", "awk"}

    targets: list[str] = []
    args = [t for t in tokens[1:] if not t.startswith("-")]

    if verb in read_verbs:
        targets.extend(a for a in args if a.startswith("/") or a.startswith("."))
    elif verb in regex_verbs and args:
        # Last positional arg is usually the path/search target.
        target = args[-1]
        if target.startswith("/") or target.startswith("."):
            targets.append(target)
    elif verb in {"cp", "mv"} and len(args) >= 2:
        targets.extend(args[:-1])

    return targets


def check_bash_command(command: str) -> tuple[bool, str]:
    """Heuristic check on a bash command. Defense-in-depth only.

    Returns (allowed: bool, reason: str).
    """
    policy = get_policy().filesystem

    # Check dangerous patterns
    for pattern, desc in _DANGEROUS_PATTERNS:
        if pattern.search(command):
            return False, f"Bash command contains dangerous pattern: {desc}"

    for pattern, desc in _NETWORK_PATTERNS:
        if pattern.search(command):
            return False, f"Bash command contains blocked pattern: {desc}"

    # Check write targets against policy
    for target in _extract_write_targets(command):
        allowed, reason = check_path_access(target, "write")
        if not allowed:
            return False, f"Bash write target blocked: {reason}"

    # Heuristic read checks for obvious file reads.
    for target in _extract_read_targets(command):
        allowed, reason = check_path_access(target, "read")
        if not allowed:
            return False, f"Bash read target blocked: {reason}"

    return True, "allowed"
