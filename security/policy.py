"""Security policy — schema, loading, singleton, hot-reload.

Loads a YAML policy file, validates against JSON Schema, exposes frozen
dataclasses for enforcement modules. Thread-safe reads; hot-reload swaps
the entire object atomically (Python GIL guarantees attribute assignment).
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger("nexus.security")

_SCHEMA_PATH = Path(__file__).parent / "schema.json"
_DEFAULT_POLICY_PATH = Path(__file__).parent / "policy.yaml"


# ---------------------------------------------------------------------------
# Frozen policy dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FilesystemPolicy:
    read_allow: tuple[str, ...] = ("/**",)
    read_deny: tuple[str, ...] = ()
    write_allow: tuple[str, ...] = ("/**",)
    write_deny: tuple[str, ...] = ()


@dataclass(frozen=True)
class NetworkPolicy:
    fetch_allow_domains: tuple[str, ...] = ("*",)
    fetch_deny_domains: tuple[str, ...] = ()
    block_private_ranges: bool = True


@dataclass(frozen=True)
class ToolPolicy:
    allowed: tuple[str, ...] = ("*",)
    denied: tuple[str, ...] = ()


@dataclass(frozen=True)
class InferencePolicy:
    model_allowlist: tuple[str, ...] = ("*",)
    max_tokens_per_session: int = 0
    system_prompt_immutable: bool = False


@dataclass(frozen=True)
class CredentialPolicy:
    redact_patterns: tuple[str, ...] = ()
    redact_env_vars: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuditPolicy:
    enabled: bool = True
    log_tool_args: bool = True
    log_result_preview: bool = True
    retention_days: int = 90


@dataclass(frozen=True)
class SecurityPolicy:
    version: int = 1
    filesystem: FilesystemPolicy = field(default_factory=FilesystemPolicy)
    network: NetworkPolicy = field(default_factory=NetworkPolicy)
    tools: ToolPolicy = field(default_factory=ToolPolicy)
    inference: InferencePolicy = field(default_factory=InferencePolicy)
    credentials: CredentialPolicy = field(default_factory=CredentialPolicy)
    audit: AuditPolicy = field(default_factory=AuditPolicy)


# ---------------------------------------------------------------------------
# YAML -> dataclass conversion
# ---------------------------------------------------------------------------

def _to_tuple(val, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Convert list/None to tuple of strings. Uses default if val is None."""
    if val is None:
        return default
    return tuple(str(v) for v in val)


# Defaults for sub-policies (matching frozen dataclass defaults)
_FS_DEFAULTS = FilesystemPolicy()
_NET_DEFAULTS = NetworkPolicy()
_TOOL_DEFAULTS = ToolPolicy()
_INF_DEFAULTS = InferencePolicy()


def _parse_policy(data: dict) -> SecurityPolicy:
    """Convert raw YAML dict to frozen SecurityPolicy."""
    fs = data.get("filesystem") or {}
    net = data.get("network") or {}
    tools = data.get("tools") or {}
    inf = data.get("inference") or {}
    cred = data.get("credentials") or {}
    audit = data.get("audit") or {}

    return SecurityPolicy(
        version=data.get("version", 1),
        filesystem=FilesystemPolicy(
            read_allow=_to_tuple(fs.get("read_allow"), _FS_DEFAULTS.read_allow),
            read_deny=_to_tuple(fs.get("read_deny"), _FS_DEFAULTS.read_deny),
            write_allow=_to_tuple(fs.get("write_allow"), _FS_DEFAULTS.write_allow),
            write_deny=_to_tuple(fs.get("write_deny"), _FS_DEFAULTS.write_deny),
        ),
        network=NetworkPolicy(
            fetch_allow_domains=_to_tuple(net.get("fetch_allow_domains"), _NET_DEFAULTS.fetch_allow_domains),
            fetch_deny_domains=_to_tuple(net.get("fetch_deny_domains"), _NET_DEFAULTS.fetch_deny_domains),
            block_private_ranges=net.get("block_private_ranges", True),
        ),
        tools=ToolPolicy(
            allowed=_to_tuple(tools.get("allowed"), _TOOL_DEFAULTS.allowed),
            denied=_to_tuple(tools.get("denied"), _TOOL_DEFAULTS.denied),
        ),
        inference=InferencePolicy(
            model_allowlist=_to_tuple(inf.get("model_allowlist"), _INF_DEFAULTS.model_allowlist),
            max_tokens_per_session=inf.get("max_tokens_per_session", 0),
            system_prompt_immutable=inf.get("system_prompt_immutable", False),
        ),
        credentials=CredentialPolicy(
            redact_patterns=_to_tuple(cred.get("redact_patterns")),
            redact_env_vars=_to_tuple(cred.get("redact_env_vars")),
        ),
        audit=AuditPolicy(
            enabled=audit.get("enabled", True),
            log_tool_args=audit.get("log_tool_args", True),
            log_result_preview=audit.get("log_result_preview", True),
            retention_days=audit.get("retention_days", 90),
        ),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_schema(data: dict) -> list[str]:
    """Validate raw dict against JSON Schema. Returns list of error messages."""
    try:
        import jsonschema
    except ImportError:
        log.warning("jsonschema not installed — skipping policy schema validation")
        return []

    if not _SCHEMA_PATH.exists():
        log.warning("Schema file not found at %s — skipping validation", _SCHEMA_PATH)
        return []

    schema = json.loads(_SCHEMA_PATH.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    return [f"{'.'.join(str(p) for p in e.absolute_path)}: {e.message}" for e in validator.iter_errors(data)]


# ---------------------------------------------------------------------------
# Loading & singleton
# ---------------------------------------------------------------------------

_current_policy: SecurityPolicy = SecurityPolicy()  # permissive default
_policy_path: Optional[Path] = None


def load_policy(path: str | Path | None = None) -> SecurityPolicy:
    """Load and validate a security policy from YAML.

    Falls back to default (permissive) policy if the file doesn't exist.
    Raises ValueError on schema validation failure.
    """
    global _current_policy, _policy_path

    if path is None:
        # Check env var, then default
        import os
        env_path = os.environ.get("SECURITY_POLICY_PATH")
        if env_path:
            path = Path(env_path)
        else:
            path = _DEFAULT_POLICY_PATH

    path = Path(path)
    _policy_path = path

    if not path.exists():
        log.info("No security policy at %s — using permissive defaults", path)
        _current_policy = SecurityPolicy()
        return _current_policy

    log.info("Loading security policy from %s", path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Security policy must be a YAML mapping, got {type(raw).__name__}")

    errors = _validate_schema(raw)
    if errors:
        raise ValueError(f"Security policy validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    policy = _parse_policy(raw)
    _current_policy = policy
    log.info("Security policy v%d loaded (%d fs rules, %d net rules, %d redact patterns)",
             policy.version,
             len(policy.filesystem.read_allow) + len(policy.filesystem.write_allow),
             len(policy.network.fetch_allow_domains) + len(policy.network.fetch_deny_domains),
             len(policy.credentials.redact_patterns))
    return policy


def get_policy() -> SecurityPolicy:
    """Return the current security policy (singleton)."""
    return _current_policy


def reload_policy() -> SecurityPolicy:
    """Reload policy from the last loaded path. For hot-reload."""
    if _policy_path is None:
        return load_policy()
    return load_policy(_policy_path)


# ---------------------------------------------------------------------------
# Policy matching helpers (used by enforcement modules)
# ---------------------------------------------------------------------------

def matches_glob(path: str, patterns: tuple[str, ...]) -> bool:
    """Check if a path matches any of the given glob patterns.

    Supports ** for recursive directory matching (unlike fnmatch which
    treats ** identically to *). Converts ** patterns to regex.
    """
    for pattern in patterns:
        if "**" in pattern:
            # Convert glob pattern to regex: ** matches any path components
            regex = re.escape(pattern).replace(r"\*\*", ".*").replace(r"\*", "[^/]*").replace(r"\?", ".")
            if re.fullmatch(regex, path):
                return True
        elif fnmatch.fnmatch(path, pattern):
            return True
    return False


def matches_domain(domain: str, patterns: tuple[str, ...]) -> bool:
    """Check if a domain matches any of the given patterns."""
    domain = domain.lower()
    for pattern in patterns:
        pattern = pattern.lower()
        if pattern == "*":
            return True
        if fnmatch.fnmatch(domain, pattern):
            return True
    return False


# ---------------------------------------------------------------------------
# Hot-reload watcher (Phase 3 — started by nexus.py)
# ---------------------------------------------------------------------------

class PolicyWatcher:
    """Watches the policy file for changes and reloads atomically.

    Uses SHA256 comparison to detect changes. Monotonic version counter
    ensures new version > current. Failed validation keeps previous policy
    (last known good).
    """

    def __init__(self, interval: float = 10.0):
        self._interval = interval
        self._last_hash: str = ""
        self._running = False

    def _file_hash(self) -> str:
        if _policy_path and _policy_path.exists():
            return hashlib.sha256(_policy_path.read_bytes()).hexdigest()
        return ""

    async def watch_loop(self) -> None:
        """Async poll loop — run as asyncio.create_task()."""
        self._running = True
        self._last_hash = self._file_hash()
        log.info("Policy watcher started (interval=%ss, path=%s)", self._interval, _policy_path)

        while self._running:
            await asyncio.sleep(self._interval)
            try:
                current_hash = self._file_hash()
                if current_hash and current_hash != self._last_hash:
                    log.info("Policy file changed (hash %s -> %s), reloading...",
                             self._last_hash[:12], current_hash[:12])
                    old_policy = get_policy()
                    try:
                        new_policy = reload_policy()
                        if new_policy.version < old_policy.version:
                            log.warning("New policy v%d < current v%d — rejecting downgrade",
                                        new_policy.version, old_policy.version)
                            # Restore old policy
                            global _current_policy
                            _current_policy = old_policy
                        else:
                            self._last_hash = current_hash
                            log.info("Policy reloaded: v%d -> v%d", old_policy.version, new_policy.version)
                    except Exception as e:
                        log.error("Policy reload failed (keeping LKG v%d): %s", old_policy.version, e)
            except Exception as e:
                log.error("Policy watcher error: %s", e)

    def stop(self) -> None:
        self._running = False
