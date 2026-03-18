"""Recipient profile management — YAML loading, matching, backend instantiation."""

import importlib
import logging
from pathlib import Path

import yaml

log = logging.getLogger("hal-mail")

# Backend registry — mirrors backends/__init__.py
_BACKEND_REGISTRY = {
    "vllm": ("backends.vllm", "VLLMBackend"),
    "anthropic_api": ("backends.anthropic_api", "AnthropicAPIBackend"),
    "gemini_api": ("backends.gemini_api", "GeminiAPIBackend"),
    "bedrock_api": ("backends.bedrock_api", "BedrockAPIBackend"),
    "ollama": ("backends.ollama", "OllamaBackend"),
}


def _default_profile() -> dict:
    return {
        "backend": "vllm",
        "model": None,
        "personality": "",
        "max_reply_length": 3000,
        "tools_enabled": True,
    }


def load_profiles(path: str | Path) -> dict:
    """Load recipient profiles from YAML file.

    Returns dict with 'default' and 'recipients' keys.
    """
    path = Path(path)
    if not path.exists():
        log.warning("Profiles file not found: %s — using defaults", path)
        return {"default": _default_profile(), "recipients": []}

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    # Validate default
    default = data.get("default", {})
    default.setdefault("backend", "vllm")
    default.setdefault("model", None)
    default.setdefault("personality", "")
    default.setdefault("max_reply_length", 3000)
    default.setdefault("tools_enabled", True)

    # Validate recipients
    recipients = []
    for r in data.get("recipients", []):
        if "match" not in r:
            log.warning("Recipient profile missing 'match' field, skipping")
            continue
        r.setdefault("backend", default["backend"])
        r.setdefault("model", default.get("model"))
        r.setdefault("personality", default["personality"])
        r.setdefault("max_reply_length", default["max_reply_length"])
        r.setdefault("tools_enabled", default["tools_enabled"])
        recipients.append(r)

    return {"default": default, "recipients": recipients}


def match_profile(sender_addr: str, profiles: dict) -> dict:
    """Match sender address to a recipient profile.

    Iterates recipients in order; first `in sender_addr.lower()` match wins.
    Falls back to default profile.
    """
    addr = sender_addr.lower().strip()
    for r in profiles.get("recipients", []):
        pattern = r["match"].lower()
        if pattern in addr:
            return r
    return profiles.get("default", _default_profile())


def create_backends(profiles: dict) -> dict:
    """Instantiate one backend per unique backend type referenced in profiles.

    Returns dict mapping backend name -> backend instance.
    """
    backend_names = {profiles["default"]["backend"]}
    for r in profiles.get("recipients", []):
        backend_names.add(r["backend"])

    backends = {}
    for name in sorted(backend_names):
        if name not in _BACKEND_REGISTRY:
            log.warning("Unknown backend '%s' in profiles, skipping", name)
            continue
        try:
            module_path, class_name = _BACKEND_REGISTRY[name]
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            backends[name] = cls()
            log.info("Initialized backend: %s", name)
        except Exception as e:
            log.error("Failed to initialize backend '%s': %s", name, e)

    return backends
