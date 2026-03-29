"""Backend factory — lazy instances keyed by backend name."""

import logging

log = logging.getLogger("nexus")

_REGISTRY = {
    "ollama": ("backends.ollama", "OllamaBackend"),
    "claude_code": ("backends.claude_code", "ClaudeCodeBackend"),
    "anthropic_api": ("backends.anthropic_api", "AnthropicAPIBackend"),
    "bedrock_api": ("backends.bedrock_api", "BedrockAPIBackend"),
    "gemini_api": ("backends.gemini_api", "GeminiAPIBackend"),
    "codex_cli": ("backends.codex_cli", "CodexCLIBackend"),
    "gemini_cli": ("backends.gemini_cli", "GeminiCLIBackend"),
    "hybrid": ("backends.hybrid", "HybridBackend"),
    "vllm": ("backends.vllm", "VLLMBackend"),
}

CLI_BACKENDS = frozenset({"claude_code", "codex_cli", "gemini_cli"})

_backend_instances: dict[str, object] = {}


def get_backend(name: str | None = None):
    """Return a lazily initialized backend instance.

    If no name is provided, the current config.ENGINE_BACKEND is used.
    Each backend is cached independently so per-session routing does not
    mutate process-global state.
    """
    if name is None:
        from config import ENGINE_BACKEND
        name = ENGINE_BACKEND

    if name in _backend_instances:
        return _backend_instances[name]

    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown ENGINE_BACKEND: {name!r}. "
            f"Valid options: {', '.join(sorted(_REGISTRY))}"
        )

    module_path, class_name = _REGISTRY[name]

    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    instance = cls()

    _backend_instances[name] = instance
    log.info("Initialized backend: %s", instance.name)
    return instance


def reset_backend(name: str | None = None):
    """Reset one cached backend, or all cached backends when name is None."""
    if name is None:
        _backend_instances.clear()
        return
    _backend_instances.pop(name, None)
