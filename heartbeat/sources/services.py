"""Service health checks — vLLM, Ollama, Whisper, TTS, Vision."""

import os

import httpx

_ENDPOINTS = {
    "vllm": os.environ.get("VLLM_URL", "").replace("/v1", ""),
    "ollama": os.environ.get("OLLAMA_URL", ""),
    "whisper": os.environ.get("WHISPER_URL", "").rsplit("/", 1)[0] if os.environ.get("WHISPER_URL") else "",
    "tts": os.environ.get("TTS_URL", ""),
    "vision": os.environ.get("VISION_URL", "").replace("/v1", "") if os.environ.get("VISION_ENABLED", "").lower() == "true" else "",
}


async def _check(client: httpx.AsyncClient, name: str, url: str) -> dict:
    """Check a single endpoint. Returns {name, status, latency_ms}."""
    if not url:
        return {"name": name, "status": "unconfigured"}
    try:
        r = await client.get(url, timeout=5.0)
        return {"name": name, "status": "online", "http_code": r.status_code}
    except httpx.ConnectError:
        return {"name": name, "status": "offline", "error": "connection_refused"}
    except httpx.TimeoutException:
        return {"name": name, "status": "offline", "error": "timeout"}
    except Exception as e:
        return {"name": name, "status": "offline", "error": str(e)}


async def collect() -> dict:
    """Check all TC service endpoints."""
    async with httpx.AsyncClient() as client:
        checks = {name: _check(client, name, url) for name, url in _ENDPOINTS.items() if url}
        results = {}
        for name, coro in checks.items():
            results[name] = await coro

    online = sum(1 for r in results.values() if r.get("status") == "online")
    return {
        "online": online,
        "total": len(results),
        "endpoints": results,
    }
