"""Shared cloud LLM callers — Gemini (Google), xAI Grok, Bedrock (Claude), DeepSeek.

Used by intel_deep_analysis for the AI council (parallel significance scoring).
All callers use only urllib/google-genai/boto3 — no heavy SDK dependencies.

Env vars:
    GOOGLE_API_KEY         — Google AI / Gemini API key (primary)
    GEMINI_API_KEY         — Google AI / Gemini API key (fallback)
    XAI_API_KEY            — xAI API key (Grok)
    AWS_ACCESS_KEY_ID      — AWS Bedrock access key
    AWS_SECRET_ACCESS_KEY  — AWS Bedrock secret key
    DEEPSEEK_API_KEY       — DeepSeek API key
"""

import json
import logging
import os
import re
import urllib.request

log = logging.getLogger("nexus")

# Lazy-init Gemini client (google-genai imported on first use)
_gemini_client = None

# Map legacy Bedrock model IDs to Gemini models
_GEMINI_MODEL_MAP = {
    "us.anthropic.claude-sonnet-4-6": "gemini-3-flash-preview",
    "us.anthropic.claude-haiku-4-5-20251001": "gemini-3-flash-preview",
    "us.anthropic.claude-opus-4-6": "gemini-3.1-pro-preview",
    "us.anthropic.claude-opus-4-6-v1": "gemini-3.1-pro-preview",
}


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY / GEMINI_API_KEY not set")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _resolve_model(model_id: str) -> str:
    """Resolve legacy Bedrock model IDs to Gemini model names."""
    return _GEMINI_MODEL_MAP.get(model_id, model_id)


def call_claude_bedrock(system_prompt: str, user_prompt: str, timeout: int = 60,
                        temperature: float = 0.3,
                        model_id: str = "gemini-3-flash-preview") -> str:
    """Call Gemini via google-genai SDK. Returns text content.

    Kept as call_claude_bedrock for backward compatibility with existing callers.
    """
    from google.genai import types
    client = _get_gemini_client()
    model = _resolve_model(model_id)
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=4096,
        system_instruction=system_prompt,
    )
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=config,
    )
    return (response.text or "").strip()


def call_claude_bedrock_haiku(system_prompt: str, user_prompt: str, timeout: int = 60,
                              temperature: float = 0.3) -> str:
    """Call Gemini 3.0 Flash (fast/cheap). Backward-compatible name."""
    return call_claude_bedrock(system_prompt, user_prompt, timeout, temperature,
                               model_id="gemini-3-flash-preview")


# Backward-compatible aliases
call_gemini_flash = call_claude_bedrock_haiku


def call_xai_grok(system_prompt: str, user_prompt: str, timeout: int = 60,
                  temperature: float = 0.3) -> str:
    """Call xAI Grok via Responses API with live web search. Returns text content."""
    api_key = os.environ.get("XAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("XAI_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "PureTensor-Nexus/2.0",
    }
    payload = {
        "model": "grok-4.20-0309-non-reasoning",
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_output_tokens": 4096,
        "temperature": temperature,
        "tools": [{"type": "web_search"}],
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request("https://api.x.ai/v1/responses",
                                data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode())

    # Extract text from Responses API output structure
    for item in result.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    return part.get("text", "").strip()
    return ""


# Lazy-init Bedrock client
_bedrock_client = None


def _get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        import boto3
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
    return _bedrock_client


def call_bedrock_sonnet(system_prompt: str, user_prompt: str, timeout: int = 60,
                        temperature: float = 0.3) -> str:
    """Call Claude Sonnet 4.6 via AWS Bedrock. Returns text content."""
    client = _get_bedrock_client()
    response = client.converse(
        modelId="us.anthropic.claude-sonnet-4-6",
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        system=[{"text": system_prompt}],
        inferenceConfig={"temperature": temperature, "maxTokens": 4096},
    )
    return response["output"]["message"]["content"][0]["text"].strip()


# Backward-compatible alias
call_claude_haiku = call_claude_bedrock_haiku


def call_deepseek(system_prompt: str, user_prompt: str, timeout: int = 60,
                   temperature: float = 0.3) -> str:
    """Call DeepSeek via OpenAI-compatible Chat Completions API. Returns text content."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "max_tokens": 4096,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request("https://api.deepseek.com/chat/completions",
                                data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode())

    choices = result.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "").strip()


def extract_json(text: str) -> dict | list | None:
    """Extract JSON object or array from LLM response text."""
    # Try markdown code block first
    m = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try bare JSON
    for pattern in [r'\{.*\}', r'\[.*\]']:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue
    return None
