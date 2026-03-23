"""Shared cloud LLM callers — Azure OpenAI, xAI Grok, Bedrock (Claude), DeepSeek.

Used by intel_deep_analysis for the AI council (parallel significance scoring).
All callers use only urllib/openai/boto3 — no heavy SDK dependencies.

Env vars:
    AZURE_OPENAI_API_KEY       — Azure OpenAI API key
    AZURE_OPENAI_ENDPOINT      — Azure OpenAI endpoint URL
    AZURE_OPENAI_API_VERSION   — Azure OpenAI API version (default: 2024-12-01-preview)
    GOOGLE_API_KEY             — Google AI / Gemini API key (deep research only)
    GEMINI_API_KEY             — Google AI / Gemini API key (fallback, deep research only)
    XAI_API_KEY                — xAI API key (Grok)
    AWS_ACCESS_KEY_ID          — AWS Bedrock access key
    AWS_SECRET_ACCESS_KEY      — AWS Bedrock secret key
    DEEPSEEK_API_KEY           — DeepSeek API key
"""

import json
import logging
import os
import re
import urllib.request

log = logging.getLogger("nexus")

# Lazy-init Azure OpenAI client
_azure_client = None

AZURE_DEPLOYMENT = "gpt-5-1-chat"


def _get_azure_client():
    global _azure_client
    if _azure_client is None:
        from openai import AzureOpenAI
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        if not api_key or not endpoint:
            raise RuntimeError("AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT not set")
        _azure_client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
    return _azure_client


def call_azure_openai(system_prompt: str, user_prompt: str, timeout: int = 60,
                      temperature: float = 0.3,
                      deployment: str = AZURE_DEPLOYMENT) -> str:
    """Call Azure OpenAI via chat completions. Returns text content."""
    client = _get_azure_client()
    kwargs = dict(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=4096,
        timeout=timeout,
    )
    # GPT-5.1 models only support default temperature (1)
    if not deployment.startswith("gpt-5"):
        kwargs["temperature"] = temperature
    response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


# Backward-compatible aliases — all text generation now routes through Azure OpenAI
call_gemini_flash = call_azure_openai
call_claude_bedrock = call_azure_openai
call_claude_bedrock_haiku = call_azure_openai


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
call_claude_haiku = call_azure_openai


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
