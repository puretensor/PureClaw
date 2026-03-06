# Handover: Bedrock Empty Response Bug

## Problem

HAL (PureClaw Telegram bot) returns "(empty response)" when users send messages. The bot is running on K3s (fox-n1) using the `bedrock_api` backend with `us.anthropic.claude-sonnet-4-6`.

## Symptoms

- User sends "hello HAL" via Telegram
- Bot shows "us.anthropic.claude-sonnet-4-6 processing... (new session)"
- Bot responds with "(empty response)"
- Logs show: `[bedrock-stream] Bedrock usage: in=11962 (cache_read=0, cache_write=6935) out=63 cost=$0.0368`
- 63 output tokens consumed but no visible text returned

## Root Cause (diagnosed, not yet fixed)

The `_build_converse_kwargs()` method in `backends/bedrock_api.py` enables extended thinking:

```python
kwargs["additionalModelRequestFields"] = {
    "thinking": {"type": "enabled", "budget_tokens": 10000},
}
```

With thinking enabled, the model spends output tokens on internal reasoning. If all tokens go to thinking and the text response is empty (or just whitespace), the stream consumer returns `text=""` and the tool loop returns `"(empty response)"`.

The flow:
1. `_consume_stream()` collects `text_parts` only from `"text"` type blocks
2. Thinking blocks go into `anthropic_blocks` but NOT into `text_parts`
3. If the response is all thinking + empty/whitespace text: `"\n".join(text_parts).strip()` → `""`
4. Tool loop at `backends/tools.py:1412` returns `last_text or "(empty response)"`

## What I Tried (and failed)

1. **`"type": "adaptive"` with `"budgetTokens": 10000`** → Bedrock error: `thinking.adaptive.budgetTokens: Extra inputs are not permitted`
2. **`"type": "enabled"` with `"budgetTokens": 10000`** → Bedrock error: `thinking.enabled.budget_tokens: Field required` (wrong casing — Bedrock uses snake_case in `additionalModelRequestFields`)
3. **`"type": "enabled"` with `"budget_tokens": 10000`** → Currently deployed but UNTESTED (user hasn't sent a message yet to verify)

## Key Files

- **`backends/bedrock_api.py`** — The Bedrock backend. Key methods:
  - `_build_converse_kwargs()` (~line 475) — Builds the converse request, sets thinking config
  - `_consume_stream()` (~line 295) — Parses streaming response events
  - `call_streaming()` (~line 654) — Entry point for Telegram messages
  - `call_sync()` (~line 565) — Entry point for observer/background calls

- **`backends/tools.py`** — Shared tool loop:
  - `run_tool_loop_async()` (line 1364) — Returns `"(empty response)"` when `last_text` is empty

- **`config.py`** — `BEDROCK_MAX_TOKENS = 64000`, `BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"`

## Debug Logging Already Added

I added debug logging to `_consume_stream()` that logs:
- Each stream event's keys
- Final result: text_parts count/chars, tool_calls count, blocks count, stop_reason
- Each block type with content length (thinking chars, text preview, tool_use name)

These will appear at INFO level in pod logs: `ssh fox-n1 'kubectl logs -n nexus deploy/nexus --tail=50'`

## Possible Fixes to Try

1. **The current deploy might work** — `"type": "enabled", "budget_tokens": 10000` is deployed but untested. Send a Telegram message to @hal_claude_bot to test.

2. **If thinking still causes empty responses**, consider:
   - Switch back to `"type": "adaptive"` WITHOUT any budgetTokens (this was the original config that produced the empty response, but maybe intermittently)
   - Remove thinking entirely: delete the `additionalModelRequestFields` block
   - Add a fallback: if `text_parts` is empty but `anthropic_blocks` has thinking, extract the thinking text as the response (ugly but functional)

3. **If Bedrock rejects the snake_case too**, the `additionalModelRequestFields` format may differ by model. Check the AWS Bedrock docs for the exact schema for `us.anthropic.claude-sonnet-4-6` thinking parameters.

## How to Deploy

```bash
cd ~/nexus && bash k8s/deploy.sh
```

This rebuilds the container image on fox-n1, imports to k3s containerd, and restarts the pod. Takes ~2 minutes.

## How to Test

1. Send any message to @hal_claude_bot on Telegram
2. Check logs: `ssh fox-n1 'kubectl logs -n nexus deploy/nexus --tail=50' 2>&1 | grep -v getUpdates`
3. Look for `[bedrock-stream] Result:` lines showing text_parts count and block breakdown

## Context

- Pod: K3s namespace `nexus` on fox-n1 (100.103.248.9)
- Image: `nexus:v2.0.0` (rebuilt each deploy)
- The bot was working before — this may be a regression from a Bedrock API change, or the thinking feature was added recently and never worked correctly for simple messages.
