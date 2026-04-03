"""Evaluate gathered signals — calls LLM to assess severity and recommend actions.

Uses the cheapest available backend (haiku via engine failover).
No autonomous actions here — just assessment.
"""

import json
import logging

log = logging.getLogger("nexus")

EVALUATE_PROMPT = """\
You are an infrastructure monitoring agent evaluating system signals.
Analyze the following gathered data and return a JSON object with:
- "severity": integer 0-3 (0=all clear, 1=minor, 2=needs attention, 3=critical)
- "summary": one-line summary of overall status
- "findings": list of notable findings (max 5), each with "item" and "severity" (0-3)
- "recommended_actions": list of suggested actions (empty if severity 0)

Severity guide:
- 0: Everything nominal. No action needed.
- 1: Minor issues that don't need immediate attention (e.g. a few down probe targets that are expected to be off).
- 2: Issues that the operator should know about (e.g. unhealthy pods, disk usage >80%, multiple alerts).
- 3: Critical — immediate attention required (e.g. Ceph degraded, multiple critical pods down, data loss risk).

Be concise. Only flag genuinely actionable items. Nodes/services that are intentionally powered off are severity 0.

Gathered data:
{data}

Return ONLY valid JSON, no markdown fences."""


def evaluate_signals(gathered: dict) -> dict:
    """Evaluate gathered data via LLM. Returns severity assessment.

    Returns:
        {"severity": int, "summary": str, "findings": list, "recommended_actions": list}
    """
    from engine import call_sync

    # Strip large nested structures to fit context
    compact = {}
    for key, val in gathered.items():
        if key in ("gathered_at", "duration_ms"):
            compact[key] = val
        elif isinstance(val, dict):
            # Remove verbose nested lists, keep summaries
            slim = {}
            for k, v in val.items():
                if k == "namespaces":
                    # Summarize: only show namespaces with issues
                    if isinstance(v, dict):
                        issues = {ns: s for ns, s in v.items() if s.get("not_ready", 0) > 0}
                        slim["namespaces_with_issues"] = issues if issues else "none"
                elif k == "endpoints":
                    # Summarize: only show offline endpoints
                    if isinstance(v, dict):
                        offline = {n: s for n, s in v.items() if s.get("status") != "online"}
                        slim["offline_endpoints"] = offline if offline else "none"
                elif k in ("recently_updated", "recent_events"):
                    slim[k] = len(v) if isinstance(v, list) else v
                else:
                    slim[k] = v
            compact[key] = slim

    prompt = EVALUATE_PROMPT.format(data=json.dumps(compact, indent=2, default=str))

    try:
        result = call_sync(prompt, model="haiku", timeout=30)
        text = result.get("result", "").strip()

        # Parse JSON from response (handle markdown fences)
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        assessment = json.loads(text)

        # Validate required fields
        assessment.setdefault("severity", 0)
        assessment.setdefault("summary", "No summary")
        assessment.setdefault("findings", [])
        assessment.setdefault("recommended_actions", [])
        assessment["severity"] = max(0, min(3, int(assessment["severity"])))

        return assessment

    except json.JSONDecodeError as e:
        log.warning("[evaluate] Failed to parse LLM response as JSON: %s", e)
        return {
            "severity": 1,
            "summary": "Evaluation parse error — manual review recommended",
            "findings": [{"item": f"LLM response was not valid JSON: {e}", "severity": 1}],
            "recommended_actions": ["Review gathered data manually"],
            "raw_response": text[:500] if text else "",
        }
    except Exception as e:
        log.warning("[evaluate] LLM call failed: %s", e)
        return {
            "severity": 1,
            "summary": f"Evaluation failed: {e}",
            "findings": [],
            "recommended_actions": [],
        }
