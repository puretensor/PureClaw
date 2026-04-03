"""Act on evaluated signals — dispatches notifications and autonomous actions.

Only invoked when severity >= 2. Uses sonnet for complex reasoning.
"""

import json
import logging

log = logging.getLogger("nexus")

SEVERITY_EMOJI = {0: "", 1: "", 2: "!!", 3: "CRITICAL"}

ACT_PROMPT = """\
You are an infrastructure agent. The monitoring system has flagged issues requiring attention.

Assessment:
{assessment}

Original gathered data summary:
{gathered_summary}

For each recommended action, determine if you can execute it autonomously or if it requires operator approval.
- Autonomous: restart a pod, clear a non-critical alert, run a diagnostic command
- Requires approval: delete data, modify DNS, change service configuration, send external email

Return a JSON object with:
- "message": concise Telegram notification text (max 500 chars, use plain text not markdown)
- "autonomous_actions": list of actions you can safely take now (each: {{"action": str, "command": str}})
- "approval_needed": list of actions requiring operator approval (each: {{"action": str, "reason": str}})

Return ONLY valid JSON."""


def dispatch(gathered: dict, assessment: dict, proactivity: str = "advisor") -> dict:
    """Dispatch actions based on assessment severity and proactivity level.

    Proactivity levels:
        observer  — log only, never notify
        advisor   — notify on severity 2+, never act autonomously
        assistant — notify on severity 2+, act autonomously on severity 3
        partner   — notify on severity 1+, act autonomously on severity 2+

    Returns: {"notified": bool, "actions_taken": list, "message": str}
    """
    severity = assessment.get("severity", 0)
    summary = assessment.get("summary", "")

    # Determine thresholds from proactivity level
    notify_threshold = {"observer": 99, "advisor": 2, "assistant": 2, "partner": 1}.get(proactivity, 2)
    act_threshold = {"observer": 99, "advisor": 99, "assistant": 3, "partner": 2}.get(proactivity, 99)

    result = {"notified": False, "actions_taken": [], "message": summary}

    if severity < notify_threshold:
        log.info("[heartbeat] Severity %d below notify threshold %d (%s) — log only",
                 severity, notify_threshold, proactivity)
        return result

    # Build notification
    severity_label = SEVERITY_EMOJI.get(severity, str(severity))
    findings_text = ""
    for f in assessment.get("findings", [])[:5]:
        findings_text += f"\n- {f.get('item', '?')}"

    actions_text = ""
    for a in assessment.get("recommended_actions", [])[:3]:
        actions_text += f"\n- {a}"

    message = f"[Heartbeat {severity_label}] {summary}"
    if findings_text:
        message += f"\n\nFindings:{findings_text}"
    if actions_text:
        message += f"\n\nRecommended:{actions_text}"

    result["message"] = message
    result["notified"] = True

    # Autonomous action (only at act_threshold)
    if severity >= act_threshold:
        try:
            from engine import call_sync

            gathered_summary = {k: v for k, v in gathered.items()
                                if k not in ("gathered_at", "duration_ms")
                                and isinstance(v, dict)
                                and v.get("error") is None}

            prompt = ACT_PROMPT.format(
                assessment=json.dumps(assessment, indent=2, default=str),
                gathered_summary=json.dumps(gathered_summary, indent=2, default=str),
            )

            llm_result = call_sync(prompt, model="sonnet", timeout=60)
            text = llm_result.get("result", "").strip()

            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            plan = json.loads(text)

            # Execute autonomous actions
            for action in plan.get("autonomous_actions", []):
                cmd = action.get("command", "")
                if cmd:
                    log.info("[heartbeat] Autonomous action: %s", action.get("action", cmd))
                    result["actions_taken"].append(action.get("action", cmd))

            # Append approval-needed items to message
            for item in plan.get("approval_needed", []):
                message += f"\n\n[Needs approval] {item.get('action', '?')}: {item.get('reason', '')}"

            # Use the LLM's crafted message if available
            if plan.get("message"):
                result["message"] = plan["message"]
                if findings_text:
                    result["message"] += f"\n\nFindings:{findings_text}"

        except Exception as e:
            log.warning("[heartbeat] Action dispatch failed: %s", e)
            result["message"] += f"\n\n(Action planning failed: {e})"

    return result
