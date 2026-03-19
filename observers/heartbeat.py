"""Heartbeat checklist observer — reads HEARTBEAT.md and acts on it.

If HEARTBEAT.md is empty or absent, this observer returns silently (no LLM
call, no cost). When items are present, it sends them to the LLM for
evaluation and reports actionable findings via Telegram.
"""

from observers.base import Observer, ObserverContext, ObserverResult


class HeartbeatObserver(Observer):
    name = "heartbeat"
    schedule = "0 8,12,16,20 * * *"  # 4x/day during waking hours

    def run(self, ctx: ObserverContext) -> ObserverResult:
        from memory import get_heartbeat

        checklist = get_heartbeat()
        if not checklist.strip():
            return ObserverResult(success=True)  # Silent — no cost

        prompt = (
            "You are HAL, PureTensor's infrastructure agent. "
            "The following is your heartbeat checklist — items to check right now. "
            "For each item, use tools to check the actual status, then report findings. "
            "Be concise. Only report actionable items.\n\n"
            f"Checklist:\n{checklist}"
        )

        try:
            result = self.call_llm(prompt, model="sonnet")
            response = result.get("result", "").strip()
        except Exception as e:
            return ObserverResult(
                success=False,
                error=f"Heartbeat LLM call failed: {e}",
            )

        if response:
            self.send_telegram(f"*Heartbeat Check*\n\n{response}")

        return ObserverResult(success=True, message=response[:200] if response else "")
