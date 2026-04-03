"""Nightly memory consolidation — prune stale entries, extract patterns, update pgvector.

Runs at 03:00 UTC daily. Reads the last 24h of journal entries + MEMORY.md + LESSONS.md,
identifies patterns, flags stale/contradictory memories, and writes proposals for review.

Auto-applies low-risk changes:
  - Stale journal cleanup (>30 days)
  - pgvector pruning (archived entries older than 90 days)

Writes higher-risk proposals to /data/memory/consolidation-proposals.md for operator review.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from observers.base import Observer, ObserverContext, ObserverResult

log = logging.getLogger("nexus")

MEMORY_DIR = Path(os.environ.get("MEMORY_DIR", "/data/memory"))
JOURNAL_DIR = MEMORY_DIR / "journal"
PROPOSALS_FILE = MEMORY_DIR / "consolidation-proposals.md"
JOURNAL_RETENTION_DAYS = 30
PGVECTOR_ARCHIVE_PRUNE_DAYS = 90

CONSOLIDATION_PROMPT = """\
You are a memory consolidation agent. Review the following data and identify:

1. **Patterns**: Recurring themes across recent journal entries
2. **Stale memories**: Facts in MEMORY.md that may be outdated
3. **New lessons**: Insights from recent work worth adding to LESSONS.md
4. **Contradictions**: Conflicts between stored facts

Recent journal entries (last 24h):
{journals}

Current MEMORY.md:
{memory_md}

Current LESSONS.md:
{lessons_md}

Return a JSON object with:
- "patterns": list of observed patterns (max 3)
- "stale_candidates": list of memory entries that may need updating (max 5, each with "text" and "reason")
- "new_lessons": list of proposed new lessons (max 3)
- "contradictions": list of contradictions found (max 3)
- "summary": one-line summary of consolidation findings

Return ONLY valid JSON, no markdown fences."""


class NightlyConsolidationObserver(Observer):
    name = "nightly_consolidation"
    schedule = "0 3 * * *"  # 03:00 UTC daily

    def _read_recent_journals(self, hours: int = 24) -> str:
        """Read journal entries from the last N hours."""
        if not JOURNAL_DIR.exists():
            return "(no journal directory)"

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        entries = []

        for f in sorted(JOURNAL_DIR.glob("*.md"), reverse=True)[:3]:
            try:
                content = f.read_text().strip()
                if content:
                    entries.append(f"## {f.stem}\n{content}")
            except Exception:
                continue

        return "\n\n".join(entries) if entries else "(no recent journals)"

    def _read_file(self, path: Path, max_chars: int = 4000) -> str:
        """Read a file, truncating if too large."""
        if not path.exists():
            return ""
        content = path.read_text().strip()
        if len(content) > max_chars:
            content = content[:max_chars] + "\n...(truncated)"
        return content

    def _cleanup_old_journals(self) -> int:
        """Delete journal files older than retention period. Returns count deleted."""
        if not JOURNAL_DIR.exists():
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=JOURNAL_RETENTION_DAYS)
        deleted = 0

        for f in JOURNAL_DIR.glob("*.md"):
            try:
                # Parse date from filename (YYYY-MM-DD.md)
                file_date = datetime.strptime(f.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if file_date < cutoff:
                    f.unlink()
                    deleted += 1
            except (ValueError, OSError):
                continue

        return deleted

    def _prune_pgvector_archives(self) -> int:
        """Delete archived pgvector facts older than prune threshold. Returns count deleted."""
        try:
            import asyncio
            from memory_rag import MEMORY_RAG_ENABLED, get_pool
            if not MEMORY_RAG_ENABLED:
                return 0

            async def _prune():
                pool = await get_pool()
                async with pool.acquire() as conn:
                    result = await conn.execute(
                        "DELETE FROM facts WHERE archived = true AND updated_at < now() - interval '%s days'"
                        % PGVECTOR_ARCHIVE_PRUNE_DAYS
                    )
                    # Parse "DELETE N" response
                    return int(result.split()[-1]) if result else 0

            return asyncio.run(_prune())
        except Exception as e:
            log.warning("[consolidation] pgvector prune failed: %s", e)
            return 0

    def _write_proposals(self, analysis: dict):
        """Write consolidation proposals to file for operator review."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [f"# Consolidation Proposals ({timestamp})\n"]
        lines.append(f"**Summary:** {analysis.get('summary', 'N/A')}\n")

        patterns = analysis.get("patterns", [])
        if patterns:
            lines.append("## Patterns Observed")
            for p in patterns:
                lines.append(f"- {p}")
            lines.append("")

        stale = analysis.get("stale_candidates", [])
        if stale:
            lines.append("## Potentially Stale Memories")
            for s in stale:
                lines.append(f"- **{s.get('text', '?')[:80]}...** -- {s.get('reason', '?')}")
            lines.append("")

        lessons = analysis.get("new_lessons", [])
        if lessons:
            lines.append("## Proposed New Lessons")
            for l in lessons:
                lines.append(f"- {l}")
            lines.append("")

        contradictions = analysis.get("contradictions", [])
        if contradictions:
            lines.append("## Contradictions Found")
            for c in contradictions:
                lines.append(f"- {c}")
            lines.append("")

        PROPOSALS_FILE.write_text("\n".join(lines))

    def run(self, ctx: ObserverContext) -> ObserverResult:
        """Run nightly consolidation."""
        # Auto-apply: clean old journals
        journals_deleted = self._cleanup_old_journals()
        if journals_deleted:
            log.info("[consolidation] Cleaned %d old journal files", journals_deleted)

        # Auto-apply: prune archived pgvector entries
        pgvector_pruned = self._prune_pgvector_archives()
        if pgvector_pruned:
            log.info("[consolidation] Pruned %d archived pgvector facts", pgvector_pruned)

        # Read inputs for LLM analysis
        journals = self._read_recent_journals(hours=24)
        memory_md = self._read_file(MEMORY_DIR / "MEMORY.md")
        lessons_md = self._read_file(MEMORY_DIR / "LESSONS.md")

        if not memory_md and not journals:
            return ObserverResult(
                success=True,
                message="Nothing to consolidate",
                data={"journals_deleted": journals_deleted, "pgvector_pruned": pgvector_pruned},
            )

        # LLM analysis
        prompt = CONSOLIDATION_PROMPT.format(
            journals=journals[:3000],
            memory_md=memory_md[:3000],
            lessons_md=lessons_md[:2000],
        )

        try:
            result_text = self.call_llm(prompt, model="haiku", timeout=30)
            if not result_text:
                raise ValueError("Empty LLM response")

            # Parse JSON
            text = result_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            analysis = json.loads(text)
        except Exception as e:
            log.warning("[consolidation] LLM analysis failed: %s", e)
            return ObserverResult(
                success=True,
                message=f"Consolidation auto-cleanup done, LLM analysis failed: {e}",
                data={"journals_deleted": journals_deleted, "pgvector_pruned": pgvector_pruned},
            )

        # Write proposals for operator review
        self._write_proposals(analysis)

        summary = analysis.get("summary", "Consolidation complete")
        stale_count = len(analysis.get("stale_candidates", []))
        lesson_count = len(analysis.get("new_lessons", []))

        # Notify if there are actionable proposals
        if stale_count > 0 or lesson_count > 0:
            msg = (
                f"[Nightly Consolidation] {summary}\n"
                f"- {stale_count} stale memory candidates\n"
                f"- {lesson_count} proposed new lessons\n"
                f"- {journals_deleted} old journals cleaned\n"
                f"- {pgvector_pruned} archived facts pruned\n\n"
                f"Review: /data/memory/consolidation-proposals.md"
            )
            self.send_telegram(msg)

        return ObserverResult(
            success=True,
            message=summary,
            data={
                "journals_deleted": journals_deleted,
                "pgvector_pruned": pgvector_pruned,
                "stale_candidates": stale_count,
                "new_lessons": lesson_count,
            },
        )
