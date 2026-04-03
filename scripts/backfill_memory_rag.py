#!/usr/bin/env python3
"""Backfill existing markdown memories into pgvector.

Parses MEMORY.md, LESSONS.md, CONTEXT.md, SOUL.md, USER.md, and topic files,
then embeds and stores each entry in the nexus_memory.facts table.

Usage:
    python3 scripts/backfill_memory_rag.py [--dry-run]

Requires MEMORY_RAG_ENABLED=true in environment (or .env).
"""

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

# Add project root to path
project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir))

# Load .env
env_path = project_dir / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

# Force enable for backfill
os.environ["MEMORY_RAG_ENABLED"] = "true"


def parse_bullet_lines(content: str) -> list[str]:
    """Extract lines starting with '- ' from markdown content."""
    return [line[2:].strip() for line in content.splitlines() if line.startswith("- ") and len(line) > 3]


def parse_sections(content: str) -> list[str]:
    """Extract meaningful sections from structured markdown (CONTEXT.md, SOUL.md)."""
    sections = []
    current = []

    for line in content.splitlines():
        if line.startswith("# ") or line.startswith("## "):
            if current:
                text = "\n".join(current).strip()
                if len(text) > 20:
                    sections.append(text)
                current = []
            current.append(line)
        elif line.strip():
            current.append(line)

    if current:
        text = "\n".join(current).strip()
        if len(text) > 20:
            sections.append(text)

    return sections


async def backfill(dry_run: bool = False):
    from memory import MEMORY_DIR, MEMORY_MD, CONTEXT_MD, LESSONS_MD, SOUL_MD, USER_MD
    from memory import _read_md, _SYSTEM_FILES
    from memory_rag import store_fact, fact_count, get_pool

    print(f"Memory directory: {MEMORY_DIR}")
    print(f"RAG enabled: {os.environ.get('MEMORY_RAG_ENABLED')}")

    if not dry_run:
        # Verify connection
        pool = await get_pool()
        count = await fact_count()
        print(f"Existing facts in pgvector: {count}")

    entries = []

    # 1. MEMORY.md bullet lines
    content = _read_md(MEMORY_MD)
    for line in parse_bullet_lines(content):
        # Try to extract category from [bracket] prefix
        m = re.match(r"\[(\w+)\]\s*(.*)", line)
        if m:
            entries.append({"content": m.group(2), "source": "MEMORY.md", "category": m.group(1).lower()})
        else:
            entries.append({"content": line, "source": "MEMORY.md", "category": "general"})

    # 2. LESSONS.md bullet lines
    content = _read_md(LESSONS_MD)
    for line in parse_bullet_lines(content):
        entries.append({"content": line, "source": "LESSONS.md", "category": "lesson"})

    # 3. CONTEXT.md sections (structured context, not bullet lines)
    content = _read_md(CONTEXT_MD)
    for section in parse_sections(content):
        entries.append({"content": section[:2000], "source": "CONTEXT.md", "category": "infrastructure", "importance": 7})

    # 4. SOUL.md sections
    content = _read_md(SOUL_MD)
    for section in parse_sections(content):
        entries.append({"content": section[:2000], "source": "SOUL.md", "category": "general", "importance": 8})

    # 5. USER.md sections
    content = _read_md(USER_MD)
    for section in parse_sections(content):
        entries.append({"content": section[:2000], "source": "USER.md", "category": "preference", "importance": 8})

    # 6. Topic files
    if MEMORY_DIR.exists():
        for f in sorted(MEMORY_DIR.glob("*.md")):
            if f.name in _SYSTEM_FILES:
                continue
            topic_content = _read_md(f)
            for line in parse_bullet_lines(topic_content):
                entries.append({"content": line, "source": f.stem, "category": f.stem.replace("-", "_")})

    print(f"\nParsed {len(entries)} entries to backfill:")
    sources = {}
    for e in entries:
        sources[e["source"]] = sources.get(e["source"], 0) + 1
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")

    if dry_run:
        print("\n[DRY RUN] No entries stored.")
        return

    # Store each entry
    stored = 0
    skipped = 0
    failed = 0

    for i, entry in enumerate(entries):
        try:
            fact_id = await store_fact(
                content=entry["content"],
                source=entry["source"],
                category=entry["category"],
                importance=entry.get("importance", 5),
                channel="backfill",
            )
            if fact_id:
                stored += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            print(f"  ERROR entry {i}: {exc}")

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(entries)} (stored={stored}, skipped={skipped}, failed={failed})")

    final_count = await fact_count()
    print(f"\nBackfill complete: stored={stored}, skipped={skipped}, failed={failed}")
    print(f"Total facts in pgvector: {final_count}")


def main():
    parser = argparse.ArgumentParser(description="Backfill markdown memories into pgvector")
    parser.add_argument("--dry-run", action="store_true", help="Parse and count without storing")
    args = parser.parse_args()

    asyncio.run(backfill(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
