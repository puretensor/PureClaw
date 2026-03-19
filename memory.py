"""Persistent memory layer for PureClaw — markdown-based file store at /data/memory/.

Engine-agnostic: all backends (claude_code, bedrock, codex, gemini, vllm, ollama)
get identical context via get_memories_for_injection().

Canonical store: /data/memory/ (PVC-backed)
  CONTEXT.md  — fleet topology, services, credentials, preferences
  LESSONS.md  — accumulated lessons learned (append-only)
  MEMORY.md   — runtime memories from save_memory() tool
  *.md        — topic-specific detail files

Public API:
  save_memory(text, topic=None)      — append to MEMORY.md or topic file
  update_memory(old_text, new_text, topic=None) — find-and-replace
  remove_memory(text_or_index)       — remove matching line from MEMORY.md
  list_memories()                    — parse MEMORY.md bullet lines
  list_topic_files()                 — list *.md topic files
  read_topic_file(name)              — read a topic file
  search_memories(query)             — search across all memory files
  get_memories_for_injection()       — CONTEXT + LESSONS + MEMORY for prompt injection
  get_shared_context()               — synced PureClaw MEMORY.md (read-only)
  memory_count()                     — count bullet lines in MEMORY.md
  add_memory(text, category)         — compat shim → save_memory()
"""

import json
import logging
import os
import re
from pathlib import Path

from config import log, AGENT_NAME

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MEMORY_DIR = Path(os.environ.get("MEMORY_DIR", "/data/memory"))
MEMORY_MD = MEMORY_DIR / "MEMORY.md"
CONTEXT_MD = MEMORY_DIR / "CONTEXT.md"
LESSONS_MD = MEMORY_DIR / "LESSONS.md"
SHARED_CONTEXT_PATH = Path(os.environ.get(
    "SHARED_CONTEXT_PATH", "/data/sync/pureclaw_memory.md"
))

# System files — not user topic files, excluded from list_topic_files()
_SYSTEM_FILES = {"MEMORY.md", "CONTEXT.md", "LESSONS.md", "HEARTBEAT.md"}

MAX_MEMORY_LINES = 200
MAX_CONTEXT_LINES = 500
MAX_TOPIC_FILE_SIZE = 50 * 1024  # 50KB

# Legacy path for migration
_LEGACY_JSON_PATH = Path(os.environ.get(
    "MEMORY_PATH", os.path.expanduser("~/.hal/memory.json")
))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir():
    """Create memory directory if needed."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_write(path: Path, content: str):
    """Write file atomically via tmp+rename."""
    _ensure_dir()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _read_md(path: Path) -> str:
    """Read a markdown file, returning empty string if missing."""
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        log.warning("Failed to read %s: %s", path, exc)
        return ""


def _bullet_lines(content: str) -> list[str]:
    """Extract lines starting with '- ' from markdown content."""
    return [line for line in content.splitlines() if line.startswith("- ")]


def _sanitize_topic(name: str) -> str:
    """Sanitize topic name to a safe filename."""
    name = re.sub(r"[^a-z0-9_-]", "", name.lower().strip())
    if not name:
        name = "general"
    if not name.endswith(".md"):
        name += ".md"
    return name


# ---------------------------------------------------------------------------
# Migration (idempotent — runs on first import)
# ---------------------------------------------------------------------------

def _migrate_from_json():
    """Convert legacy memory.json to MEMORY.md if needed."""
    if MEMORY_MD.exists():
        return  # Already migrated
    if not _LEGACY_JSON_PATH.exists():
        return  # Nothing to migrate

    try:
        data = json.loads(_LEGACY_JSON_PATH.read_text(encoding="utf-8"))
        memories = data.get("memories", {})
        if not memories:
            return

        lines = [f"# {AGENT_NAME} Memory", ""]
        for mem in memories.values():
            text = mem.get("text", "")
            cat = mem.get("category", "general")
            if text:
                lines.append(f"- [{cat}] {text}")

        _ensure_dir()
        _atomic_write(MEMORY_MD, "\n".join(lines) + "\n")

        # Rename old file (preserve, don't delete)
        migrated = _LEGACY_JSON_PATH.with_suffix(".json.migrated")
        _LEGACY_JSON_PATH.rename(migrated)
        log.info("Migrated %d memories from JSON to MEMORY.md", len(memories))
    except Exception as exc:
        log.warning("Migration from memory.json failed: %s", exc)


# Run migration on import
_migrate_from_json()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_memory(text: str, topic: str | None = None) -> str:
    """Append a bullet line to MEMORY.md or a topic file.

    Returns the text that was saved.
    """
    text = text.strip()
    if not text:
        return ""

    if topic:
        filename = _sanitize_topic(topic)
        path = MEMORY_DIR / filename
        existing = _read_md(path)

        # Check size limit
        if len(existing.encode("utf-8")) >= MAX_TOPIC_FILE_SIZE:
            log.warning("Topic file %s exceeds %d bytes, not appending", filename, MAX_TOPIC_FILE_SIZE)
            return text

        if not existing:
            content = f"# {topic.title()}\n\n- {text}\n"
        else:
            content = existing.rstrip("\n") + f"\n- {text}\n"

        _atomic_write(path, content)
    else:
        existing = _read_md(MEMORY_MD)
        if not existing:
            content = f"# {AGENT_NAME} Memory\n\n- {text}\n"
        else:
            content = existing.rstrip("\n") + f"\n- {text}\n"

        line_count = len(content.splitlines())
        if line_count > MAX_MEMORY_LINES:
            log.warning("MEMORY.md has %d lines (limit %d)", line_count, MAX_MEMORY_LINES)

        _atomic_write(MEMORY_MD, content)

    return text


def update_memory(old_text: str, new_text: str, topic: str | None = None) -> bool:
    """Find-and-replace text in MEMORY.md or a topic file. Returns True if replaced."""
    if not old_text or not new_text:
        return False

    if topic:
        path = MEMORY_DIR / _sanitize_topic(topic)
    else:
        path = MEMORY_MD

    content = _read_md(path)
    if not content or old_text not in content:
        return False

    updated = content.replace(old_text, new_text, 1)
    _atomic_write(path, updated)
    return True


def remove_memory(text_or_index) -> bool:
    """Remove a matching line from MEMORY.md.

    Accepts either:
    - A string: removes first line containing that text
    - An int: removes the Nth bullet line (1-indexed)

    Returns True if a line was removed.
    """
    content = _read_md(MEMORY_MD)
    if not content:
        return False

    lines = content.splitlines()

    if isinstance(text_or_index, int):
        bullets = [(i, line) for i, line in enumerate(lines) if line.startswith("- ")]
        idx = text_or_index - 1  # 1-indexed
        if 0 <= idx < len(bullets):
            line_idx = bullets[idx][0]
            lines.pop(line_idx)
            _atomic_write(MEMORY_MD, "\n".join(lines) + "\n")
            return True
        return False

    # String match — find first line containing the text
    text = str(text_or_index)
    for i, line in enumerate(lines):
        if text in line:
            lines.pop(i)
            _atomic_write(MEMORY_MD, "\n".join(lines) + "\n")
            return True
    return False


def list_memories() -> list[dict]:
    """Parse MEMORY.md bullet lines into a list of dicts.

    Returns: [{"text": "...", "line_num": N}, ...]
    """
    content = _read_md(MEMORY_MD)
    if not content:
        return []

    result = []
    for i, line in enumerate(content.splitlines(), 1):
        if line.startswith("- "):
            result.append({"text": line[2:], "line_num": i})
    return result


def list_topic_files() -> list[dict]:
    """List topic files in MEMORY_DIR (excluding MEMORY.md).

    Returns: [{"name": "...", "size": N}, ...]
    """
    if not MEMORY_DIR.exists():
        return []

    result = []
    for f in sorted(MEMORY_DIR.glob("*.md")):
        if f.name in _SYSTEM_FILES:
            continue
        result.append({
            "name": f.stem,
            "size": f.stat().st_size,
        })
    return result


def read_topic_file(name: str) -> str:
    """Read a specific topic file's content."""
    filename = _sanitize_topic(name)
    path = MEMORY_DIR / filename
    return _read_md(path)


def search_memories(query: str) -> list[dict]:
    """Case-insensitive search across MEMORY.md and all topic files.

    Returns: [{"text": "...", "source": "MEMORY.md" or topic name}, ...]
    """
    if not query:
        return []

    q = query.lower()
    results = []

    # Search system files (CONTEXT, LESSONS, MEMORY)
    for sys_file in (CONTEXT_MD, LESSONS_MD, MEMORY_MD):
        content = _read_md(sys_file)
        for line in content.splitlines():
            if q in line.lower():
                results.append({"text": line, "source": sys_file.name})

    # Search topic files
    if MEMORY_DIR.exists():
        for f in MEMORY_DIR.glob("*.md"):
            if f.name in _SYSTEM_FILES:
                continue
            topic_content = _read_md(f)
            for line in topic_content.splitlines():
                if q in line.lower():
                    results.append({"text": line, "source": f.stem})

    return results


def get_memories_for_injection() -> str:
    """Return all canonical memory (CONTEXT + LESSONS + MEMORY) for prompt injection.

    This is the single injection point for all engine backends. Every backend
    (claude_code, bedrock, codex, gemini, vllm, ollama) gets identical context.

    Returns empty string if no files exist.
    """
    parts = []

    # 1. Operational context (fleet topology, services, credentials, preferences)
    context = _read_md(CONTEXT_MD)
    if context.strip():
        ctx_lines = context.splitlines()
        if len(ctx_lines) > MAX_CONTEXT_LINES:
            ctx_lines = ctx_lines[:MAX_CONTEXT_LINES]
            ctx_lines.append(f"\n... (truncated at {MAX_CONTEXT_LINES} lines)")
        parts.append(f"[{AGENT_NAME} Context]\n" + "\n".join(ctx_lines))

    # 2. Lessons learned (accumulated corrections and patterns)
    lessons = _read_md(LESSONS_MD)
    if lessons.strip():
        parts.append(f"[{AGENT_NAME} Lessons]\n" + lessons)

    # 3. Runtime memories (save_memory() entries)
    memory = _read_md(MEMORY_MD)
    if memory.strip():
        mem_lines = memory.splitlines()
        if len(mem_lines) > MAX_MEMORY_LINES:
            mem_lines = mem_lines[:MAX_MEMORY_LINES]
            mem_lines.append(f"\n... (truncated at {MAX_MEMORY_LINES} lines)")
        parts.append(f"[{AGENT_NAME} Memory]\n" + "\n".join(mem_lines))

    # 4. Recent journal entries
    try:
        journal = get_recent_journals(days=2)
        if journal.strip():
            parts.append(f"[{AGENT_NAME} Journal]\n" + journal)
    except Exception:
        pass

    return "\n\n".join(parts)


def memory_count() -> int:
    """Return number of bullet lines in MEMORY.md."""
    content = _read_md(MEMORY_MD)
    return len(_bullet_lines(content))


# ---------------------------------------------------------------------------
# Backward compatibility shim
# ---------------------------------------------------------------------------

def add_memory(text: str, category: str = "general") -> str:
    """Compat shim for /remember command. Maps to save_memory().

    Returns the saved text (old API returned a key, but callers only use it for display).
    """
    topic = category if category != "general" else None
    return save_memory(text, topic=topic)


# ---------------------------------------------------------------------------
# Shared context (PureClaw instance sync)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Daily journal subsystem
# ---------------------------------------------------------------------------

JOURNAL_DIR = MEMORY_DIR / "journal"
JOURNAL_MAX_AGE_DAYS = 30
_last_purge_date: str | None = None


def journal_append(entry: str) -> None:
    """Append a timestamped line to today's journal file."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%H:%M")

    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    path = JOURNAL_DIR / f"{today}.md"

    line = f"- [{timestamp}] {entry}\n"
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as exc:
        log.warning("Failed to write journal entry: %s", exc)

    # Lazy purge: once per day
    global _last_purge_date
    if _last_purge_date != today:
        _last_purge_date = today
        try:
            purge_old_journals()
        except Exception as exc:
            log.warning("Journal purge failed: %s", exc)


def get_journal(date_str: str | None = None) -> str:
    """Read a specific day's journal. Default: today."""
    from datetime import datetime, timezone
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = JOURNAL_DIR / f"{date_str}.md"
    return _read_md(path)


def get_recent_journals(days: int = 2) -> str:
    """Return concatenated journal entries for the last N days."""
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    parts = []
    for i in range(days):
        date = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        content = get_journal(date)
        if content.strip():
            parts.append(f"## {date}\n{content}")
    return "\n".join(parts)


def purge_old_journals(max_age_days: int = JOURNAL_MAX_AGE_DAYS) -> int:
    """Delete journal files older than max_age_days. Returns count deleted."""
    from datetime import datetime, timezone, timedelta
    if not JOURNAL_DIR.exists():
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    deleted = 0
    for f in JOURNAL_DIR.glob("*.md"):
        if f.stem < cutoff_str:
            try:
                f.unlink()
                deleted += 1
            except Exception:
                pass
    if deleted:
        log.info("Purged %d old journal files", deleted)
    return deleted


# ---------------------------------------------------------------------------
# Heartbeat checklist
# ---------------------------------------------------------------------------

HEARTBEAT_MD = MEMORY_DIR / "HEARTBEAT.md"


def get_heartbeat() -> str:
    """Read the heartbeat checklist file."""
    return _read_md(HEARTBEAT_MD)


def save_heartbeat(content: str) -> None:
    """Write the heartbeat checklist file."""
    _ensure_dir()
    _atomic_write(HEARTBEAT_MD, content)


def get_shared_context() -> str:
    """Read the synced PureClaw MEMORY.md for injection into the agent's prompt."""
    if not SHARED_CONTEXT_PATH.exists():
        return ""
    try:
        text = SHARED_CONTEXT_PATH.read_text(encoding="utf-8").strip()
        return f"[PureClaw Shared Memory]\n{text}" if text else ""
    except Exception as exc:
        log.warning("Failed to read shared context: %s", exc)
        return ""
