"""Hybrid RAG retrieval for agent memory — pgvector + BM25 with RRF fusion.

Mirrors Alexandria's search/api.py pattern. Connects to the nexus_memory
database (separate from Vantage) and uses nomic-embed-text (768-dim) via
vLLM for embeddings.

All operations are async (asyncpg). Callers should use asyncio.run() or
await directly if already in an async context.

Gated behind MEMORY_RAG_ENABLED env var. When disabled, all functions
return gracefully without connecting to PostgreSQL.
"""

import asyncio
import hashlib
import logging
import os
from datetime import datetime, timezone

log = logging.getLogger("nexus")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MEMORY_RAG_ENABLED = os.environ.get("MEMORY_RAG_ENABLED", "false").lower() in ("true", "1", "yes")
MEMORY_PG_URL = os.environ.get(
    "MEMORY_PG_URL",
    "postgresql://vantage:vantage@postgres-postgresql.databases.svc:5432/nexus_memory",
)
MEMORY_EMBED_URL = os.environ.get("MEMORY_EMBED_URL", os.environ.get("VLLM_URL", "http://localhost:8200"))
MEMORY_EMBED_MODEL = os.environ.get("MEMORY_EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = 768
RRF_K = 60  # RRF constant (matches Alexandria)

# ---------------------------------------------------------------------------
# Connection pool (lazy init)
# ---------------------------------------------------------------------------

_pool = None


async def get_pool():
    """Get or create the asyncpg connection pool."""
    global _pool
    if _pool is None:
        import asyncpg
        _pool = await asyncpg.create_pool(
            MEMORY_PG_URL,
            min_size=1,
            max_size=5,
            command_timeout=10,
        )
    return _pool


async def close_pool():
    """Close the connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def _embedding_to_pgvector(vec: list[float]) -> str:
    """Convert embedding list to pgvector literal string."""
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


async def embed_text(text: str) -> list[float]:
    """Embed text using nomic-embed-text via vLLM OpenAI-compatible endpoint."""
    import httpx

    url = f"{MEMORY_EMBED_URL}/v1/embeddings"
    truncated = text[:2000]  # nomic-embed-text context limit

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            url,
            json={"model": MEMORY_EMBED_MODEL, "input": [truncated]},
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


async def store_fact(
    content: str,
    source: str = "manual",
    category: str = "general",
    importance: int = 5,
    session_id: str | None = None,
    channel: str = "unknown",
    expires_at: datetime | None = None,
) -> int | None:
    """Embed and store a new memory fact. Returns fact ID or None on failure."""
    if not MEMORY_RAG_ENABLED:
        return None

    try:
        pool = await get_pool()

        # Deduplicate: skip if identical content exists (not archived)
        content_hash = hashlib.md5(content.encode()).hexdigest()
        existing = await pool.fetchval(
            "SELECT id FROM facts WHERE md5(content) = $1 AND archived = false LIMIT 1",
            content_hash,
        )
        if existing:
            log.debug("[memory_rag] Duplicate content, skipping (id=%d)", existing)
            return existing

        # Embed
        embedding = await embed_text(content)
        vec_str = _embedding_to_pgvector(embedding)

        # Insert
        fact_id = await pool.fetchval(
            """
            INSERT INTO facts (content, source, category, importance, embedding,
                               session_id, channel, expires_at)
            VALUES ($1, $2, $3, $4, $5::vector, $6, $7, $8)
            RETURNING id
            """,
            content, source, category, importance, vec_str,
            session_id, channel, expires_at,
        )
        log.info("[memory_rag] Stored fact #%d (%s/%s)", fact_id, source, category)
        return fact_id

    except Exception as exc:
        log.warning("[memory_rag] store_fact failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Search (hybrid BM25 + vector with RRF fusion)
# ---------------------------------------------------------------------------


async def search_facts(
    query: str,
    limit: int = 5,
    category: str | None = None,
    min_importance: int = 1,
) -> list[dict]:
    """Hybrid search: BM25 + vector similarity fused via RRF.

    Returns list of dicts with keys: id, content, source, category,
    importance, rrf_score, sources (list of 'bm25'/'semantic').
    """
    if not MEMORY_RAG_ENABLED:
        return []

    try:
        pool = await get_pool()
        overfetch = limit * 3

        # Build WHERE clause for active facts
        where = "archived = false AND (expires_at IS NULL OR expires_at > now())"
        params = []
        param_idx = 1

        if category:
            where += f" AND category = ${param_idx}"
            params.append(category)
            param_idx += 1

        if min_importance > 1:
            where += f" AND importance >= ${param_idx}"
            params.append(min_importance)
            param_idx += 1

        # BM25 search via tsvector
        bm25_query = f"""
            SELECT id, content, source, category, importance,
                   ts_rank_cd(content_tsv, plainto_tsquery('english', ${param_idx})) AS score
            FROM facts
            WHERE {where} AND content_tsv @@ plainto_tsquery('english', ${param_idx})
            ORDER BY score DESC
            LIMIT {overfetch}
        """
        bm25_params = params + [query]
        bm25_rows = await pool.fetch(bm25_query, *bm25_params)

        # Vector search
        embedding = await embed_text(query)
        vec_str = _embedding_to_pgvector(embedding)

        sem_query = f"""
            SELECT id, content, source, category, importance,
                   1 - (embedding <=> ${param_idx}::vector) AS score
            FROM facts
            WHERE {where} AND embedding IS NOT NULL
            ORDER BY embedding <=> ${param_idx}::vector
            LIMIT {overfetch}
        """
        sem_params = params + [vec_str]
        sem_rows = await pool.fetch(sem_query, *sem_params)

        # RRF fusion
        bm25_results = [dict(r) for r in bm25_rows]
        sem_results = [dict(r) for r in sem_rows]

        return _rrf_fuse(bm25_results, sem_results, limit)

    except Exception as exc:
        log.warning("[memory_rag] search_facts failed: %s", exc)
        return []


def _rrf_fuse(
    bm25_results: list[dict],
    semantic_results: list[dict],
    limit: int,
) -> list[dict]:
    """Fuse BM25 and semantic results via Reciprocal Rank Fusion.

    Matches Alexandria's search/api.py _rrf_fuse() pattern.
    """
    scores: dict[int, float] = {}
    items: dict[int, dict] = {}

    for rank, item in enumerate(bm25_results):
        fid = item["id"]
        scores[fid] = scores.get(fid, 0) + 1.0 / (RRF_K + rank)
        if fid not in items:
            item["sources"] = ["bm25"]
            items[fid] = item
        else:
            items[fid]["sources"].append("bm25")

    for rank, item in enumerate(semantic_results):
        fid = item["id"]
        scores[fid] = scores.get(fid, 0) + 1.0 / (RRF_K + rank)
        if fid not in items:
            item["sources"] = ["semantic"]
            items[fid] = item
        else:
            items[fid]["sources"].append("semantic")

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    results = []
    for fid, rrf_score in ranked[:limit]:
        item = items[fid]
        results.append({
            "id": item["id"],
            "content": item["content"],
            "source": item["source"],
            "category": item["category"],
            "importance": item["importance"],
            "rrf_score": round(rrf_score, 6),
            "sources": item["sources"],
        })

    return results


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------


async def touch_facts(fact_ids: list[int]) -> None:
    """Increment access_count and update last_accessed for retrieved facts."""
    if not MEMORY_RAG_ENABLED or not fact_ids:
        return
    try:
        pool = await get_pool()
        await pool.execute(
            """
            UPDATE facts
            SET access_count = access_count + 1, last_accessed = now()
            WHERE id = ANY($1::bigint[])
            """,
            fact_ids,
        )
    except Exception as exc:
        log.warning("[memory_rag] touch_facts failed: %s", exc)


async def archive_fact(content: str) -> bool:
    """Soft-archive a fact by content match. Returns True if archived."""
    if not MEMORY_RAG_ENABLED:
        return False
    try:
        pool = await get_pool()
        result = await pool.execute(
            "UPDATE facts SET archived = true, updated_at = now() WHERE content = $1 AND archived = false",
            content,
        )
        return "UPDATE" in result
    except Exception as exc:
        log.warning("[memory_rag] archive_fact failed: %s", exc)
        return False


async def is_available() -> bool:
    """Check if pgvector memory is reachable (2s timeout)."""
    if not MEMORY_RAG_ENABLED:
        return False
    try:
        pool = await get_pool()
        await asyncio.wait_for(pool.fetchval("SELECT 1"), timeout=2.0)
        return True
    except Exception:
        return False


async def fact_count() -> int:
    """Return count of active (non-archived) facts."""
    if not MEMORY_RAG_ENABLED:
        return 0
    try:
        pool = await get_pool()
        return await pool.fetchval(
            "SELECT count(*) FROM facts WHERE archived = false"
        )
    except Exception:
        return 0
