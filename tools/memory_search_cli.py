#!/usr/bin/env python3
"""CLI wrapper for pgvector memory search.

Used by Claude Code /memory-search skill. Connects to pgvector on fox-n1
via Tailscale (K3s NodePort or cluster DNS depending on context).

Usage:
    python3 memory_search_cli.py "query text" [--limit 5] [--category general]
"""

import argparse
import asyncio
import os
import sys

# Default connection for tensor-core (via Tailscale to fox-n1 K3s NodePort)
DEFAULT_PG_URL = os.environ.get("MEMORY_PG_URL", "")
DEFAULT_EMBED_URL = os.environ.get(
    "MEMORY_EMBED_URL",
    "http://localhost:11434",  # Ollama on tensor-core
)
DEFAULT_EMBED_MODEL = os.environ.get("MEMORY_EMBED_MODEL", "nomic-embed-text")


async def search(query: str, limit: int = 5, category: str | None = None) -> list[dict]:
    """Perform hybrid search against pgvector."""
    import asyncpg
    import httpx

    # Embed the query
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            f"{DEFAULT_EMBED_URL}/v1/embeddings",
            json={"model": DEFAULT_EMBED_MODEL, "input": query},
        )
        r.raise_for_status()
        embedding = r.json()["data"][0]["embedding"]

    pool = await asyncpg.create_pool(DEFAULT_PG_URL, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            emb_str = "[" + ",".join(str(x) for x in embedding) + "]"

            # BM25 results
            bm25_sql = """
                SELECT id, content, source, category,
                       ts_rank_cd(content_tsv, plainto_tsquery('english', $1)) AS rank
                FROM facts
                WHERE archived = false
                  AND content_tsv @@ plainto_tsquery('english', $1)
            """
            params_bm25 = [query]
            if category:
                bm25_sql += " AND category = $2"
                params_bm25.append(category)
            bm25_sql += " ORDER BY rank DESC LIMIT $" + str(len(params_bm25) + 1)
            params_bm25.append(limit * 2)
            bm25_rows = await conn.fetch(bm25_sql, *params_bm25)

            # Semantic results
            sem_sql = f"""
                SELECT id, content, source, category,
                       1 - (embedding <=> '{emb_str}'::vector) AS similarity
                FROM facts
                WHERE archived = false
            """
            params_sem = []
            if category:
                sem_sql += " AND category = $1"
                params_sem.append(category)
            sem_sql += f" ORDER BY embedding <=> '{emb_str}'::vector LIMIT " + str(limit * 2)
            sem_rows = await conn.fetch(sem_sql, *params_sem)

        # RRF fusion (k=60)
        k = 60
        scores: dict[int, dict] = {}

        for rank, row in enumerate(bm25_rows):
            fid = row["id"]
            scores.setdefault(fid, {"content": row["content"], "source": row["source"],
                                     "category": row["category"], "rrf": 0.0, "methods": []})
            scores[fid]["rrf"] += 1.0 / (k + rank + 1)
            scores[fid]["methods"].append("bm25")

        for rank, row in enumerate(sem_rows):
            fid = row["id"]
            scores.setdefault(fid, {"content": row["content"], "source": row["source"],
                                     "category": row["category"], "rrf": 0.0, "methods": []})
            scores[fid]["rrf"] += 1.0 / (k + rank + 1)
            scores[fid]["methods"].append("vector")

        ranked = sorted(scores.values(), key=lambda x: x["rrf"], reverse=True)[:limit]
        return ranked
    finally:
        await pool.close()


def main():
    if not DEFAULT_PG_URL:
        print("Error: MEMORY_PG_URL environment variable not configured")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Semantic memory search via pgvector")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--limit", type=int, default=5, help="Max results (default 5)")
    parser.add_argument("--category", help="Filter by category")
    args = parser.parse_args()

    results = asyncio.run(search(args.query, limit=args.limit, category=args.category))

    if not results:
        print(f"No results for: {args.query}")
        sys.exit(0)

    print(f"Found {len(results)} results for: {args.query}\n")
    for i, r in enumerate(results, 1):
        methods = "+".join(r["methods"])
        print(f"{i}. [{r['category']}] ({r['source']}, {methods}, rrf={r['rrf']:.4f})")
        # Truncate long content for display
        content = r["content"]
        if len(content) > 200:
            content = content[:200] + "..."
        print(f"   {content}\n")


if __name__ == "__main__":
    main()
