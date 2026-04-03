-- NEXUS Agent Memory RAG Schema
-- Creates a separate database for agent memory embeddings.
-- Uses pgvector for dense vectors and native tsvector for BM25-style search.
-- Hybrid retrieval via Reciprocal Rank Fusion (RRF), matching Alexandria's pattern.
--
-- Prerequisites: PostgreSQL 16+ with pgvector extension available.
--
-- Usage:
--   1. Connect as superuser: psql -h <host> -U postgres
--   2. CREATE DATABASE nexus_memory;
--   3. \c nexus_memory
--   4. \i 001_memory_rag.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- Core memory facts table
CREATE TABLE IF NOT EXISTS facts (
    id              bigserial PRIMARY KEY,
    content         text NOT NULL,
    source          text NOT NULL DEFAULT 'manual',     -- manual, session_extract, observer, lesson, context
    category        text NOT NULL DEFAULT 'general',    -- infrastructure, preference, decision, lesson, project, general
    importance      smallint NOT NULL DEFAULT 5 CHECK (importance BETWEEN 1 AND 10),
    embedding       vector(768),

    -- Full-text search (generated column)
    content_tsv     tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(content, ''))
    ) STORED,

    -- Provenance
    session_id      text,
    channel         text DEFAULT 'unknown',             -- telegram, claude_code, email, observer, terminal

    -- Lifecycle
    access_count    integer NOT NULL DEFAULT 0,
    last_accessed   timestamptz,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    expires_at      timestamptz,                        -- NULL = permanent
    archived        boolean NOT NULL DEFAULT false
);

-- HNSW index for vector similarity search (cosine distance)
CREATE INDEX IF NOT EXISTS idx_facts_embedding
    ON facts USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_facts_tsv
    ON facts USING gin (content_tsv);

-- Filtered index for active (non-archived, non-expired) facts
CREATE INDEX IF NOT EXISTS idx_facts_active
    ON facts (created_at DESC)
    WHERE archived = false AND (expires_at IS NULL OR expires_at > now());

-- Category + importance for filtered retrieval
CREATE INDEX IF NOT EXISTS idx_facts_category
    ON facts (category, importance DESC);

-- Session provenance lookup
CREATE INDEX IF NOT EXISTS idx_facts_session
    ON facts (session_id)
    WHERE session_id IS NOT NULL;

-- Content deduplication helper
CREATE INDEX IF NOT EXISTS idx_facts_content_hash
    ON facts (md5(content))
    WHERE archived = false;
