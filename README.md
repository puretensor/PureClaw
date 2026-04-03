# PureClaw

Multi-channel agentic AI platform with structured identity, hybrid vector memory, operator approval gates, and Prometheus observability. 8 LLM backends, 21 tools, 5 channels, and two-tier autonomous heartbeat. Local GPU inference on NVIDIA Blackwell with automatic failover to cloud.

- **8 swappable backends** -- NVIDIA Nemotron Super (vLLM), Ollama, Anthropic, AWS Bedrock, Claude Code, Codex, Gemini, and Hybrid routing
- **21 built-in tools** -- shell, file I/O, web search/fetch, subagent parallelism, phone calls, task tracking, persistent memory, semantic memory search (pgvector)
- **5 channels** -- Telegram, Discord, WhatsApp (multi-instance bridge), email (IMAP with auto-classification), and Terminal WebSocket
- **17+ background observers** -- two-tier heartbeat, email digest, threat intel, content publishing, git security audits, daily/weekly reports, nightly memory consolidation, and more
- **Structured identity** -- SOUL.md and USER.md define agent personality and operator profile, injected at highest priority across all backends
- **Hybrid RAG memory** -- pgvector + BM25 with Reciprocal Rank Fusion. Markdown is source of truth; vector search augments retrieval
- **Rule of Two** -- high-risk tool calls (kubectl, helm, email, mesh dispatch) require operator approval via Telegram inline buttons before execution
- **Prometheus metrics** -- `/metrics` endpoint with LLM cost tracking, tool execution counters, observer health, and heartbeat severity
- **Declarative YAML security policies** -- filesystem ACLs, SSRF protection, credential redaction, model allowlists, fail-closed in production, hot-reload without restart

## Quick Start

```bash
git clone https://github.com/puretensor/PureClaw.git
cd PureClaw
pip install -r requirements.txt
cp .env.example .env
```

If you use mesh, Alertmanager, or the WhatsApp bridge, also configure `MESH_SHARED_SECRET`, `ALERTMANAGER_WEBHOOK_SECRET`, and `WA_WEBHOOK_SECRET` as described in [`deploy/hardening_rollout.md`](deploy/hardening_rollout.md).

Set two required values in `.env`:

```bash
TELEGRAM_BOT_TOKEN=your-token-from-botfather
AUTHORIZED_USER_ID=your-telegram-user-id
```

Pick a backend and start:

```bash
# Default: NVIDIA Nemotron Super via vLLM (requires GPU)
ENGINE_BACKEND=vllm

# No GPU? Use Ollama (free, runs on CPU)
ENGINE_BACKEND=ollama

python3 nexus.py
```

## Engine Backends

| Engine | Type | Tools | Cost |
|--------|------|-------|------|
| **Nemotron Super** (default) | Local GPU via vLLM | 21 built-in | Free |
| **Ollama** | Local (CPU or GPU) | 21 built-in | Free |
| **Anthropic API** | Cloud | 21 built-in | Per token |
| **AWS Bedrock** | Cloud | 21 built-in | Per token |
| **Claude Code** | CLI agent | Full agentic | Subscription |
| **Codex** | CLI agent | Full agentic | Subscription |
| **Gemini** | CLI agent | Full agentic | Subscription |
| **Hybrid** | API + CLI routing | Both | Mixed |

Switch backends live with `/backend`, `/nemotron`, `/sonnet`, `/opus`, or `/ollama`. If the primary fails, automatic failover kicks in (configurable, defaults to Bedrock Sonnet).

## Channels

**Telegram** -- streaming responses, inline keyboards, voice notes (Whisper transcription + TTS), photo/document handling, data cards, Rule of Two approval buttons.

**Discord** -- streaming responses, slash commands.

**WhatsApp** -- multi-instance baileys bridge, group chat support with mention-only mode.

**Email** -- IMAP polling with 4-lane classification (ignore / notify / auto-reply / follow-up). Draft queue with approve/reject from Telegram. Idempotent processing with content-hash dedup.

**Terminal** -- authenticated WebSocket channel for the [PureClaw CLI](https://github.com/puretensor/PureClaw-Cli). Requires API key (fail-closed when unconfigured).

## Memory Architecture

PureClaw uses a layered memory system with structured identity at the top:

```
SOUL.md        -> Agent identity, values, behavioral anchors
USER.md        -> Operator profile, preferences, working style
CONTEXT.md     -> Operational context and active projects
LESSONS.md     -> Learned patterns and corrections
MEMORY.md      -> Indexed facts and knowledge
journals/      -> Daily timestamped logs
pgvector       -> Hybrid semantic search (768-dim nomic-embed-text + BM25 + RRF)
```

Memory is injected into every LLM call in priority order. The `search_memory_rag` tool provides semantic search across all stored facts. Markdown files are the source of truth; pgvector provides retrieval augmentation.

File operations use advisory locking (`fcntl.flock`) to prevent race conditions on concurrent read-modify-write cycles.

## Heartbeat Pipeline

Two-tier autonomous monitoring with adaptive scheduling:

1. **Gather** (free) -- async collectors pull from K3s, Ceph, services, Prometheus, Gmail, GitHub, Gitea
2. **Evaluate** (cheap) -- local Nemotron scores severity 0-3 with structured assessment
3. **Act** (conditional) -- Telegram notification at severity 2+, autonomous remediation at severity 3

Schedules: every 30 min during business hours (Mon-Fri 8-18 UTC), every 2 hours overnight. All runs logged to daily journal with key metrics.

## Security

### Declarative Policies

YAML-based security policies with hot-reload (10s poll). Controls filesystem access, network fetch domains, tool allowlists, model restrictions, and credential redaction. Fail-closed in production -- missing policy file raises RuntimeError.

### Rule of Two

High-risk tool calls require operator approval before execution:

- `kubectl apply/delete/drain/cordon/scale/patch/rollout undo`
- `systemctl stop/disable/mask`
- `helm install/upgrade/uninstall/delete`
- `docker rm/rmi/kill/stop/system prune`
- `rm -rf` on non-tmp paths, `DROP TABLE/DATABASE`
- All email sends and mesh dispatches
- File writes/edits to system paths (`/etc/`, `/usr/`, `/bin/`, `/sbin/`)

Pending actions are stored in SQLite and resolved via Telegram inline buttons. Timeout after 120 seconds.

### Subagent Isolation

Spawned subagents run with a `read_only` tool profile and cannot nest. All subagent tool executions are audit-linked to the parent session.

### Audit Trail

Every tool execution, LLM call, observer run, and policy violation is recorded to SQLite with SHA256 content hashes, duration, and policy decision. Audit failures are logged at WARNING level with Prometheus counters.

## Observability

### Prometheus Metrics

Exposed on `:9876/metrics` (Prometheus text format) alongside `/healthz` (liveness probe).

| Metric | Type | Labels |
|--------|------|--------|
| `nexus_llm_calls_total` | counter | backend, model |
| `nexus_llm_tokens_total` | counter | backend, model, direction |
| `nexus_llm_cost_usd_total` | counter | backend, model |
| `nexus_llm_latency_seconds` | histogram | backend |
| `nexus_tool_executions_total` | counter | tool_name, decision |
| `nexus_observer_runs_total` | counter | observer_name, status |
| `nexus_memory_ops_total` | counter | operation |
| `nexus_heartbeat_severity` | gauge | -- |
| `nexus_uptime_seconds` | gauge | -- |
| `nexus_audit_failures_total` | counter | function |

### Data Cards

Instant structured responses that bypass the LLM. No tokens used.

| Command | What it does |
|---------|-------------|
| `/weather` | Weather + 3-day forecast |
| `/markets` | Stock indices, crypto, commodities, FX |
| `/trains` | UK rail departures (Darwin Push Port via Kafka) |
| `/nodes` | Infrastructure node status (Prometheus) |

## Architecture

```
nexus.py
  +-- Identity        SOUL.md, USER.md -- agent personality and operator profile
  +-- Security        YAML policies, Rule of Two, audit trail, path ACLs, SSRF, redaction
  +-- Backends (8)    vLLM, Ollama, Anthropic, Bedrock, Claude Code, Codex, Gemini, Hybrid
  +-- Channels (5)    Telegram, Discord, WhatsApp, Email, Terminal WebSocket
  +-- Tools (21)      Shell, files, web, subagents, phone, tasks, memory, semantic search
  +-- Observers (17+) Cron-scheduled background intelligence + two-tier heartbeat
  +-- Dispatcher      Data cards (weather, markets, trains, nodes)
  +-- Scheduler       User-defined tasks and reminders
  +-- Memory          Markdown files + daily journal + pgvector hybrid RAG
  +-- Metrics         Prometheus counters/gauges/histograms on :9876
  +-- Compression     Two-tier context compression (tool truncation + LLM summarization)
  +-- Failover        Automatic backend failover with health probes
  +-- Mesh            Distributed agent mesh (multi-node coordination)
```

## Deployment

```bash
# systemd
sudo cp nexus.service /etc/systemd/system/
sudo systemctl enable --now nexus.service

# Docker
docker build -t pureclaw .
docker run -d --env-file .env --name pureclaw pureclaw

# Kubernetes
kubectl apply -f k8s/
```

Security policies are mounted as ConfigMaps in K8s and hot-reload within 10 seconds -- no pod restart needed. Sensitive credentials (database URLs, API keys) use K8s Secrets.

## Tech Stack

- **Runtime:** Python 3.12, asyncio, aiohttp
- **LLM:** vLLM, Ollama, Anthropic SDK, boto3 (Bedrock), google-genai, OpenAI SDK
- **Channels:** python-telegram-bot, discord.py, baileys (Node.js sidecar), websockets
- **Data:** SQLite (WAL mode), PostgreSQL + pgvector (hybrid RAG), confluent-kafka
- **Memory:** nomic-embed-text (768-dim), BM25 + vector with RRF fusion, asyncpg
- **Voice:** faster-whisper, edge-tts
- **Documents:** reportlab, python-docx, openpyxl, python-pptx, fpdf2
- **Observability:** Prometheus text format exporter, Grafana dashboards
- **Infra:** Docker, K8s manifests, systemd unit

## License

MIT
