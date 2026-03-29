# PureClaw

Multi-channel agentic AI platform with 8 LLM backends, 19 tools, and declarative security policies. Local GPU inference on NVIDIA Blackwell, automatic failover to cloud.

- **8 swappable backends** -- NVIDIA Nemotron Super (vLLM), Ollama, Anthropic, AWS Bedrock, Claude Code, Codex, Gemini, and Hybrid routing
- **19 built-in tools** -- shell, file I/O, web search/fetch, subagent parallelism, phone calls, task tracking, persistent memory
- **4 channels** -- Telegram, Discord, WhatsApp (multi-instance bridge), and email (IMAP with auto-classification)
- **17 background observers** -- email digest, threat intel, content publishing, git security audits, daily reports, real-time rail data, and more
- **Declarative YAML security policies** -- filesystem ACLs, SSRF protection, credential redaction, model allowlists, hot-reload without restart

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
| **Nemotron Super** (default) | Local GPU via vLLM | 19 built-in | Free |
| **Ollama** | Local (CPU or GPU) | 19 built-in | Free |
| **Anthropic API** | Cloud | 19 built-in | Per token |
| **AWS Bedrock** | Cloud | 19 built-in | Per token |
| **Claude Code** | CLI agent | Full agentic | Subscription |
| **Codex** | CLI agent | Full agentic | Subscription |
| **Gemini** | CLI agent | Full agentic | Subscription |
| **Hybrid** | API + CLI routing | Both | Mixed |

Switch backends live with `/backend`, `/nemotron`, `/sonnet`, `/opus`, or `/ollama`. If the primary fails, automatic failover kicks in (configurable, defaults to Bedrock Sonnet).

## Channels

**Telegram** -- streaming responses, inline keyboards, voice notes (Whisper transcription + TTS), photo/document handling, data cards.

**Discord** -- streaming responses, slash commands.

**WhatsApp** -- multi-instance baileys bridge, group chat support with mention-only mode.

**Email** -- IMAP polling with 4-lane classification (ignore / notify / auto-reply / follow-up). Draft queue with approve/reject from Telegram.

**Terminal** -- WebSocket channel for the [PureClaw CLI](https://github.com/puretensor/PureClaw-Cli).

## Data Cards

Instant structured responses that bypass the LLM. No tokens used.

| Command | What it does |
|---------|-------------|
| `/weather` | Weather + 3-day forecast |
| `/markets` | Stock indices, crypto, commodities, FX |
| `/trains` | UK rail departures (Darwin Push Port via Kafka) |
| `/nodes` | Infrastructure node status (Prometheus) |

## Observers

Background tasks on cron schedules. All optional -- they run if configured but won't break anything if dependencies are missing.

Examples: email digest (hourly), morning briefing (7:30 AM), threat intelligence (hourly), daily/weekly PDF reports, git security audits, real-time rail data consumer, memory sync, pipeline watchdog, heartbeat checklists.

## Architecture

```
nexus.py
  +-- Security        YAML policies, audit trail, path ACLs, SSRF protection, credential redaction
  +-- Backends (8)    vLLM, Ollama, Anthropic, Bedrock, Claude Code, Codex, Gemini, Hybrid
  +-- Channels (5)    Telegram, Discord, WhatsApp, Email, Terminal WebSocket
  +-- Tools (19)      Shell, files, web, subagents, phone, tasks, memory
  +-- Observers (17)  Cron-scheduled background intelligence
  +-- Dispatcher      Data cards (weather, markets, trains, nodes)
  +-- Scheduler       User-defined tasks and reminders
  +-- Memory          Persistent markdown files + daily journal
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

Security policies are mounted as ConfigMaps in K8s and hot-reload within 10 seconds -- no pod restart needed.

## Tech Stack

- **Runtime:** Python 3.12, asyncio
- **LLM:** vLLM, Ollama, Anthropic SDK, boto3 (Bedrock), google-genai, OpenAI SDK
- **Channels:** python-telegram-bot, discord.py, baileys (Node.js sidecar), websockets
- **Data:** SQLite (WAL mode), confluent-kafka, Pillow (card rendering)
- **Voice:** faster-whisper, edge-tts
- **Documents:** reportlab, python-docx, openpyxl, python-pptx, fpdf2
- **Infra:** Docker, K8s manifests, systemd unit

## License

MIT

```
