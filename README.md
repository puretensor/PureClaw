# PureClaw

**Enterprise-grade agentic AI framework with declarative security policies, audit trails, and local GPU inference on NVIDIA Blackwell.**

PureClaw is a security-hardened agentic AI system built around [NVIDIA Nemotron Super](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1), served locally via [vLLM](https://github.com/vllm-project/vllm) on NVIDIA RTX PRO 6000 Blackwell GPUs. 8 swappable LLM backends, 19 tools, declarative YAML security policies, full audit logging, credential redaction, SSRF protection, and policy hot-reload without restart. One agent, zero cloud dependency.

PureClaw connects through Telegram, Discord, WhatsApp, and email. It runs 17 autonomous background observers for continuous operational intelligence, with automatic failover to AWS Bedrock when the primary backend fails. Cloud backends (Anthropic, OpenAI, Google) are available as fallbacks when local GPU inference isn't needed.

[pureclaw.ai](https://pureclaw.ai) | [GitHub](https://github.com/puretensor/PureClaw)

---

## Why NVIDIA Nemotron Super

NVIDIA Nemotron Super is a 120B-parameter Mixture-of-Experts model (12B active) purpose-built for agentic workloads — tool calling, multi-step reasoning, and structured output. Running it locally on Blackwell GPUs gives you:

- **Full agentic tool use** — native function calling with 19 built-in tools (shell, file ops, web search, subagents, memory)
- **Zero latency to inference** — no network round-trips to cloud APIs, model runs on local VRAM
- **No token costs** — unlimited inference on your own hardware
- **No data leaves your network** — prompts, tool results, conversation history all stay local
- **262K context window** — handles long conversations, large codebases, multi-document analysis
- **Continuous batching via vLLM** — handles 32+ concurrent requests efficiently on dual RTX PRO 6000 (192GB VRAM)

PureClaw is designed as an agentic scaffolding layer around Nemotron Super — the model handles reasoning and tool selection, PureClaw handles tool execution, conversation state, streaming, and multi-channel delivery.

---

## What It Does

You message your bot. PureClaw routes your message to NVIDIA Nemotron Super (or whichever engine is active), streams the response back in real time, and gives the model access to 19 built-in tools — shell commands, file operations, web search, phone calls, specialist agents, subagent parallelism, task tracking, and persistent memory.

Beyond conversation, PureClaw runs 17 background observers (email monitoring, threat intelligence, content publishing, daily reports, git security audits, heartbeat checklists), handles email with auto-reply for VIP senders and approve/reject drafts for everyone else, schedules reminders, compresses long conversations automatically, profiles users through an onboarding flow, keeps a daily interaction journal, and serves instant data cards for weather, markets, and trains — all from Telegram or Discord.

---

## Engine Backends

Eight engines. NVIDIA Nemotron Super is the default. Swap anytime with `/backend` or `/nemotron`. If the primary engine fails, PureClaw automatically fails over to AWS Bedrock Sonnet — no manual intervention needed.

| Engine | Type | What It Is | Tools | Cost |
|--------|------|-----------|-------|------|
| **NVIDIA Nemotron Super** (default) | Local GPU | [Nemotron Super 120B](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1) via vLLM on Blackwell GPUs | 19 built-in | Free |
| **Ollama** | Local | Any open model via [Ollama](https://ollama.com) | 19 built-in | Free |
| **Anthropic API** | Cloud API | Direct [Anthropic Messages API](https://docs.anthropic.com/en/docs/build-with-claude/overview) with prompt caching | 19 built-in | Pay per token |
| **AWS Bedrock** | Cloud API | Claude via [AWS Bedrock](https://aws.amazon.com/bedrock/) Converse API | 19 built-in | Pay per token |
| **Claude Code** | CLI Agent | Anthropic's [Claude Code](https://claude.ai/claude-code) CLI | Full agentic | Subscription |
| **Codex** | CLI Agent | OpenAI's [Codex](https://openai.com/index/codex/) CLI | Full agentic | Subscription |
| **Gemini** | CLI Agent | Google's [Gemini CLI](https://github.com/google-gemini/gemini-cli) | Full agentic | Subscription |
| **Hybrid** | Advanced | Routes between Gemini API (fast) and Claude Code CLI (power) | Both | Mixed |

**Which should I pick?**

- **Best experience (recommended):** **NVIDIA Nemotron Super** via vLLM. Full agentic tool calling, streaming, 262K context, zero cost per token. Requires NVIDIA GPUs with sufficient VRAM (48GB+ recommended, 96GB+ ideal).
- **No GPU?** Start with **Ollama**. It's free, runs CPU inference, and works out of the box.
- **Want cloud fallback?** **Anthropic API** or **AWS Bedrock** give you Claude with full tool use and streaming.
- **Maximum capability?** **Hybrid** routes simple queries to the fast API backend and complex tasks to Claude Code CLI.

The API backends (Nemotron/vLLM, Ollama, Anthropic, Bedrock) use PureClaw's 19 built-in tools. The CLI backends (Claude Code, Codex, Gemini) bring their own sandboxes and tool execution. Hybrid uses both.

---

## Quick Start

### 1. Create your Telegram bot

Open Telegram, search for [@BotFather](https://t.me/BotFather), and send `/newbot`. Follow the prompts. You'll get a token like `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`. Save it.

### 2. Find your Telegram user ID

Send a message to [@userinfobot](https://t.me/userinfobot) in Telegram. It will reply with your numeric user ID (e.g. `123456789`). This locks the bot to you — nobody else can use it.

### 3. Clone and install

```bash
git clone https://github.com/puretensor/PureClaw.git
cd PureClaw
pip install -r requirements.txt
```

### 4. Configure

```bash
cp .env.example .env
```

Open `.env` in a text editor and set these two required values:

```bash
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
AUTHORIZED_USER_ID=123456789
```

### 5. Set up your engine

Pick one (you can always add more later):

<details>
<summary><strong>NVIDIA Nemotron Super (default, recommended)</strong></summary>

**Requirements:** NVIDIA GPU with 48GB+ VRAM. Dual RTX PRO 6000 Blackwell (192GB total) is the reference configuration. Any NVIDIA GPU supported by vLLM will work — RTX 4090, A100, H100, etc.

Install vLLM:
```bash
pip install vllm
```

Download and serve NVIDIA Nemotron Super:
```bash
vllm serve nvidia/nemotron-3-super \
  --port 5000 \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Set it in `.env`:
```bash
ENGINE_BACKEND=vllm
VLLM_URL=http://127.0.0.1:5000/v1
VLLM_MODEL=nvidia/nemotron-3-super
VLLM_MAX_TOKENS=32768
```

**That's it.** NVIDIA Nemotron Super is the default engine. PureClaw's vLLM backend handles tool calling, conversation history, streaming, and context compression automatically.

For quantized variants (FP8, NVFP4) that fit in less VRAM, see the [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/).

</details>

<details>
<summary><strong>Ollama (free, local, no GPU required)</strong></summary>

Install Ollama:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Pull a model (pick one):
```bash
ollama pull llama3.2        # 3B, runs on any machine
ollama pull qwen3:8b        # 8B, good balance
ollama pull qwen3:32b       # 32B, needs ~20GB RAM
ollama pull qwen3:235b      # 235B MoE, needs serious GPU
```

Set it in `.env`:
```bash
ENGINE_BACKEND=ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
```

</details>

<details>
<summary><strong>Anthropic API (cloud fallback, pay per token)</strong></summary>

Get an API key from [console.anthropic.com](https://console.anthropic.com/).

Set it in `.env`:
```bash
ENGINE_BACKEND=anthropic_api
ANTHROPIC_API_KEY=sk-ant-your-key-here
ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

Supports prompt caching for reduced cost on long conversations.

</details>

<details>
<summary><strong>AWS Bedrock (Claude via AWS)</strong></summary>

Configure AWS credentials (`~/.aws/credentials` or environment variables). Then set in `.env`:
```bash
ENGINE_BACKEND=bedrock_api
AWS_DEFAULT_REGION=us-east-1
BEDROCK_MODEL=us.anthropic.claude-sonnet-4-6
BEDROCK_MAX_TOKENS=64000
```

Uses the Bedrock Converse API with full tool support.

</details>

<details>
<summary><strong>Claude Code (Anthropic, subscription)</strong></summary>

Install the Claude Code CLI:
```bash
npm install -g @anthropic-ai/claude-code
# or use the standalone installer:
curl -fsSL https://claude.ai/install.sh | sh
```

Authenticate (opens a browser):
```bash
claude login
```

Set it in `.env`:
```bash
ENGINE_BACKEND=claude_code
```

Claude Code uses your Anthropic subscription (Max plan). No API key needed.

</details>

<details>
<summary><strong>Codex (OpenAI, subscription or API credit)</strong></summary>

Install the Codex CLI:
```bash
npm install -g @openai/codex
```

Set it in `.env`:
```bash
ENGINE_BACKEND=codex_cli
CODEX_MODEL=gpt-5.2-codex
OPENAI_API_KEY=sk-proj-your-key-here
```

</details>

<details>
<summary><strong>Gemini (Google, subscription)</strong></summary>

Install the Gemini CLI:
```bash
npm install -g @google/gemini-cli
```

Authenticate:
```bash
gemini auth
```

Set it in `.env`:
```bash
ENGINE_BACKEND=gemini_cli
```

</details>

<details>
<summary><strong>Hybrid (API + CLI, advanced)</strong></summary>

Routes between a fast API backend (Gemini) and a powerful CLI backend (Claude Code).

```bash
ENGINE_BACKEND=hybrid
HYBRID_DEFAULT=api
```

</details>

### 6. Start

```bash
python3 nexus.py
```

Open your bot in Telegram and send a message. You should see a streaming response from NVIDIA Nemotron Super.

### 7. Production deployment

<details>
<summary><strong>systemd</strong></summary>

```bash
sudo cp nexus.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nexus.service
```

</details>

<details>
<summary><strong>Docker</strong></summary>

```bash
docker build -t pureclaw .
docker run -d --env-file .env --name pureclaw pureclaw
```

</details>

<details>
<summary><strong>Kubernetes</strong></summary>

Manifests are in `k8s/`:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml    # create from .env first
kubectl apply -f k8s/pvcs.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/security-policy-configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Or use the deploy script:
```bash
bash k8s/deploy.sh
```

The security policy is mounted as a ConfigMap at `/app/security/policy.yaml`. Update with `kubectl apply` and the `PolicyWatcher` reloads it within 10 seconds -- no pod restart needed.

</details>

---

## Telegram Commands

### Conversation

| Command | What it does |
|---------|-------------|
| `/new [name]` | Archive current session, start fresh |
| `/session [name]` | List sessions, switch to one, or create new |
| `/session delete <name>` | Delete a named session |
| `/history` | List archived sessions |
| `/resume <n>` | Restore an archived session by number |
| `/status` | Show current engine, model, session info |

### Engine Switching

| Command | What it does |
|---------|-------------|
| `/nemotron` | Quick-switch to NVIDIA Nemotron Super (local) |
| `/backend` | Tap to select from all available engines |
| `/sonnet` | Quick-switch to Claude Sonnet |
| `/opus` | Quick-switch to Claude Opus |
| `/ollama` | Quick-switch to local Ollama model |
| `/bedrock` | Quick-switch to Claude Sonnet via AWS Bedrock |

### User Profile & Onboarding

| Command | What it does |
|---------|-------------|
| `/start` | First-run onboarding: sets your name, timezone, and shows capabilities |
| `/profile` | View your current profile (name, timezone, preferences) |
| `/profile set name <value>` | Set your display name |
| `/profile set timezone <value>` | Set your timezone (e.g. `Europe/London`, `America/New_York`) |
| `/profile set <key> <value>` | Store a custom preference |

New users are guided through an onboarding flow on first `/start` that captures their name and timezone. Returning users see the help menu.

### Memory

PureClaw remembers things across sessions using markdown memory files. Memories persist to disk.

| Command | What it does |
|---------|-------------|
| `/remember <fact>` | Store a persistent memory (e.g. `/remember prefer dark mode`) |
| `/remember --topic infrastructure <fact>` | Store to a topic file (infrastructure, lessons, projects, etc.) |
| `/forget <key or number>` | Remove a memory |
| `/memories` | List all memories and topic files |

The AI engine also has three memory tools (`save_memory`, `read_memory`, `list_memory`) that let it read and write memories autonomously during conversation.

### Daily Journal

PureClaw keeps a daily journal of interactions at `/data/memory/journal/YYYY-MM-DD.md`. Journal entries are automatically injected into LLM context for short-term continuity across session resets.

| Command | What it does |
|---------|-------------|
| `/journal` | Show today's journal entries |
| `/journal yesterday` | Show yesterday's entries |
| `/journal YYYY-MM-DD` | Show entries for a specific date |
| `/journal purge` | Manually clean up old journal files (auto-purged after 30 days) |

### Scheduling

| Command | What it does |
|---------|-------------|
| `/schedule 5pm generate a status report` | Run a full AI prompt at 5pm |
| `/schedule daily 8am check my emails` | Recurring daily prompt |
| `/remind tomorrow 9am call the dentist` | Simple notification (no AI, just a ping) |
| `/cancel <n>` | Cancel a scheduled task by number |

Time formats: `5pm`, `9:30am`, `17:00`, `tomorrow 9am`, `monday`, `9 feb`, `in 30 minutes`, `daily 8am`, `weekly monday 9am`.

### Data Cards

Instant structured responses that bypass the AI engine for speed. No tokens used.

| Command | What it does |
|---------|-------------|
| `/weather [location]` | Weather with 3-day forecast (defaults to your configured location) |
| `/markets` | Stock indices, crypto, commodities, and FX — all in one card |
| `/trains [from] [to]` | UK train departures |
| `/nodes` | Infrastructure node status (requires Prometheus) |

### Voice

| Command | What it does |
|---------|-------------|
| `/voice on` | Enable voice responses (AI replies include audio) |
| `/voice off` | Disable voice responses |
| Send a voice note | Transcribed via Whisper, then processed by your AI engine |

### Email & Follow-ups

| Command | What it does |
|---------|-------------|
| `/drafts` | View pending email drafts with Approve / Reject buttons |
| `/followups` | List emails waiting for a reply |
| `/followups resolve <n>` | Mark a follow-up as resolved |

### Infrastructure

| Command | What it does |
|---------|-------------|
| `/check nodes` | Quick health check |
| `/restart <service> [node]` | Restart a service (with confirmation) |
| `/logs <service> [node] [n]` | Tail service logs |
| `/disk [node]` | Disk usage |
| `/top [node]` | System overview |
| `/deploy <site>` | Trigger deploy webhook (with confirmation) |
| `/calendar [today\|week]` | Google Calendar events |

---

## Features

### Automatic Failover

If the primary LLM backend fails (network error, timeout, rate limit), PureClaw automatically retries on AWS Bedrock Sonnet. The failover backend is lazy-initialized on first failure and reused for subsequent calls. Users see a status notification when failover activates. Configurable via `FAILOVER_ENABLED` and `FAILOVER_BACKEND`.

### User Profiles & Onboarding

First-time users are guided through a conversational onboarding flow that captures their name and timezone. Profile data is stored in SQLite and injected into LLM context so the AI knows who it's talking to. Profiles persist across sessions and can be updated anytime with `/profile set`.

### Group Chat Behavior

When deployed in group chats (Telegram groups, WhatsApp groups), PureClaw defaults to silent. It only responds when directly addressed (@mentioned, replied to, or name invoked). One response per prompt, no consecutive messages, no unsolicited input.

### Streaming Responses

Responses stream in real time — you see the text appear word by word in Telegram and Discord. Tool usage (file reads, shell commands, web searches) shows live status updates. NVIDIA Nemotron Super's streaming is served directly from local vLLM — no cloud API latency.

### Tool Use

When using API backends (Nemotron/vLLM, Ollama, Anthropic, Bedrock), PureClaw provides 19 built-in tools:

| Tool | What it does |
|------|-------------|
| `bash` | Execute shell commands |
| `read_file` | Read file contents (with optional offset/limit) |
| `write_file` | Create or overwrite files |
| `edit_file` | Find-and-replace within files |
| `glob` | Find files by pattern |
| `grep` | Search file contents with regex |
| `web_search` | Search the web (SearXNG or DuckDuckGo) |
| `web_fetch` | Fetch and extract text from any URL |
| `enter_plan_mode` | Switch to read-only exploration (blocks writes) |
| `exit_plan_mode` | Return to full execution mode |
| `make_phone_call` | Outbound phone calls via voice AI |
| `einherjar_dispatch` | Dispatch tasks to specialist agent council (legal, financial, engineering) |
| `spawn_subagent` | Spawn parallel subagents for concurrent research |
| `create_task` | Create tracked tasks that persist across sessions |
| `update_task` | Update task status and append notes |
| `list_tasks` | List tracked tasks with status filters |
| `save_memory` | Write to persistent markdown memory |
| `read_memory` | Read from memory files or search across all |
| `list_memory` | List all memory files and their sizes |

NVIDIA Nemotron Super excels at tool calling — its MoE architecture was specifically trained for agentic workloads with multi-step tool use, making it the ideal model for PureClaw's tool loop.

The CLI engines (Claude Code, Codex, Gemini) bring their own tools — they handle file operations, code execution, and web search through their own sandboxed environments.

### Plan Mode & Subagents

**Plan mode** lets the AI enter a read-only exploration phase before making changes. In plan mode, write tools (`bash`, `write_file`, `edit_file`) are blocked — the AI can only read files, search, and reason. When it has a plan, it exits back to full execution mode.

**Subagents** run focused subtasks in parallel. Each subagent gets its own conversation context and configurable tool access. Useful for concurrent research, parallel analysis, or delegating focused tasks. Subagents cannot spawn further subagents.

### Task Management

The AI can create, track, and update persistent tasks across sessions. Tasks have titles, descriptions, priorities (low/medium/high/critical), and statuses (pending/in_progress/done/cancelled). Tasks survive session resets.

### Memory

PureClaw uses a file-based memory system:

- **MEMORY.md** — Main memory file, always loaded into context
- **Topic files** — Separate markdown files for detailed notes (e.g. `infrastructure.md`, `lessons.md`)
- **Three LLM tools** — `save_memory`, `read_memory`, `list_memory` let the AI manage its own memory autonomously

Memories persist to disk and survive restarts. The AI can save lessons, preferences, project context, and operational knowledge.

### Daily Journal

PureClaw maintains a daily interaction journal at `/data/memory/journal/`. Each message and voice note is logged with timestamps. The two most recent days are automatically injected into LLM context, giving the AI short-term continuity across session resets. Old journals are auto-purged after 30 days.

### Email Handling

PureClaw monitors your IMAP inboxes and classifies each incoming message into one of four lanes:

| Classification | What happens |
|----------------|-------------|
| **ignore** | Spam, newsletters, no-reply — silently skipped |
| **notify** | Telegram notification only, no reply drafted |
| **auto_reply** | AI drafts and sends a reply immediately (VIP senders only) |
| **followup** | Tracked for follow-up reminders |

**Auto-reply** fires only for whitelisted VIP domains and must pass 11 gates before anything sends. See `channels/email_in.py` for the full gate list.

**Draft queue** is available for messages that need human approval — you get Telegram notifications with [Approve] / [Reject] buttons, and nothing sends until you tap.

### Context Compression

Long conversations are compressed automatically to stay within model context limits:

- **Tier 1: Tool result truncation** — Old tool results are summarized to save tokens (zero cost, always active)
- **Tier 2: LLM summarization** — When token count exceeds a threshold, older messages are replaced with an LLM-generated summary while preserving recent messages

NVIDIA Nemotron Super's 262K context window means compression triggers far less frequently than with smaller-context models.

### Voice

Send voice notes in Telegram — they're transcribed via Whisper and processed by your AI engine. Enable `/voice on` to get audio responses back (via edge-tts or a custom TTS endpoint).

### Health Probes

Background health checking for dependent services (Whisper, TTS). Services are marked offline after consecutive failures, online after one success.

---

## Observers

Background tasks that run on cron schedules inside the PureClaw process. No external cron needed.

### Active Observers (registered at startup)

| Observer | Schedule | What it does |
|----------|----------|-------------|
| Email Digest | Every hour | Summarizes unread emails across configured accounts |
| Morning Brief | 7:30 AM weekdays | Combined email + weather + calendar briefing |
| Daily Snippet | 8:00 AM weekdays | Geopolitical news brief from RSS feeds |
| Content Review | Every 2 hours | Reviews content submissions and editorial pipeline |
| Follow-up Reminder | 9:00 AM weekdays | Nags about unanswered outbound emails |
| Cyber Threat Feed | Every hour | CVE monitoring, exploit tracking, threat intelligence |
| Intel Deep Analysis | Every 12 hours | Deep geopolitical/security analysis articles |
| Memory Sync | Every 10 min | Syncs memory files between nodes |
| Daily Report | 1:00 AM daily | Compiles previous day's activity into a branded PDF |
| Weekly Report | 2:00 AM Sundays | Weekly activity summary report |
| Doc Compiler | 6:00 AM daily | Compiles queued documents into branded PDFs |
| Git Security Audit | Every 2 hours | Scans repos for secrets, credentials, sensitive data |
| Git Auto Sync | Every 4 hours | Keeps git repos synchronized across remotes |
| Pipeline Watchdog | Every 6 hours | Monitors CI/CD and deployment pipelines |
| Heartbeat | 8am, 12pm, 4pm, 8pm | Reads `/data/memory/HEARTBEAT.md` checklist; skips silently if empty (zero LLM cost) |
| Git Push | Always on | Webhook listener for git push event summaries |
| Darwin Consumer | Always on | Real-time UK rail data processing (Kafka) |

Observers are optional — they run if configured but won't break anything if their dependencies aren't set up.

---

## Architecture

```
nexus.py (entry point)
  |
  +-- Security (security/)
  |     +-- policy.py          YAML policy loading, validation, hot-reload watcher
  |     +-- audit.py           Structured audit trail (SQLite audit_log table)
  |     +-- filesystem.py      Path-based read/write ACLs + bash heuristics
  |     +-- network.py         SSRF protection + domain allow/deny
  |     +-- redact.py          Credential redaction (regex + env vars)
  |     +-- inference.py       Model allowlist, token budget, prompt integrity
  |     +-- policy.yaml        Declarative security policy (YAML)
  |     +-- schema.json        JSON Schema for policy validation
  |
  +-- Engine (backends/)
  |     +-- vllm               NVIDIA Nemotron Super via vLLM (default)
  |     +-- ollama             Local models via Ollama API
  |     +-- anthropic_api      Direct Anthropic Messages API (prompt caching)
  |     +-- bedrock_api        Claude via AWS Bedrock Converse API
  |     +-- claude_code        Claude Code CLI agent
  |     +-- codex_cli          Codex CLI agent
  |     +-- gemini_cli         Gemini CLI agent
  |     +-- hybrid             Routes between API (fast) and CLI (power)
  |     +-- tools.py           19 tools + shared execution loop + policy enforcement
  |
  +-- Channels
  |     +-- Telegram           Streaming, keyboards, voice, photos, documents
  |     +-- Discord            Streaming, slash commands
  |     +-- WhatsApp           Multi-instance bridge (baileys sidecar)
  |     +-- Email Input        IMAP polling, classification, draft generation
  |
  +-- Observers (17)           Background tasks on cron schedules + persistent threads
  +-- Dispatcher               Instant data cards (weather, markets, trains, nodes)
  +-- Draft Queue              Email drafts with Telegram approve/reject
  +-- Scheduler                User-defined tasks and reminders
  +-- Memory                   Persistent markdown memory (MEMORY.md + topic files + daily journal)
  +-- Context Compression      Two-tier conversation compression
  +-- Health Probes            Background service health checking
  +-- Failover                 Automatic failover runner for resilience
```

---

## Configuration Reference

All configuration is in `.env`. Only two values are required.

### Required

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Your bot token from [@BotFather](https://t.me/BotFather) |
| `AUTHORIZED_USER_ID` | Your Telegram user ID (from [@userinfobot](https://t.me/userinfobot)) |

### Engine Selection

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGINE_BACKEND` | `vllm` | Which engine: `vllm`, `ollama`, `anthropic_api`, `bedrock_api`, `claude_code`, `codex_cli`, `gemini_cli`, `hybrid` |

### Automatic Failover

| Variable | Default | Description |
|----------|---------|-------------|
| `FAILOVER_ENABLED` | `true` | Enable automatic failover to backup backend on primary failure |
| `FAILOVER_BACKEND` | `bedrock_api` | Which backend to fail over to (`bedrock_api`, `anthropic_api`, `gemini_api`) |

### NVIDIA Nemotron Super / vLLM (default)

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_URL` | `http://127.0.0.1:5000/v1` | vLLM OpenAI-compatible endpoint |
| `VLLM_MODEL` | `nvidia/nemotron-3-super` | Model name (Nemotron Super by default) |
| `VLLM_TOOLS_ENABLED` | `true` | Enable/disable tool use |
| `VLLM_TOOL_MAX_ITER` | `10` | Max tool call iterations per response |
| `VLLM_TOOL_TIMEOUT` | `60` | Per-tool-call timeout (seconds) |
| `VLLM_TOTAL_TIMEOUT` | `300` | Total request timeout (seconds) |
| `VLLM_MAX_TOKENS` | `32768` | Max tokens per response |

### Ollama

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `qwen3:235b` | Model name (must be pulled first) |
| `OLLAMA_TOOLS_ENABLED` | `true` | Enable/disable tool use |
| `OLLAMA_TOOL_MAX_ITER` | `25` | Max tool call iterations per response |
| `OLLAMA_TOOL_TIMEOUT` | `30` | Per-tool-call timeout (seconds) |
| `OLLAMA_NUM_PREDICT` | `8192` | Max tokens per response |

### Anthropic API

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (none) | Your Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-latest` | Model name |
| `ANTHROPIC_MAX_TOKENS` | `4096` | Max tokens per response |
| `ANTHROPIC_TOOLS_ENABLED` | `true` | Enable/disable tool use |
| `ANTHROPIC_TOOL_MAX_ITER` | `25` | Max tool call iterations |
| `ANTHROPIC_TOOL_TIMEOUT` | `120` | Per-tool-call timeout (seconds) |
| `ANTHROPIC_TOTAL_TIMEOUT` | `600` | Total request timeout (seconds) |

### AWS Bedrock

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region |
| `BEDROCK_MODEL` | `us.anthropic.claude-sonnet-4-6` | Bedrock model ID (inference profile format) |
| `BEDROCK_MAX_TOKENS` | `64000` | Max tokens per response |

AWS credentials are read from standard sources (`~/.aws/credentials`, env vars, IAM role).

### Claude Code CLI

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_BIN` | auto-detected | Path to `claude` binary |
| `CLAUDE_CWD` | `/home/user` | Working directory for Claude |
| `CLAUDE_TIMEOUT` | `1800` | Timeout in seconds |
| `CLAUDE_MODEL` | `sonnet` | Model hint (sonnet, opus, haiku) |

### Codex CLI

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEX_BIN` | auto-detected | Path to `codex` binary |
| `CODEX_MODEL` | (Codex default) | Model name (e.g. `gpt-5.2-codex`, `o3`) |
| `CODEX_CWD` | `/home/user` | Working directory |
| `OPENAI_API_KEY` | (none) | Your OpenAI API key |

### Gemini CLI

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_BIN` | auto-detected | Path to `gemini` binary |
| `GEMINI_CLI_MODEL` | (Gemini default) | Model name (e.g. `gemini-3-flash-preview`) |

### Hybrid

| Variable | Default | Description |
|----------|---------|-------------|
| `HYBRID_DEFAULT` | `api` | Default routing: `api` or `cli` |
| `HYBRID_CLI_TIMEOUT` | `1800` | CLI backend timeout (seconds) |
| `HYBRID_API_TIMEOUT` | `300` | API backend timeout (seconds) |

### Discord

| Variable | Default | Description |
|----------|---------|-------------|
| `DISCORD_BOT_TOKEN` | (none) | Discord bot token |
| `DISCORD_AUTHORIZED_USER_ID` | (none) | Your Discord user ID |

### Subagent

| Variable | Default | Description |
|----------|---------|-------------|
| `SUBAGENT_MODEL` | `sonnet` | Model for spawned subagents |
| `SUBAGENT_MAX_ITER` | `15` | Max tool iterations per subagent |
| `SUBAGENT_TIMEOUT` | `180` | Subagent timeout (seconds) |

### Context Compression

| Variable | Default | Description |
|----------|---------|-------------|
| `COMPRESS_TRIGGER_TOKENS` | `100000` | Token count that triggers LLM summarization |
| `PRESERVE_RECENT_MESSAGES` | `40` | Recent messages to keep verbatim |
| `SUMMARY_MODEL` | `us.anthropic.claude-haiku-4-5-20251001` | Model used for summarization |

### Security Policy

| Variable | Default | Description |
|----------|---------|-------------|
| `SECURITY_POLICY_PATH` | `security/policy.yaml` | Path to YAML security policy file |

See [Security Framework](#security-framework) for full policy schema documentation.

### Other

| Variable | Description |
|----------|-------------|
| `AGENT_NAME` | Your agent's name (shown in prompts) |
| `AGENT_PERSONALITY` | Personality injected into system prompt |
| `SEARXNG_URL` | Self-hosted SearXNG URL for private web search |
| `WEATHER_DEFAULT_LOCATION` | Default location for `/weather` |
| `PROMETHEUS_URL` | Prometheus server for `/nodes` and health monitoring |
| `WHISPER_URL` | Whisper API endpoint for voice transcription |
| `ALERT_BOT_TOKEN` | Separate bot token for alert notifications (defaults to main bot) |

---

## NVIDIA GPU Configuration

### Reference Setup (PureTensor Infrastructure)

| Component | Specification |
|-----------|--------------|
| **GPU** | 2x NVIDIA RTX PRO 6000 Blackwell (96GB VRAM each, 192GB total) |
| **CPU** | AMD Threadripper PRO 9975WX 32C/64T |
| **RAM** | 512GB DDR5-4800 ECC |
| **Storage** | 2x Samsung 9100 PRO 4TB Gen5 NVMe RAID0 (29 GB/s read) |
| **Model** | NVIDIA Nemotron Super 120B-A12B (NVFP4 quantization) |
| **Serving** | vLLM with tensor parallelism across both GPUs |

### Minimum Requirements

| Configuration | GPU | VRAM | Model Variant |
|--------------|-----|------|---------------|
| Entry | RTX 4090 (24GB) | 24GB | Nemotron Super NVFP4 (fits in ~20GB) |
| Recommended | RTX 5090 / A100 (48-80GB) | 48-80GB | Nemotron Super FP8 |
| Production | 2x RTX PRO 6000 Blackwell | 192GB | Nemotron Super FP16/BF16 (full precision) |

### vLLM Service Configuration

```bash
# systemd service example (vllm-nemotron.service)
vllm serve nvidia/nemotron-3-super \
  --port 5000 \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --gpu-memory-utilization 0.95
```

---

## Project Structure

```
PureClaw/
+-- nexus.py                    Entry point -- starts all subsystems
+-- config.py                   Environment loading, system prompt, logging
+-- db.py                       SQLite: sessions, drafts, follow-ups, tasks, audit_log, user_profiles
+-- engine.py                   Engine abstraction (sync + streaming + audit + automatic failover)
+-- memory.py                   Persistent markdown memory system
+-- scheduler.py                Task scheduler (/schedule, /remind)
+-- context_compression.py      Two-tier conversation compression
+-- health_probes.py            Background service health checking
|
+-- security/                   Enterprise security framework
|     +-- policy.py             YAML policy loading, validation, hot-reload watcher
|     +-- policy.yaml           Default security policy (ships permissive)
|     +-- schema.json           JSON Schema for policy validation
|     +-- audit.py              Structured audit trail (fire-and-forget)
|     +-- filesystem.py         Path-based read/write ACLs
|     +-- network.py            SSRF protection + domain allow/deny
|     +-- redact.py             Credential redaction (regex + env vars)
|     +-- inference.py          Model allowlist, token budget, prompt integrity
|
+-- backends/
|     +-- base.py               Backend Protocol definition
|     +-- vllm.py               NVIDIA Nemotron Super via vLLM (default)
|     +-- ollama.py             Ollama backend with tool loop
|     +-- anthropic_api.py      Anthropic Messages API backend
|     +-- bedrock_api.py        AWS Bedrock Converse API backend
|     +-- gemini_api.py         Google Gemini API backend
|     +-- claude_code.py        Claude Code CLI backend
|     +-- codex_cli.py          Codex CLI backend
|     +-- gemini_cli.py         Gemini CLI backend
|     +-- hybrid.py             API + CLI routing backend
|     +-- tools.py              19 tools + shared execution loop + policy enforcement
|     +-- __init__.py           Backend factory (lazy singleton)
|
+-- channels/
|     +-- telegram/             Bot setup, commands, callbacks, streaming
|     +-- discord/              Discord bot handlers and streaming
|     +-- whatsapp.py           WhatsApp bridge (baileys sidecar)
|     +-- email_in.py           IMAP polling email input
|
+-- observers/                  17 background observers (cron + persistent)
+-- dispatcher/                 Data cards (weather, markets, trains, nodes)
+-- drafts/                     Email draft queue with approve/reject
+-- handlers/                   Telegram handlers (voice, photo, document, location)
+-- tools/                      Integration tools (Gmail, Google Calendar)
+-- prompts/                    System prompts
+-- failover/                   Automatic failover runner
+-- k8s/                        Kubernetes manifests + security policy ConfigMap
+-- tests/                      Test suite (990+ tests)
+-- Dockerfile                  Container build
+-- nexus.service               systemd unit file
```

---

## Testing

```bash
python3 -m pytest tests/ -v
```

990+ tests covering backends, tools, security framework, observers, channels, handlers, and integration points. The security test suite (104 tests) covers policy loading/validation, filesystem ACLs, SSRF protection, credential redaction, and audit logging.

---

## Requirements

- Python 3.11+
- A Telegram bot token
- At least one engine: NVIDIA GPU with vLLM, Ollama installed, an API key, or a CLI tool authenticated

### Python Dependencies

```
python-telegram-bot>=21.0
discord.py>=2.3.0
python-dotenv>=1.0.0
aiohttp>=3.9.0
Pillow>=10.0.0
edge-tts>=6.1.0
confluent-kafka>=2.3.0
openai>=2.6.0
anthropic>=0.42.0
boto3>=1.35.0
fpdf2>=2.8.0
google-api-python-client>=2.100.0
google-auth-oauthlib>=1.2.0
google-genai>=1.0.0
reportlab>=4.4.0
pdfplumber>=0.11.0
pypdfium2>=4.0.0
python-docx>=1.0.0
openpyxl>=3.1.0
python-pptx>=1.0.0
pypdf>=4.0.0
img2pdf>=0.5.0
pandas>=2.0.0
```

### System Dependencies (optional)

- **NVIDIA GPU + CUDA** — Required for vLLM / Nemotron Super (recommended)
- **ffmpeg** — Required for voice output
- **ripgrep** — Used by the `grep` tool (falls back to Python regex if absent)
- **Node.js 18+** — Required for CLI engines (claude, codex, gemini) and document generation skills
- **Ollama** — Required for the `ollama` backend
- **pandoc** — Used for document format conversion

---

## Security Framework

PureClaw ships with a declarative security framework designed for enterprise compliance (ISO 27001, Cyber Essentials, G-Cloud). All enforcement flows through a single choke point (`execute_tool()`) and is configured via a YAML policy file that can be hot-reloaded without restarting the agent.

### Policy-Driven Access Control

Security policies are defined in `security/policy.yaml` (or mounted as a Kubernetes ConfigMap). The policy schema is validated against `security/schema.json` at load time.

```yaml
# Example: restrictive policy for a production deployment
version: 2
filesystem:
  read_allow: ["/data/**", "/app/**"]
  read_deny: ["**/.env", "**/*secret*", "/etc/shadow"]
  write_allow: ["/data/**", "/output/**"]
  write_deny: ["/etc/**", "/usr/**", "/proc/**"]
network:
  fetch_allow_domains: ["*.github.com", "api.openai.com"]
  block_private_ranges: true    # Blocks RFC 1918, loopback, link-local (SSRF protection)
tools:
  allowed: ["bash", "read_file", "grep", "web_search"]
  denied: ["web_fetch"]         # Deny takes precedence over allow
inference:
  model_allowlist: ["claude-*", "nvidia/*"]
  max_tokens_per_session: 500000
credentials:
  redact_patterns:
    - "sk-[a-zA-Z0-9_-]{20,}"   # OpenAI/Anthropic keys
    - "ghp_[a-zA-Z0-9]{36,}"    # GitHub tokens
    - "AKIA[A-Z0-9]{16}"        # AWS access keys
  redact_env_vars: ["ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN"]
audit:
  enabled: true
  retention_days: 90
```

### Security Modules

| Module | What it does |
|--------|-------------|
| `security/policy.py` | Frozen dataclass schema, YAML loading, JSON Schema validation, `PolicyWatcher` for hot-reload (10s poll, last-known-good on failure, monotonic version enforcement) |
| `security/audit.py` | Fire-and-forget audit logging to SQLite `audit_log` table. Records every tool execution, LLM call, observer run, and policy violation. Redaction applied before logging. |
| `security/filesystem.py` | Path-based read/write ACLs with symlink resolution. Heuristic bash command analysis (defense-in-depth). Deny rules take precedence. |
| `security/network.py` | SSRF protection: blocks RFC 1918, loopback, link-local, reserved, and multicast addresses. Domain allow/deny lists with glob matching. DNS resolution check. |
| `security/redact.py` | Credential redaction via regex patterns + environment variable values. Applied to tool outputs before LLM sees them, to conversation history before API calls, and to audit log entries. |
| `security/inference.py` | Model allowlist validation, cumulative token budget per session (24h TTL), system prompt immutability check. |

### How Enforcement Works

All tool calls flow through `execute_tool()` in `backends/tools.py`. This is the single enforcement point:

1. **Plan mode check** -- write tools blocked in read-only mode
2. **Tool allowlist/denylist** -- is this tool permitted by policy?
3. **Filesystem ACL** -- for file tools, is the path in the allow list and not in the deny list?
4. **Bash heuristic** -- for shell commands, check for dangerous patterns and denied write targets
5. **Network egress** -- for `web_fetch`, validate URL against domain list and SSRF rules
6. **Execute** -- run the tool
7. **Credential redaction** -- scrub secrets from the result before returning to the LLM
8. **Audit log** -- record the execution (tool name, args hash, result hash, duration, policy decision)

LLM calls are audited in `engine.py`. Observer runs are audited in `observers/registry.py`. All four API backends (Anthropic, Bedrock, vLLM, Ollama) redact conversation history before sending to the inference API.

### Policy Hot-Reload

The `PolicyWatcher` runs as an async task, polling the policy file every 10 seconds. Changes are detected via SHA256 hash comparison. On change:

- New policy is validated against JSON Schema
- Version must be >= current (monotonic, no downgrades)
- Failed validation keeps the previous policy (last known good)
- Atomic swap via Python attribute assignment (GIL-safe)

In Kubernetes, mount the policy as a ConfigMap. `kubectl apply` updates the file, and the watcher reloads within 10 seconds -- no pod restart needed.

### Baseline Protections

- **Single-user only** -- locked to one Telegram/Discord user ID per channel
- **No telemetry** -- no analytics, no tracking, no phoning home
- **No cloud dependency** -- NVIDIA Nemotron Super + vLLM keeps everything local; cloud backends use your own keys
- **Gated email replies** -- VIP auto-replies pass 11 safety gates; all other drafts require explicit Telegram approval
- **Plan mode** -- AI can self-restrict to read-only exploration before making changes
- **Git security auditing** -- automated scanning for secrets and sensitive data in repos
- **Container isolation** -- runs as unprivileged uid 1000 in Kubernetes with resource limits

### Known Limitations

These are documented for compliance transparency:

1. **Bash tool is heuristically checked, not sandboxed.** `check_bash_command()` is static analysis, not a security boundary. Mitigation: container runs as unprivileged uid 1000, K8s resource limits.
2. **No L7 traffic inspection.** URL checks are at the application layer. An agent could use `bash` + `curl` to bypass `web_fetch` checks. Mitigation: bash heuristic checks + audit logging.
3. **Single-tenant only.** PureClaw serves one authorized user. Multi-tenancy would require per-user policy scopes.

---

## Built With

- [NVIDIA Nemotron Super](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1) -- agentic supermodel (default engine)
- [vLLM](https://github.com/vllm-project/vllm) -- high-throughput model serving
- [NVIDIA RTX PRO 6000 Blackwell](https://www.nvidia.com/en-us/design-visualization/rtx-pro-6000/) -- inference hardware
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) -- Telegram integration
- [discord.py](https://github.com/Rapptz/discord.py) -- Discord integration
- [OpenAI Python SDK](https://github.com/openai/openai-python) -- vLLM client (OpenAI-compatible API)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) -- Anthropic API + prompt caching
- [Google GenAI SDK](https://github.com/googleapis/python-genai) -- Gemini API backend

---

## Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/puretensor">
        <img src="https://github.com/puretensor.png" width="100px;" alt="PureTensor"/>
        <br />
        <sub><b>PureTensor</b></sub>
      </a>
      <br />
      Architecture, design, infrastructure
    </td>
  </tr>
</table>

---

## License

MIT License. See [LICENSE](LICENSE) for details.
