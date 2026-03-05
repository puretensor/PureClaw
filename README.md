# PureClaw

**Your AI agent. Your hardware. Your rules.**

PureClaw is a personal AI agent that lives in Telegram, Discord, and your inbox. It connects to whichever AI engine you choose — a local model on your own GPU, a cloud API, or a CLI agent like Claude Code. One bot, eight engines, no lock-in.

[pureclaw.ai](https://pureclaw.ai) | [GitHub](https://github.com/puretensor/PureClaw)

---

## What It Does

You message your bot. PureClaw routes your message to whichever AI engine is active, streams the response back in real time, and gives the engine access to 19 built-in tools — shell commands, file operations, web search, phone calls, specialist agents, subagent parallelism, task tracking, and persistent memory.

Beyond conversation, PureClaw runs 20 background observers (email monitoring, threat intelligence, content publishing, daily reports, git security audits), handles email drafts with approve/reject buttons, schedules reminders, compresses long conversations automatically, and serves instant data cards for weather, markets, and trains — all from Telegram or Discord.

---

## Choose Your Engine

Eight engines. Swap anytime with `/backend` or one line in `.env`.

| Engine | Type | What It Is | Tools | Cost |
|--------|------|-----------|-------|------|
| **Ollama** (default) | Local | Any open model via [Ollama](https://ollama.com) | 19 built-in | Free |
| **vLLM** | Local | [vLLM](https://github.com/vllm-project/vllm) OpenAI-compatible endpoint | 19 built-in | Free |
| **Anthropic API** | Cloud API | Direct [Anthropic Messages API](https://docs.anthropic.com/en/docs/build-with-claude/overview) with prompt caching | 19 built-in | Pay per token |
| **AWS Bedrock** | Cloud API | Claude via [AWS Bedrock](https://aws.amazon.com/bedrock/) Converse API | 19 built-in | Pay per token |
| **Claude Code** | CLI Agent | Anthropic's [Claude Code](https://claude.ai/claude-code) CLI | Full agentic | Subscription |
| **Codex** | CLI Agent | OpenAI's [Codex](https://openai.com/index/codex/) CLI | Full agentic | Subscription |
| **Gemini** | CLI Agent | Google's [Gemini CLI](https://github.com/google-gemini/gemini-cli) | Full agentic | Subscription |
| **Hybrid** | Advanced | Routes between Bedrock API (fast) and Claude Code CLI (power) | Both | Mixed |

**Which should I pick?**

- **Just want to try it?** Start with **Ollama**. It's free, runs on your machine, and works out of the box with any GGUF model.
- **Want the best experience?** **AWS Bedrock** or **Anthropic API** give you Claude with full tool use, streaming, and prompt caching. PureClaw is developed and tested against these.
- **Have a local vLLM setup?** The **vLLM** backend works with any OpenAI-compatible endpoint — Qwen, Llama, Mistral, etc.
- **Want maximum capability?** **Hybrid** automatically routes simple queries to the fast API backend and complex tasks to Claude Code CLI.
- **Already use a CLI agent?** **Claude Code**, **Codex**, and **Gemini** delegate tool execution to their own binaries.

The API backends (Ollama, vLLM, Anthropic, Bedrock) use PureClaw's 19 built-in tools. The CLI backends (Claude Code, Codex, Gemini) bring their own sandboxes and tool execution. Hybrid uses both.

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
<summary><strong>Ollama (default, free, local)</strong></summary>

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

**That's it.** Ollama is the default — if you don't set `ENGINE_BACKEND` at all, PureClaw uses Ollama.

</details>

<details>
<summary><strong>vLLM (local, OpenAI-compatible)</strong></summary>

Install and start vLLM with your model:
```bash
pip install vllm
vllm serve your-model-name --port 8200
```

Set it in `.env`:
```bash
ENGINE_BACKEND=vllm
VLLM_URL=http://localhost:8200/v1
VLLM_MODEL=your-model-name
```

Works with any model vLLM supports — Qwen, Llama, Mistral, etc.

</details>

<details>
<summary><strong>Anthropic API (direct, pay per token)</strong></summary>

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

Verify it works:
```bash
claude -p "Say hello"
```

Set it in `.env`:
```bash
ENGINE_BACKEND=claude_code
```

Claude Code uses your Anthropic subscription (Max plan). No API key needed — authentication is handled by the CLI.

</details>

<details>
<summary><strong>Codex (OpenAI, subscription or API credit)</strong></summary>

Install the Codex CLI:
```bash
npm install -g @openai/codex
```

Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY=sk-proj-your-key-here
```

Verify it works:
```bash
codex exec "Say hello" --json --dangerously-bypass-approvals-and-sandbox
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

Authenticate (opens a browser):
```bash
gemini auth
```

Verify it works:
```bash
gemini -p "Say hello" --output-format json --yolo
```

Set it in `.env`:
```bash
ENGINE_BACKEND=gemini_cli
```

</details>

<details>
<summary><strong>Hybrid (API + CLI, advanced)</strong></summary>

The hybrid backend routes between a fast API backend (Bedrock) and a powerful CLI backend (Claude Code). Simple queries go to the API; complex multi-step tasks are handed to the CLI.

Configure both backends, then:
```bash
ENGINE_BACKEND=hybrid
HYBRID_DEFAULT=api
HYBRID_CLI_TIMEOUT=1800
HYBRID_API_TIMEOUT=300
```

</details>

### 6. Start

```bash
python3 nexus.py
```

Open your bot in Telegram and send a message. You should see a streaming response.

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
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Or use the deploy script:
```bash
bash k8s/deploy.sh
```

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
| `/backend` | Tap to select from all available engines |
| `/sonnet` | Quick-switch to Claude Sonnet |
| `/opus` | Quick-switch to Claude Opus |
| `/haiku` | Quick-switch to Claude Haiku |
| `/ollama` | Quick-switch to local model |

### Memory

PureClaw remembers things across sessions using markdown memory files. Memories persist to disk.

| Command | What it does |
|---------|-------------|
| `/remember <fact>` | Store a persistent memory (e.g. `/remember prefer dark mode`) |
| `/remember --topic infrastructure <fact>` | Store to a topic file (infrastructure, lessons, projects, etc.) |
| `/forget <key or number>` | Remove a memory |
| `/memories` | List all memories and topic files |

The AI engine also has three memory tools (`save_memory`, `read_memory`, `list_memory`) that let it read and write memories autonomously during conversation.

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

### Streaming Responses

Responses stream in real time — you see the text appear word by word in Telegram and Discord, just like ChatGPT's interface. Tool usage (file reads, shell commands, web searches) shows live status updates.

### Tool Use

When using API backends (Ollama, vLLM, Anthropic, Bedrock), PureClaw provides 19 built-in tools:

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

The CLI engines (Claude Code, Codex, Gemini) bring their own tools — they handle file operations, code execution, and web search through their own sandboxed environments. The Hybrid backend uses both.

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

### Email Draft Queue

PureClaw can monitor your email and draft replies. The workflow is human-in-the-loop:

```
Incoming email
  -> AI classifies and drafts a reply
  -> You get a Telegram notification with [Approve] [Reject] buttons
  -> Approve: sends the reply and creates a follow-up tracker
  -> Reject: discards the draft
```

You always have the final say. Nothing sends without your tap.

### Context Compression

Long conversations are compressed automatically to stay within model context limits:

- **Tier 1: Tool result truncation** — Old tool results are summarized to save tokens (zero cost, always active)
- **Tier 2: LLM summarization** — When token count exceeds a threshold, older messages are replaced with an LLM-generated summary while preserving recent messages

Configurable via `COMPRESS_TRIGGER_TOKENS`, `PRESERVE_RECENT_MESSAGES`, and `SUMMARY_MODEL`.

### Voice

Send voice notes in Telegram — they're transcribed via Whisper and processed by your AI engine. Enable `/voice on` to get audio responses back (via edge-tts or a custom TTS endpoint).

### Health Probes

Background health checking for dependent services (Whisper, TTS). Services are marked offline after consecutive failures, online after one success. Telegram voice handlers use this for graceful failover.

---

## Observers

Background tasks that run on cron schedules inside the PureClaw process. No external cron needed.

### Active Observers (registered at startup)

| Observer | Schedule | What it does |
|----------|----------|-------------|
| Email Digest | Every hour | Summarizes unread emails across configured accounts |
| Morning Brief | 7:30 AM weekdays | Combined email + weather + calendar briefing |
| Daily Snippet | 8:00 AM weekdays | Geopolitical news brief from RSS feeds |
| Bretalon Review | Every 2 hours | Reviews content submissions and editorial pipeline |
| Bretalon AutoPublish | 6:00 AM Mon/Wed-Fri | Automated article publishing pipeline |
| Bretalon Reply | Every 15 min | Monitors and responds to article comments |
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
| Git Push | Always on | Webhook listener for git push event summaries |
| Darwin Consumer | Always on | Real-time UK rail data processing (Kafka) |

### Available but Disabled

| Observer | Why disabled |
|----------|-------------|
| Node Health | Alerting handled by Alertmanager |
| Alertmanager Monitor | Alerts suppressed from agent interface |
| Intel Briefing | Replaced by Intel Deep Analysis |

Observers are optional — they run if configured but won't break anything if their dependencies aren't set up.

---

## Architecture

```
nexus.py (entry point)
  |
  +-- Engine (backends/)
  |     +-- ollama             Local models via Ollama API
  |     +-- vllm              Local models via OpenAI-compatible API
  |     +-- anthropic_api      Direct Anthropic Messages API (prompt caching)
  |     +-- bedrock_api        Claude via AWS Bedrock Converse API
  |     +-- claude_code        Claude Code CLI agent
  |     +-- codex_cli          Codex CLI agent
  |     +-- gemini_cli         Gemini CLI agent
  |     +-- hybrid             Routes between API (fast) and CLI (power)
  |     +-- tools.py           19 tools + shared execution loop + plan mode
  |
  +-- Channels
  |     +-- Telegram           Streaming, keyboards, voice, photos, documents
  |     +-- Discord            Streaming, slash commands
  |     +-- Email Input        IMAP polling, classification, draft generation
  |
  +-- Observers (20)           Background tasks on cron schedules + persistent threads
  +-- Dispatcher               Instant data cards (weather, markets, trains, nodes)
  +-- Draft Queue              Email drafts with Telegram approve/reject
  +-- Scheduler                User-defined tasks and reminders
  +-- Memory                   Persistent markdown memory (MEMORY.md + topic files)
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
| `ENGINE_BACKEND` | `ollama` | Which engine: `ollama`, `vllm`, `anthropic_api`, `bedrock_api`, `claude_code`, `codex_cli`, `gemini_cli`, `hybrid` |

### Ollama

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `qwen3:235b` | Model name (must be pulled first) |
| `OLLAMA_TOOLS_ENABLED` | `true` | Enable/disable tool use |
| `OLLAMA_TOOL_MAX_ITER` | `25` | Max tool call iterations per response |
| `OLLAMA_TOOL_TIMEOUT` | `30` | Per-tool-call timeout (seconds) |
| `OLLAMA_NUM_PREDICT` | `8192` | Max tokens per response |

### vLLM

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_URL` | `http://localhost:8200/v1` | vLLM OpenAI-compatible endpoint |
| `VLLM_MODEL` | (your model path) | Model name or path |
| `VLLM_TOOLS_ENABLED` | `true` | Enable/disable tool use |
| `VLLM_TOOL_MAX_ITER` | `10` | Max tool call iterations per response |
| `VLLM_TOOL_TIMEOUT` | `60` | Per-tool-call timeout (seconds) |
| `VLLM_TOTAL_TIMEOUT` | `300` | Total request timeout (seconds) |
| `VLLM_MAX_TOKENS` | `8192` | Max tokens per response |

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
| `GEMINI_CLI_MODEL` | (Gemini default) | Model name (e.g. `gemini-2.5-flash`) |

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

## Project Structure

```
PureClaw/
+-- nexus.py                    Entry point — starts all subsystems
+-- config.py                   Environment loading, system prompt, logging
+-- db.py                       SQLite: sessions, drafts, follow-ups, tasks
+-- engine.py                   Engine abstraction (sync + streaming)
+-- memory.py                   Persistent markdown memory system
+-- scheduler.py                Task scheduler (/schedule, /remind)
+-- context_compression.py      Two-tier conversation compression
+-- health_probes.py            Background service health checking
|
+-- backends/
|     +-- base.py               Backend Protocol definition
|     +-- ollama.py             Ollama backend with tool loop
|     +-- vllm.py               vLLM OpenAI-compatible backend
|     +-- anthropic_api.py      Anthropic Messages API backend
|     +-- bedrock_api.py        AWS Bedrock Converse API backend
|     +-- claude_code.py        Claude Code CLI backend
|     +-- codex_cli.py          Codex CLI backend
|     +-- gemini_cli.py         Gemini CLI backend
|     +-- hybrid.py             API + CLI routing backend
|     +-- tools.py              19 tools + shared execution loop + plan mode
|     +-- __init__.py           Backend factory (lazy singleton)
|
+-- channels/
|     +-- telegram/             Bot setup, commands, callbacks, streaming
|     +-- discord/              Discord bot handlers and streaming
|     +-- email_in.py           IMAP polling email input
|
+-- observers/                  20 background observers (cron + persistent)
+-- dispatcher/                 Data cards (weather, markets, trains, nodes)
+-- drafts/                     Email draft queue with approve/reject
+-- handlers/                   Telegram handlers (voice, photo, document, location)
+-- tools/                      Integration tools (Gmail, Google Calendar)
+-- prompts/                    System prompts
+-- failover/                   Automatic failover runner
+-- k8s/                        Kubernetes deployment manifests
+-- tests/                      Test suite (886 tests)
+-- Dockerfile                  Container build
+-- nexus.service               systemd unit file
```

---

## Testing

```bash
python3 -m pytest tests/ -v
```

886 tests covering backends, tools, observers, channels, handlers, and integration points.

---

## Requirements

- Python 3.11+
- A Telegram bot token
- At least one engine: Ollama installed, a vLLM endpoint, an API key, or a CLI tool authenticated

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

- **ffmpeg** — Required for voice output
- **ripgrep** — Used by the `grep` tool (falls back to Python regex if absent)
- **Node.js 18+** — Required for CLI engines (claude, codex, gemini) and document generation skills
- **Ollama** — Required for the `ollama` backend
- **pandoc** — Used for document format conversion

---

## Security

- **Single-user only** — locked to one Telegram/Discord user ID
- **No telemetry** — no analytics, no tracking, no phoning home
- **No cloud dependency** — Ollama and vLLM keep everything local; cloud backends use your own keys/subscriptions
- **Human-in-the-loop email** — drafts require explicit approval before sending
- **Plan mode** — AI can self-restrict to read-only exploration before making changes
- **Git security auditing** — automated scanning for secrets and sensitive data in repos
- **Your hardware** — not a managed platform, not a SaaS product

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
