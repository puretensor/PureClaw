# Codex Audit: PureClaw (Nexus) -- Multi-Channel Agentic AI Platform

## What You're Reviewing

PureClaw is a production multi-channel agentic AI platform. 50K lines of Python, 149 files. It runs 24/7 on bare metal (AMD Threadripper PRO 9975WX, 512GB RAM, 2x RTX PRO 6000 Blackwell) as the primary AI agent for PureTensor infrastructure.

It handles Telegram, Discord, WhatsApp, email, and terminal channels. It has 8 swappable LLM backends (local GPU + cloud), 19 tools (including shell execution), 30+ background observers, declarative YAML security policies, and a mesh networking layer for multi-agent coordination.

**This is not a chatbot.** It executes shell commands, sends emails, manages infrastructure, publishes content, and makes autonomous decisions. It runs as root on the primary compute node.

## Architecture Overview

```
Channels (Telegram, Discord, WhatsApp, Email, Terminal)
    |
    v
nexus.py (main entrypoint, channel orchestration)
    |
    v
engine.py (LLM dispatch, tool loop, conversation management)
    |
    ├── backends/ (8 LLM backends: vllm, ollama, anthropic, bedrock, claude_code, codex, gemini, hybrid)
    ├── tools/ (shell, file I/O, web, email, calendar, subagents, memory, tasks)
    ├── dispatcher/ (data cards: weather, markets, trains, infra status -- bypass LLM)
    ├── handlers/ (document, photo, voice, vision, file output, keyboards)
    ├── security/ (YAML policies, filesystem ACLs, SSRF protection, credential redaction, model allowlists)
    ├── mesh/ (multi-agent networking: registry, authority, message passing)
    ├── observers/ (30+ background cron tasks: email digest, threat intel, daily reports, git audits, etc.)
    ├── failover/ (automatic backend failover)
    ├── hal_mail/ (email processing: IMAP polling, 4-lane classification, draft queue)
    └── prompts/ (system prompts per agent persona)

Supporting:
    config.py (all configuration)
    db.py (SQLite persistence)
    memory.py (persistent agent memory)
    scheduler.py (cron-based observer scheduling)
    context_compression.py (conversation context management)
    health_probes.py (K8s liveness/readiness)
```

## Files to Read

Read every Python file in the repo. Start with these in order for architectural understanding:

### Core (read first)
1. `nexus.py` -- main entrypoint, channel setup, startup
2. `engine.py` -- LLM dispatch, tool execution loop, conversation management
3. `config.py` -- all configuration and environment variables
4. `db.py` -- SQLite persistence layer
5. `memory.py` -- persistent agent memory system
6. `scheduler.py` -- observer scheduling

### Backends (LLM integration)
7. `backends/base.py` -- backend interface
8. `backends/vllm.py` -- NVIDIA Nemotron via vLLM (primary local)
9. `backends/anthropic_api.py` -- Anthropic direct API
10. `backends/bedrock_api.py` -- AWS Bedrock
11. `backends/claude_code.py` -- Claude Code CLI agent
12. `backends/codex_cli.py` -- Codex CLI agent
13. `backends/hybrid.py` -- API + CLI routing
14. `backends/tools.py` -- tool definitions shared across backends
15. `backends/ollama.py` -- Ollama local inference

### Security (critical path)
16. `security/policy.py` -- YAML policy engine
17. `security/filesystem.py` -- filesystem ACLs
18. `security/network.py` -- SSRF protection
19. `security/redact.py` -- credential redaction
20. `security/audit.py` -- security audit logging
21. `security/inference.py` -- model allowlists
22. `security/*.yaml` -- policy definitions

### Channels
23. `channels/base.py` -- channel interface
24. `channels/telegram/` -- Telegram (primary channel, streaming, commands, callbacks)
25. `channels/discord/` -- Discord
26. `channels/whatsapp/` -- WhatsApp multi-instance bridge
27. `channels/email_in.py` -- email ingestion
28. `channels/terminal/` -- WebSocket terminal

### Tools & Handlers
29. `tools/` -- built-in tools (shell, gmail, calendar)
30. `handlers/` -- document, photo, voice, vision, file output

### Observers
31. `observers/base.py` -- observer interface
32. `observers/registry.py` -- observer registration
33. All observer files in `observers/`

### Mesh
34. `mesh/` -- multi-agent networking layer

### Other
35. `hal_mail/` -- email processing pipeline
36. `context_compression.py` -- context window management
37. `health_probes.py` -- K8s probes
38. `failover/runner.py` -- backend failover logic
39. `prompts/` -- system prompts

## What I Want From You

### 1. Architecture Review
- Is the monolithic nexus.py + engine.py core the right design, or should it be decomposed?
- How well does the backend abstraction work across 8 very different LLM providers?
- Is the observer pattern (30+ background tasks) sustainable? Are there scheduling/resource conflicts?
- Is the mesh networking layer well-designed for multi-agent coordination?
- How does the system handle graceful degradation when backends/channels/observers fail?

### 2. Code Quality & Correctness
- Find bugs. Race conditions, resource leaks, unhandled edge cases.
- Are there inconsistencies between how different backends handle tool use, streaming, and errors?
- Is the conversation/context management sound? Memory leaks? Unbounded growth?
- Is the SQLite usage safe under concurrent access from multiple channels and observers?

### 3. Security Audit
- The system executes shell commands via tools. Is the security policy enforcement complete?
- Can the YAML security policies be bypassed? Are there gaps in filesystem ACLs or SSRF protection?
- Is credential redaction thorough? What could leak through logs, error messages, or LLM context?
- Are the channel authentication/authorization models sound (Telegram user ID, Discord roles, etc.)?
- Could a malicious message from any channel lead to unauthorized actions?
- What would a red team target first?

### 4. Backend & Tool System Review
- Is the tool execution loop safe across all 8 backends?
- How robust is the failover logic? Are there edge cases where failover loops or fails silently?
- Is the hybrid routing (API for simple queries, CLI for complex) well-implemented?
- Are there token/cost management issues across the different pricing models?

### 5. Channel Review
- Is the Telegram integration production-quality? Streaming, error handling, rate limiting?
- Is the WhatsApp bridge (baileys) reliable? What are the failure modes?
- Is the email pipeline (IMAP polling, 4-lane classification, draft queue) robust?
- Are there cross-channel consistency issues?

### 6. Observer System Review
- Are the 30+ observers well-isolated from each other?
- What happens when an observer fails? Does it affect other observers or the main agent?
- Are there observers that should be refactored or removed?
- Is the scheduling system robust (missed schedules, overlapping runs, resource contention)?

### 7. Concrete Improvement Recommendations
For each finding, provide:
- **Severity**: Critical / High / Medium / Low
- **Category**: Bug, Security, Architecture, Performance, Reliability, Missing Feature
- **Description**: What's wrong or missing
- **Recommendation**: Specific fix with code-level guidance
- **Priority**: Do now / Next sprint / Backlog

## Output Format

Structure your response as:

1. **Executive Summary** (3-5 sentences)
2. **Architecture Assessment** (each major subsystem)
3. **Critical Findings** (data loss, security breach, cascading failure risk)
4. **High Priority Findings** (bugs, reliability gaps, security holes)
5. **Medium/Low Findings** (code quality, performance, missing features)
6. **Prioritized Roadmap** (top 10 improvements, ordered by impact/effort)

This platform is public on GitHub and runs 24/7 on production infrastructure. Be thorough.
