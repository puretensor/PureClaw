# Security Audit Report — PureTensor Public Repositories

**Date:** 2026-03-25
**Auditor:** Claude Code (automated security scan)
**Scope:** puretensor/pureclaw (local clone + GitHub MCP); 9 additional repos attempted

---

## Repositories Scanned

| Repository | Status | Notes |
|---|---|---|
| puretensor/pureclaw | ✅ Scanned | Local clone + GitHub MCP history |
| puretensor/bookengine | ❌ Skipped | MCP restricted to pureclaw only; git clone unavailable in this environment |
| puretensor/varangian-ai | ❌ Skipped | Same |
| puretensor/voice-kb | ❌ Skipped | Same |
| puretensor/ecommerce-agent | ❌ Skipped | Same |
| puretensor/kalima | ❌ Skipped | Same |
| puretensor/kalima-android | ❌ Skipped | Same |
| puretensor/echo-voicememo | ❌ Skipped | Same |
| puretensor/arabic-qa | ❌ Skipped | Same |
| puretensor/autopen | ❌ Skipped | Same |

> **Limitation:** The GitHub MCP server is scoped exclusively to `puretensor/pureclaw`. Outbound `git clone` is unavailable in this environment. The 9 additional repos require a separate audit run with broader access. This report covers **pureclaw only**.

---

## Summary of Findings

| Severity | Count |
|---|---|
| CRITICAL | 1 |
| WARNING | 10 |
| INFO | 5 |

---

## CRITICAL Findings

### [CRIT-1] BMC/IPMI Fleet Password Hardcoded in System Prompt

- **Repo:** puretensor/pureclaw
- **File:** `prompts/claw_infra_prompt.md`
- **Line:** 56
- **Content:** `- All BMC passwords: cons...curl`
- **Confirmed in git history:** Yes — added when the prompt was first committed and still present in HEAD.

**What this exposes:** A single shared BMC/IPMI password that grants hardware-level (lights-out) management access to the **entire physical server fleet** — tensor-core, fox-n0, fox-n1, arx1-4, mon1-3. An attacker with this password can power cycle, reflash firmware, mount virtual media, and access the server console on every machine without needing OS credentials or network authentication beyond reaching the BMC interface.

**Immediate action required:** Rotate all BMC passwords across the fleet. Do not reuse the exposed value. Consider per-node unique passwords going forward.

---

## WARNING Findings

### [WARN-1] Full Internal Network Topology Committed

- **Repo:** puretensor/pureclaw
- **Files:**
  - `prompts/claw_infra_prompt.md` lines 51–53
  - `deploy/claw-infra.env` line 5
  - `deploy/claw-ops.env` lines 5, 9
  - `deploy/claw-sentinel.env` line 5
  - `k8s/configmap.yaml` lines 9, 18, 22, 45–57

**IPs exposed (Tailscale 100.x.x.x range, LAN 192.168.x.x, fabric 10.200.0.x):**

| Node | LAN IP | Tailscale IP | 200G Fabric IP |
|---|---|---|---|
| tensor-core | 192.168.4.217 | 100.121.42.54 | 10.200.0.3 |
| fox-n0 | 192.168.4.184 | 100.69.225.18 | 10.200.0.1 |
| fox-n1 | 192.168.4.50 | 100.103.248.9 | 10.200.0.2 |

**Additional service endpoints in `k8s/configmap.yaml`:**
- vLLM: `http://100.121.42.54:5000/v1`
- Vision model: `http://100.121.42.54:5001/v1`
- Whisper STT: `http://100.121.42.54:9000/transcribe`
- TTS: `http://100.121.42.54:5580`
- Ollama: `http://100.121.42.54:11434`
- Prometheus: `http://100.80.213.1:9090`
- Alertmanager: `http://100.80.213.1:9093`
- Ceph mgr metrics: `http://100.80.213.1:9283/metrics`
- Gitea: `http://100.92.245.5:3002`
- SearXNG: `http://100.105.43.27:8080/search`
- Failover vLLM chain: `http://10.200.0.3:5000/v1`

**Why this matters:** Tailscale IPs are meant to be private to the Tailnet. Publishing them allows anyone who also joins or breaches the Tailnet to immediately map the entire fleet. Combined with CRIT-1, the attack surface is fully charted.

---

### [WARN-2] Real Telegram User ID Committed in Deployed .env Files

- **Repo:** puretensor/pureclaw
- **Files:**
  - `deploy/claw-infra.env` (line ~17)
  - `deploy/claw-ops.env`
  - `deploy/claw-sentinel.env`
- **Value:** `AUTHORIZED_USER_ID=2227...8981`
- **Confirmed in git history:** Yes (`+AUTHORIZED_USER_ID=2227...8981` appears in 3 separate commits).

This is the Telegram user ID used as the sole authorization gate for the PureClaw bot. Publishing it doesn't allow impersonation, but it confirms which Telegram account owns the bot and can be used for social engineering or targeted phishing.

---

### [WARN-3] Hardcoded Internal Tailscale IP in Source Code

- **Repo:** puretensor/pureclaw
- **File:** `tools/nexus-terminal.py`
- **Line:** 48
- **Content:** `DEFAULT_HOST = "100.103...48.9"` (fox-n1 Tailscale IP)

This is hardcoded into a distributed tool script, not just a config file. Anyone reading the source immediately knows fox-n1's Tailscale address.

---

### [WARN-4] Company Physical Address Hardcoded in Multiple Source Files

- **Repo:** puretensor/pureclaw
- **Files:**
  - `observers/doc_compiler.py` lines 241–242
  - `observers/daily_report.py` lines 226–227
  - `observers/weekly_report.py` lines 278–279
  - `prompts/pureclaw_context.md` line 174
- **Content:** `131 Continental Dr, Suite 305, Newark, DE 19713, US`

A registered business address is technically public, but embedding it as a hardcoded constant in multiple source files means it will appear on every generated PDF regardless of deployment context. If this project is used by others, their generated documents will incorrectly carry PureTensor's address. It should be a config value.

---

### [WARN-5] Telegram Bot Handles Exposed

- **Repo:** puretensor/pureclaw
- **Files:**
  - `k8s/deploy.sh` line 300 — `@puretensor_claude_bot`
  - `observers/node_health.py` line 167 — `@puretensor_alert_bot`

Both bot usernames are in the public repo. Anyone can now attempt to interact with or probe these bots. Low severity on its own, but combined with the user ID (WARN-2), an attacker knows which bot to target and who controls it.

---

### [WARN-6] SSH System Username Exposed as Code Default

- **Repo:** puretensor/pureclaw
- **Files:**
  - `observers/git_security_audit.py` line 26
  - `observers/git_auto_sync.py` line 26
  - `observers/github_activity.py` line 38
  - `observers/pipeline_watchdog.py` line 149
  - `nexus.service` line 8 (`User=puretensorai`)
  - `config.py` line 20 (`CLAUDE_CWD = ".../home/puretensorai"`)
- **Value:** OS username `puretensorai` and home path `/home/puretensorai`

The default SSH user is baked into code constants. This means the OS username is known without any authentication attempt, reducing the work needed for a brute-force or credential-stuffing attack.

---

### [WARN-7] Internal Webroot Paths Committed

- **Repo:** puretensor/pureclaw
- **Files:**
  - `failover/runner.py` lines 54–55
  - `observers/intel_briefing.py` line 55
  - `observers/cyber_threat_feed.py` line 35
- **Values:** `/var/www/cyber.puretensor.ai`, `/var/www/intel.puretensor.ai`

Internal server filesystem layout. Useful for path traversal if combined with another vulnerability.

---

### [WARN-8] Claude Memory File Path Exposed

- **Repo:** puretensor/pureclaw
- **File:** `observers/memory_sync.py` lines 30–31
- **Value:** `~/.claude/projects/-home-puretensorai/memory/MEMORY.md`

Reveals the exact path to the agent's persistent memory file on tensor-core. Useful for an attacker who gains shell access, to immediately locate and read or poison the agent's long-term memory.

---

### [WARN-9] Agent Email Addresses Committed in Kubernetes ConfigMaps

- **Repo:** puretensor/pureclaw
- **Files:**
  - `k8s/configmap.yaml` line 61
  - `k8s/hal-mail-configmap.yaml` line 39
- **Value:** `hal@puretensors.com, hal@puretensor.com, hal@puretensor.ai`

Live agent email addresses (not placeholders). Enables targeted phishing or impersonation of the AI agent.

---

### [WARN-10] Potentially Real xAI API Token in Test Fixture

- **Repo:** puretensor/pureclaw
- **File:** `tests/test_security_redact.py` lines 54, 106
- **Value:** `xai-HDG...fc3` (format matches real xAI API key)

This token appears in a test that validates the redaction module's ability to detect xAI tokens. It is likely a synthetic test value, but it follows the exact format of a real xAI API key. If it was copy-pasted from an actual key for testing and never invalidated, it may still be active. **Verify with xAI dashboard that this key is not live.**

---

## INFO Findings

### [INFO-1] Deployed .env Files Are Not Gitignored

- **File:** `.gitignore`
- **Issue:** `.gitignore` excludes `.env` and `.env.*` at the root, but `deploy/claw-infra.env`, `deploy/claw-ops.env`, and `deploy/claw-sentinel.env` do not match this pattern (they use a flat name without the leading `.`). These files are tracked and contain real operational config including internal IPs and the Telegram user ID.
- **Recommendation:** Either add `deploy/*.env` to `.gitignore` and replace committed files with `*.env.example` templates, or strip the sensitive values and use `SET_ME` placeholders like the k8s secrets.yaml does.

---

### [INFO-2] Kubernetes ConfigMap Contains Actual Service IPs Instead of Templated Values

- **File:** `k8s/configmap.yaml`
- **Issue:** The configmap uses hardcoded Tailscale IPs rather than placeholder values. This means every `kubectl apply` on any cluster will configure real production addresses. A separate `configmap.example.yaml` or Helm values file would be safer.

---

### [INFO-3] Empty Accidental File Committed

- **File:** `=TLSv1.2`
- **Commit:** `38776a2a` ("feat: add TLSv1.2 configuration file")
- **Issue:** Empty file with a name that looks like a shell argument fragment (`=TLSv1.2` from `curl --tlsv1.2`). Committed by the HAL agent. Not a security risk, but indicates the agent accidentally committed an artifact.

---

### [INFO-4] Observer State Files Committed Despite .gitignore Entry

- **Files:** `observers/.state/health_2026-03-18.json`, `health_2026-03-19.json`, `health_2026-03-24.json`
- **Issue:** `.gitignore` includes `.state/` but these files are tracked (likely added before the gitignore entry). State files contain timestamps and observer names. Low sensitivity currently, but future state files could contain more operational detail.
- **Fix:** `git rm --cached observers/.state/*.json`

---

### [INFO-5] .gitignore Does Not Cover `prompts/` Directory

- **Issue:** The `prompts/` directory contains agent system prompts with operational detail (fleet IPs, BMC credentials — see CRIT-1). This directory has no .gitignore exclusion. Any additions to it are automatically tracked.
- **Recommendation:** Review whether any prompts should be gitignored or templated; at minimum, strip all IPs and credentials from prompts before committing.

---

## Remediation Priority

1. **Immediately:** Rotate all BMC/IPMI passwords across the fleet (CRIT-1). The current password has been public for the entire git history of this repo.
2. **Today:** Verify `xai-HDG...fc3` is not an active API key (WARN-10). If real, revoke it.
3. **This week:**
   - Remove or template `deploy/*.env` files — strip real values, add `deploy/*.env` to .gitignore.
   - Remove hardcoded Tailscale/LAN/fabric IPs from `prompts/claw_infra_prompt.md` and replace with `$VARIABLE` references or SSH config hostnames.
   - Move `DEFAULT_HOST` in `tools/nexus-terminal.py` to a config file or env var.
4. **Next sprint:**
   - Move company address to config/env.
   - Template `k8s/configmap.yaml` IP values.
   - Clean up the committed state files.
   - Run `git filter-repo` or BFG Repo Cleaner to scrub historical commits of BMC credentials, IPs, and user ID if the repo is truly public (note: this rewrites history and requires force-push coordination).

---

## Repos Still Requiring Audit

The following 9 repos could not be scanned in this session due to environment constraints. They must be audited separately with direct git access:

- puretensor/bookengine
- puretensor/varangian-ai
- puretensor/voice-kb
- puretensor/ecommerce-agent
- puretensor/kalima
- puretensor/kalima-android
- puretensor/echo-voicememo
- puretensor/arabic-qa
- puretensor/autopen
