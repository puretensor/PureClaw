# PureTensor Public Repository Security Audit

**Date:** 2026-04-01 (supersedes 2026-03-29 audit)
**Auditor:** Claude Code (automated) — full repo + full git history scan
**Scope:** All accessible PureTensor public repositories

---

## Repository Access Summary

| Repository | Status | Commits Scanned |
|---|---|---|
| `puretensor/pureclaw` | **Scanned** | Full history |
| `puretensor/autopen` | **Scanned** | 6 commits |
| `puretensor/bookengine` | **Scanned** | 15 commits |
| `puretensor/echo-voicememo` | **Scanned** | 8 commits |
| `puretensor/kalima` | **Scanned** | 10 commits |
| `puretensor/varangian-ai` | **Private / Inaccessible** | — |
| `puretensor/voice-kb` | **Private / Inaccessible** | — |
| `puretensor/ecommerce-agent` | **Private / Inaccessible** | — |
| `puretensor/kalima-android` | **Private / Inaccessible** | — |
| `puretensor/arabic-qa` | **Private / Inaccessible** | — |

---

## PureClaw — Findings

### CRITICAL

---

**[CRITICAL-1] BMC/IPMI Fleet Password in Git History**

- **File:** `prompts/claw_infra_prompt.md`
- **Commit added:** `4cfa8e9` (feat: Add Claw)
- **Commit redacted:** `6402aad` (2026-03-29) — replaced with `<BMC_PASSWORD>` placeholder
- **Redacted value:** `cons****-craz****-curl` (3-word passphrase format)
- **Current HEAD:** Placeholder only (`<BMC_PASSWORD>` at line 56) — but the real value is **permanently embedded in git history**
- **Impact:** This password grants BMC/IPMI access to the entire hardware fleet: tensor-core, fox-n0, fox-n1, arx1–arx4, mon1–mon3. BMC/IPMI access enables remote power control (on/off/reset), serial console, and potential firmware-level access to all hardware.
- **Status:** If this repository was ever public on GitHub, the value has been indexed. Assume fully compromised.
- **Recommendation:** **Rotate this password immediately on all nodes.** A git history rewrite (BFG/filter-repo + force-push) is also warranted if no public forks exist, but rotation takes priority.

---

**[CRITICAL-2] Full Fleet IP Mapping in Git History**

- **File:** `prompts/claw_infra_prompt.md`
- **Commit:** `4cfa8e9` (present before `6402aad` redaction; placeholders now in HEAD)
- **Values permanently in history:**

| Node | LAN IP | Tailscale IP | Fabric IP |
|------|--------|--------------|-----------|
| tensor-core | `192.168.4.217` | `100.121.42.54` | `10.200.0.3` |
| fox-n0 | `192.168.4.184` | `100.69.225.18` | `10.200.0.1` |
| fox-n1 | `192.168.4.50` | `100.103.248.9` | `10.200.0.2` |

- **Impact:** Combined with CRITICAL-1, this constitutes a complete remote attack kit: known credentials plus exact targets. Even independently, it maps the entire private network topology.
- **Recommendation:** Consider rotating Tailscale node keys for exposed nodes. Review Tailscale ACL rules to ensure minimum-necessary access.

---

### WARNING

---

**[WARNING-1] All Internal Service Tailscale IPs and Ports in Git History**

- **Files:** `k8s/configmap.yaml` (history), `deploy/claw-*.env` (history)
- **Commits:** Pre-`6402aad` versions
- **Values permanently in history:**

| Service | Tailscale IP | Port |
|---------|-------------|------|
| vLLM (tensor-core) | `100.121.42.54` | 5000 |
| Vision model | `100.121.42.54` | 5001 |
| Whisper STT | `100.121.42.54` | 9000 |
| TTS | `100.121.42.54` | 5580 |
| Ollama | `100.121.42.54` | 11434 |
| Prometheus (mon2) | `100.80.213.1` | 9090 |
| Alertmanager (mon2) | `100.80.213.1` | 9093 |
| Gitea (mon1) | `100.92.245.5` | 3002 |
| SearXNG | `100.105.43.27` | 8080 |
| K3s NodePort (fox-n1) | `100.103.248.9` | 30876 |
| Fabric mesh (fox-n0) | `10.200.0.1` | 9880 |
| Mesh peer (mon2) | `100.80.213.1` | 9880 |

- **Also in current HEAD:** `tools/nexus-terminal.py:48` (`DEFAULT_HOST = "100.103.248.9"`), `prompts/claw_ops_prompt.md:46` (Ceph mgr REST URL with `100.80.213.1`), `security_audit.md` lines 68–69.
- **Recommendation:** Use Tailscale MagicDNS hostnames instead of IPs. Remove remaining live IPs from current HEAD (nexus-terminal.py, claw_ops_prompt.md, this security_audit.md document).

---

**[WARNING-2] Telegram Operator User ID in Git History**

- **Files:** `deploy/claw-infra.env`, `deploy/claw-ops.env`, `deploy/claw-sentinel.env` (all in history before `6402aad`)
- **Variable:** `AUTHORIZED_USER_ID`
- **Redacted value:** `2227****` (numeric Telegram user ID)
- **Current HEAD:** Replaced with `SET_ME` placeholder
- **Impact:** Exposes the Telegram user ID of the bot's authorized operator — enables identity confirmation and targeted social engineering.

---

**[WARNING-3] Deploy `.env` Files with Real Infrastructure Config Committed**

- **Files:** `deploy/claw-infra.env`, `deploy/claw-ops.env`, `deploy/claw-sentinel.env`
- **Finding:** These are real deployment environment files containing actual service architecture, vLLM model names, service ports, filesystem paths (including username `puretensorai`), and `SET_ME` placeholders for secrets. They are not covered by `.gitignore` (`deploy/*.env` pattern is missing).
- **Recommendation:** Add `deploy/*.env` to `.gitignore`. Replace tracked files with `.env.example` templates.

---

**[WARNING-4] Comprehensive Internal Infrastructure Topology in Public System Prompt**

- **File:** `prompts/pureclaw_context.md`
- **Finding:** The context document committed to the public repo contains: hardware specs for every node (CPU, RAM, NVMe, GPU), tier layout (Tier 0–3 + GCP), node roles and services, storage layout (RAID configs, ZFS pools, Ceph OSD counts/sizes), SSH patterns, GCP instance types, hosted domains (`bretalon.com`, `nesdia.com`), and IMAP account handles.
- **Impact:** Provides a detailed attack surface map. An adversary knows exactly what hardware to target, how storage is organized, and what services are hosted where.
- **Recommendation:** Replace with a generic/redacted template in the repo; inject real infrastructure details at runtime via deployment secrets.

---

**[WARNING-5] Company Registered Address Hardcoded in Production Code**

- **Files:** `prompts/pureclaw_context.md:174`, `observers/doc_compiler.py:241`, `observers/daily_report.py:226`, `observers/weekly_report.py:278`
- **Finding:** `"131 Continental Dr, Suite 305 / Newark, DE 19713, US"` hardcoded as a fixed literal.
- **Recommendation:** Move to a configurable env var (e.g. `COMPANY_ADDRESS`).

---

**[WARNING-6] Agent Email Addresses in K8s ConfigMap**

- **Files:** `k8s/configmap.yaml:61`, `k8s/hal-mail-configmap.yaml:39`
- **Finding:** `AGENT_EMAIL: "hal@puretensors.com,hal@puretensor.com,hal@puretensor.ai"` — three real operational email addresses committed in a public ConfigMap.
- **Also:** `observers/daily_snippet.py:~897` (`hal@puretensor.ai` hardcoded default), `tools/gmail.py:67` (`ops@puretensor.ai` hardcoded).
- **Recommendation:** Move to K8s Secrets or environment variables. Remove hardcoded defaults from source.

---

### INFO

---

**[INFO-1] `.gitignore` Missing Entry for `deploy/*.env`**

- **File:** `.gitignore`
- **Finding:** `.env` and `.env.*` are excluded at root, and `failover/.env*` is covered, but `deploy/claw-*.env` falls outside these patterns and is tracked by git.
- **Recommendation:** Add `deploy/*.env` (and optionally `deploy/*.conf`) to `.gitignore`.

---

**[INFO-2] `observers/.state/` Directory Committed**

- **Files:** `observers/.state/health_2026-03-18.json` through recent dates
- **Finding:** Runtime state files are committed. Contents are benign (run counts, timestamps). `.gitignore` lists `.state/` — these files predate or were committed explicitly.
- **Recommendation:** `git rm --cached observers/.state/` and ensure the `.gitignore` entry covers them going forward.

---

**[INFO-3] `=TLSv1.2` Junk File at Repo Root**

- **File:** `=TLSv1.2` (empty file at repo root)
- **Finding:** Likely a copy-paste error from a curl command. No sensitive data.
- **Recommendation:** Delete this file.

---

**[INFO-4] Filesystem Paths Expose Deployment Username**

- **Files:** `deploy/claw-infra.env:25`, `deploy/claw-ops.env:25`, `deploy/claw-sentinel.env:19`
- **Finding:** Absolute paths `/home/puretensorai/nexus/…` reveal the deployment OS username. Moot once `deploy/*.env` files are removed from git (WARNING-3/INFO-1).

---

**[INFO-5] Security Policy Files Disable Private Range Blocking**

- **Files:** `security/claw_infra_policy.yaml:28`, `security/claw_ops_policy.yaml:28`, `security/claw_sentinel_policy.yaml:29`
- **Finding:** `block_private_ranges: false` for all three Claw agents. Intentional (they need LAN/Tailscale access) but leaves these agents without SSRF protection.
- **Recommendation:** Add a comment in each policy file explicitly documenting why this is intentional.

---

## echo-voicememo — Findings

---

**[WARNING-7] Production OS Username and Paths in Committed Systemd Service Files**

- **Files:** `xtts-spanish.service` (lines 7, 10, 13, 14), `echo-voicememo-api.service` (lines 8, 9, 10, 11)
- **Finding:** Both service unit files hardcode `User=puretensorai` and absolute paths: `/home/puretensorai/…`, `/opt/xtts-v2/finetune_output/run/training/XTTS_v2.0_original_model_files`.
- **Impact:** Exposes the production OS username and deployment directory layout in a public repo. Aids privilege escalation if any access is obtained.
- **Recommendation:** Template service files using variables or document them as examples requiring customization before deployment.

---

**[INFO-6] Hardcoded Absolute Model Path in Source**

- **File:** `xtts_spanish_server.py:37`
- **Finding:** `BASE_DIR = "/opt/xtts-v2/finetune_output/run/training/XTTS_v2.0_original_model_files"` — production path embedded in source.
- **Recommendation:** Use an environment variable or config file for the model base path.

---

**[INFO-7] No Authentication on API**

- **File:** `server.py`
- **Finding:** The API has no authentication layer; `X-Device-Id` is used for scoping only. Acceptable for a local-only service but dangerous if network-exposed.

---

**[INFO-8] Unauthenticated Google Translate Usage**

- **Files:** `config.py:20`, `pipeline.py:147-150`
- **Finding:** Uses `deep_translator.GoogleTranslator` (scraping-based, no API key), sending memo content to Google without a formal data processing agreement.

---

## kalima — Findings

---

**[WARNING-8] API Key Accepted as URL Query Parameter**

- **Files:** `src/kalima/api/websocket.py:106`, `README.md:82,104`, `docs/android-ime-research.md:578`
- **Finding:** `api_key` is accepted as a WebSocket URL query parameter (`?api_key=…`). This leaks credentials into server access logs, browser history, proxy logs, and referrer headers.
- **Note:** This was previously flagged in the project's own internal audit at `tasks/codex_audit_report.md:35`.
- **Recommendation:** Remove query-parameter key acceptance. Enforce header-only authentication (`X-API-Key`).

---

**[WARNING-9] Silent Auth Bypass When `KALIMA_API_KEYS` Is Unset**

- **Files:** `src/kalima/main.py:35-36`, `src/kalima/config.py:23`
- **Finding:** If `KALIMA_API_KEYS` environment variable is absent, the valid key set is empty and all requests are accepted without authentication. The `.env.example` placeholder `change-me` is not rejected by the application.
- **Recommendation:** Add an explicit startup check that refuses to start (or logs a prominent warning) when `KALIMA_API_KEYS` is unset in non-dev mode.

---

**[WARNING-10] Production Username and Full Path in Documentation**

- **File:** `docs/research/arabic-keyboard-dictation-ux.md:380`
- **Finding:** Full path `/home/puretensorai/kalima/src/kalima/asr/deepgram.py` hardcoded — reveals the production OS username `puretensorai` and deployment directory.
- **Recommendation:** Replace with a generic path in documentation.

---

**[INFO-9] Android Emulator IP in Documentation**

- **File:** `docs/android-ime-research.md:1019`
- **Finding:** `ws://10.0.2.2:8000` appears in example code. `10.0.2.2` is the standard Android emulator loopback alias for the host machine. Not a real internal host. Included for completeness.

---

## autopen — Findings

**No credential findings.** All 6 commits scanned. No API keys, tokens, passwords, private keys, or internal IPs in any file or git history.

**[INFO-10] `.gitignore` Missing Some Credential File Patterns**

- **File:** `.gitignore`
- **Finding:** Missing: `*.p12`, `*.pfx`, `credentials*`, `secrets/`, `*service_account*.json`
- **Recommendation:** Add these patterns as hardening. The project currently has no such files but adding coverage prevents future accidents.

---

## bookengine — Findings

**No credential findings.** All 15 commits scanned. Single Python file using only local Ollama — no cloud API keys required or present.

**[INFO-11] `.gitignore` Missing Some Credential File Patterns**

- **File:** `.gitignore`
- **Finding:** Missing: `*.p12`, `*.pfx`
- **Recommendation:** Minor hardening addition.

---

## Cross-Repo: Domain Discrepancy

**[INFO-12] `puretensors.com` (with "s") vs `puretensor.ai` in Git Commit Metadata**

- **Repos:** `autopen`, `bookengine`, `pureclaw`
- **Finding:** Git commit trailers contain `Co-Authored-By: HAL <hal@puretensors.com>`. The primary domain used everywhere else is `puretensor.ai` (without "s"). `puretensors.com` may be a typo or an unregistered domain.
- **Risk:** If `puretensors.com` is not owned by PureTensor, it could be registered by a third party to conduct phishing or impersonation.
- **Recommendation:** Confirm ownership of `puretensors.com`. If it is not a PureTensor-controlled domain, register it or correct the email address in future commits.

---

## Confirmed Clean — No Findings in Any Repo

The following patterns were searched across all repos (current files + full git history) and returned **no real credential values**:

- Anthropic API keys (`sk-ant-…`)
- OpenAI API keys (`sk-proj-…`)
- xAI API keys (`xai-…`)
- GitHub PATs (`ghp_…` / `ghs_…`)
- AWS Access Key IDs (`AKIA[A-Z0-9]{16}`)
- AWS Secret Access Keys
- Telegram bot tokens (format `[0-9]{8,12}:[a-zA-Z0-9_-]{35,}`)
- PEM-encoded private keys (`BEGIN RSA`, `BEGIN EC`, `BEGIN OPENSSH`)
- Database connection strings with embedded credentials
- Google OAuth tokens
- Committed `.env` files with real secret values
- `k8s/secrets.yaml` — all `REPLACE_ME` placeholders (correct)
- Docker configs with embedded credentials
- CI/CD configs with hardcoded secrets

---

## Summary

### Findings by Severity

| Severity | Count |
|---|---|
| CRITICAL | 2 |
| WARNING | 10 |
| INFO | 12 |
| **Total** | **24** |

### Findings by Repo

| Repo | Critical | Warning | Info |
|------|----------|---------|------|
| pureclaw | 2 | 6 | 5 |
| echo-voicememo | 0 | 1 | 3 |
| kalima | 0 | 3 | 1 |
| autopen | 0 | 0 | 1 |
| bookengine | 0 | 0 | 1 |
| cross-repo | 0 | 0 | 1 |

**Repos scanned:** 5  
**Repos inaccessible (private):** 5 (`varangian-ai`, `voice-kb`, `ecommerce-agent`, `kalima-android`, `arabic-qa`)

---

## Priority Actions

1. **[IMMEDIATE]** Rotate the BMC/IPMI password on **all** fleet nodes (tensor-core, fox-n0, fox-n1, arx1–arx4, mon1–mon3). Commit `4cfa8e9` is permanently in git history — the password is compromised.
2. **[IMMEDIATE]** Assess whether `puretensor/pureclaw` was ever made public on GitHub. If so, assume the fleet IP map and BMC credential have been indexed by secret scanners.
3. **[SHORT TERM]** Add `deploy/*.env` to `.gitignore`. Replace `deploy/claw-*.env` with `.env.example` templates.
4. **[SHORT TERM]** Remove remaining live Tailscale IPs from current HEAD: `tools/nexus-terminal.py:48`, `prompts/claw_ops_prompt.md:46`, and lines 68–69 of this document.
5. **[SHORT TERM]** Fix kalima WebSocket to reject `api_key` query parameter; accept only via header (`X-API-Key`).
6. **[MEDIUM TERM]** Move `prompts/pureclaw_context.md` infrastructure topology out of the public repo or inject at runtime via deployment secrets.
7. **[MEDIUM TERM]** Move operational email addresses out of K8s ConfigMaps into Secrets/env vars.
8. **[MEDIUM TERM]** Template echo-voicememo systemd service files to remove hardcoded username and absolute paths.
9. **[MEDIUM TERM]** Consider rotating Tailscale node keys for nodes whose Tailscale IPs are in git history (tensor-core `100.121.42.54`, fox-n0 `100.69.225.18`, fox-n1 `100.103.248.9`).
10. **[LOW PRIORITY]** Verify ownership of `puretensors.com` (appears in git commit trailers across multiple repos).
11. **[LOW PRIORITY]** Delete `=TLSv1.2` junk file from pureclaw root.
12. **[LOW PRIORITY]** Audit the 5 private repos when a session with appropriate credentials is available.
