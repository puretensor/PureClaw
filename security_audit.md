# PureTensor Public Repository Security Audit

**Date:** 2026-04-05 (supersedes 2026-04-01 audit)
**Auditor:** Claude Code (automated) — full repo + full git history scan
**Scope:** All accessible PureTensor public repositories

---

## Repository Access Summary

| Repository | Status | Commits Scanned |
|---|---|---|
| `puretensor/pureclaw` | **Scanned** | 98 unique commits (full history) |
| `puretensor/autopen` | **Network inaccessible** | — |
| `puretensor/bookengine` | **Network inaccessible** | — |
| `puretensor/echo-voicememo` | **Network inaccessible** | — |
| `puretensor/kalima` | **Network inaccessible** | — |
| `puretensor/varangian-ai` | **Network inaccessible** | — |
| `puretensor/voice-kb` | **Network inaccessible** | — |
| `puretensor/ecommerce-agent` | **Network inaccessible** | — |
| `puretensor/kalima-android` | **Network inaccessible** | — |
| `puretensor/arabic-qa` | **Network inaccessible** | — |

**Note:** The audit environment has no outbound network access. Only the pre-cloned `pureclaw` repository could be scanned. Findings from external repos documented in the 2026-04-01 audit (echo-voicememo, kalima, autopen, bookengine) are retained below from that prior report; they have not been re-verified in this session.

---

## PureClaw — Findings

### CRITICAL

---

**[CRITICAL-1] BMC/IPMI Fleet Password in Git History**

- **File:** `prompts/claw_infra_prompt.md`
- **Commit added:** `4cfa8e9` / `0db2f1d` (feat: Add Claw, 2026-03-21)
- **Commit redacted:** `6402aad` / `dbd5b23` (security: redact credentials, 2026-03-29) — replaced with `<BMC_PASSWORD>` placeholder
- **Redacted value:** `cons****-craz****-curl` (3-word passphrase)
- **Current HEAD:** Placeholder only (`<BMC_PASSWORD>`) — but the real value is **permanently embedded in git history**
- **Impact:** This password grants BMC/IPMI access to the entire hardware fleet: tensor-core, fox-n0, fox-n1, arx1–arx4, mon1–mon3. BMC/IPMI enables remote power control (on/off/reset), serial console, and potential firmware-level access to every machine.
- **Status:** If this repository was ever public on GitHub, the value has been indexed by secret scanners. Assume fully compromised.
- **Recommendation:** Rotate this password immediately on all nodes. A git history rewrite (BFG Repo Cleaner / `git filter-repo` + force-push) is also warranted if no public forks exist, but rotation takes absolute priority.

---

**[CRITICAL-2] Full Fleet IP Mapping in Git History**

- **File:** `prompts/claw_infra_prompt.md`
- **Commit:** `4cfa8e9` / `0db2f1d` (present before `6402aad` redaction; placeholders now in HEAD)
- **Values permanently in history:**

| Node | LAN IP | Tailscale IP | Fabric IP |
|------|--------|--------------|-----------|
| tensor-core | `192.168.4.217` | `100.121.42.54` | `10.200.0.3` |
| fox-n0 | `192.168.4.184` | `100.69.225.18` | `10.200.0.1` |
| fox-n1 | `192.168.4.50` | `100.103.248.9` | `10.200.0.2` |

- **Impact:** Combined with CRITICAL-1, this constitutes a complete remote attack kit: known credentials plus exact targets. Even independently, it maps the entire private network topology across three address spaces (LAN, Tailscale, 200G fabric).
- **Recommendation:** Consider rotating Tailscale node keys for the exposed nodes. Review Tailscale ACL rules to enforce minimum-necessary access.

---

**[CRITICAL-3] PostgreSQL Database Password in Git History** *(NEW — post-2026-04-01 audit)*

- **Files:** `k8s/configmap.yaml`, `memory_rag.py`
- **Commits added:** `7016c24` / `760158a` (feat: PureClaw Memory Architecture Upgrade, 2026-04-03)
- **Commits redacted:** `87f6d04` / `f885cfc` (fix: audit remediation — security hardening, 2026-04-04)
- **Values permanently in history:**
  - `postgresql://postgres:PT-db-****-secure@postgres-postgresql.databases.svc.cluster.local:5432/nexus_memory` (in `k8s/configmap.yaml` as `MEMORY_PG_URL`)
  - `postgresql://postgres:PT-db-****-secure@100.103.248.9:30432/nexus_memory` (external NodePort URL — also exposes fox-n1 Tailscale IP + NodePort)
  - `postgresql://vantage:vantage@postgres-postgresql.databases.svc:5432/nexus_memory` (in `memory_rag.py` default fallback — likely development credentials for the `vantage` PostgreSQL role)
- **Impact:**
  - `PT-db-****-secure`: Active (or recently active) PostgreSQL superuser password. Grants full read/write access to `nexus_memory` database containing agent memory, conversation history, and any injected context.
  - `vantage:vantage`: Default/dev credential for the `vantage` role. If not rotated or role not dropped, provides database access.
  - The external URL also confirms the NodePort `30432` on fox-n1 is (or was) publicly reachable via Tailscale.
- **Status:** These commits were introduced AFTER the 2026-04-01 audit and then redacted in the following day's remediation, but they remain permanently in git history.
- **Recommendation:** Rotate the PostgreSQL `postgres` superuser password immediately. Verify the `vantage` role has been dropped or its password changed. Confirm `30432` NodePort is not reachable from outside the Tailscale network.

---

### WARNING

---

**[WARNING-1] All Internal Service Tailscale IPs and Ports in Git History**

- **Files:** `k8s/configmap.yaml` (history), `deploy/claw-*.env` (history)
- **Commits:** Pre-`6402aad` versions; also present in `7016c24` (PostgreSQL external URL — see CRITICAL-3)
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
| PostgreSQL NodePort (fox-n1) | `100.103.248.9` | 30432 |
| Fabric mesh (fox-n0) | `10.200.0.1` | 9880 |
| Mesh peer (mon2) | `100.80.213.1` | 9880 |

- **Also in current HEAD:** `tools/nexus-terminal.py` (`DEFAULT_HOST = "localhost"` — was previously `100.103.248.9`, redacted), `security_audit.md` table above (this document).
- **Recommendation:** Use Tailscale MagicDNS hostnames instead of IPs in all config. Ensure this security_audit.md is not left in the public repo with the IP table above — or redact this table.

---

**[WARNING-2] Telegram Operator User ID in Git History**

- **Files:** `deploy/claw-infra.env`, `deploy/claw-ops.env`, `deploy/claw-sentinel.env` (all in history before `6402aad`)
- **Variable:** `AUTHORIZED_USER_ID`
- **Redacted value:** `2227****` (Telegram numeric user ID)
- **Current HEAD:** Replaced with `<TELEGRAM_CHAT_ID>` placeholder
- **Impact:** Exposes the Telegram user ID of the bot's authorized operator. Enables identity confirmation and targeted social engineering.

---

**[WARNING-3] Deploy `.env` Files with Real Infrastructure Config Committed**

- **Files:** `deploy/claw-infra.env`, `deploy/claw-ops.env`, `deploy/claw-sentinel.env`
- **Finding:** These are real deployment environment files containing actual service architecture, vLLM model names, service ports, filesystem paths (including username `puretensorai`), and `SET_ME` placeholders for secrets. They are not covered by `.gitignore` (`deploy/*.env` pattern is missing — only root `.env` and `failover/.env*` are excluded).
- **Recommendation:** Add `deploy/*.env` to `.gitignore`. Replace tracked files with `.env.example` templates.

---

**[WARNING-4] Comprehensive Internal Infrastructure Topology in Public System Prompt**

- **File:** `prompts/pureclaw_context.md`
- **Finding:** The context document committed to the public repo contains: hardware specs for every node (CPU, RAM, NVMe, GPU model), tier layout (Tier 0–3 + GCP), node roles and services, storage layout (RAID configs, ZFS pools, Ceph OSD counts/sizes), SSH command patterns, GCP instance types, hosted domains (`bretalon.com`, `nesdia.com`), IMAP account handles (`hh`, `alan`, `yahoo`), and the company address.
- **Impact:** Provides a detailed attack surface map. An adversary knows exactly what hardware to target, how storage is organized, and what services are hosted where.
- **Recommendation:** Replace with a generic/redacted template in the repo; inject real infrastructure details at runtime via deployment secrets or memory injection.

---

**[WARNING-5] Company Registered Address Hardcoded in Production Code**

- **Files:** `prompts/pureclaw_context.md:174`, `observers/doc_compiler.py:241-242`, `observers/daily_report.py:226-227`, `observers/weekly_report.py:278-279`
- **Finding:** `"131 Continental Dr, Suite 305 / Newark, DE 19713, US"` hardcoded as a fixed literal in multiple source files.
- **Recommendation:** Move to a configurable env var (e.g. `COMPANY_ADDRESS`).

---

**[WARNING-6] Agent Email Addresses in K8s ConfigMap**

- **Files:** `k8s/configmap.yaml:61`, `k8s/hal-mail-configmap.yaml:39`
- **Finding:** `AGENT_EMAIL: "hal@puretensors.com,hal@puretensor.com,hal@puretensor.ai"` — three real operational email addresses committed in a public ConfigMap.
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

- **Files:** `observers/.state/health_2026-03-18.json` through `health_2026-04-01.json`
- **Finding:** Runtime state files are committed. Contents are benign (run counts, timestamps). `.gitignore` lists `.state/` — these files appear to have been committed explicitly.
- **Recommendation:** `git rm --cached observers/.state/` and ensure the `.gitignore` entry covers them going forward.

---

**[INFO-3] `=TLSv1.2` Junk File at Repo Root**

- **File:** `=TLSv1.2` (empty file, 0 bytes)
- **Finding:** Likely a copy-paste error from a `curl` command (`curl --tlsv1.2 ...` typed with `=` prefix). No sensitive data.
- **Recommendation:** Delete this file (`git rm "=TLSv1.2"`).

---

**[INFO-4] Filesystem Paths Expose Deployment Username**

- **Files:** `deploy/claw-infra.env:25`, `config.py:20`, `observers/git_auto_sync.py:26`, `observers/github_activity.py:38`, `backends/ollama.py:56`, `observers/memory_sync.py:30`
- **Finding:** Absolute paths `/home/puretensorai/…` reveal the production OS username. Moot once `deploy/*.env` files are removed from git (see WARNING-3/INFO-1), but the username also appears in tracked Python source files.
- **Recommendation:** Replace hardcoded `/home/puretensorai` with env var or `~` expansion where it appears outside env files.

---

**[INFO-5] Security Policy Files Disable Private Range Blocking for Claw Agents**

- **Files:** `security/claw_infra_policy.yaml:28`, `security/claw_ops_policy.yaml:28`, `security/claw_sentinel_policy.yaml:29`
- **Finding:** `block_private_ranges: false` for all three Claw agents. Intentional (they need LAN/Tailscale access) but leaves these agents without SSRF protection against their local networks.
- **Recommendation:** Add an explicit comment in each policy file documenting why this override is intentional and which network ranges are expected.

---

## Previous Audit Findings — External Repos (2026-04-01, Not Re-verified)

*The following findings were documented in the 2026-04-01 audit for repos that could not be accessed in this session.*

### echo-voicememo

**[WARNING-7] Production OS Username and Paths in Committed Systemd Service Files**
- Files: `xtts-spanish.service` (lines 7, 10, 13, 14), `echo-voicememo-api.service` (lines 8–11)
- `User=puretensorai`, paths `/home/puretensorai/…`, `/opt/xtts-v2/finetune_output/…`
- Recommendation: Template service files; remove hardcoded username and absolute paths.

**[INFO-6] Hardcoded Absolute Model Path in Source**
- File: `xtts_spanish_server.py:37`
- `BASE_DIR = "/opt/xtts-v2/finetune_output/run/training/XTTS_v2.0_original_model_files"`
- Recommendation: Use an environment variable for model base path.

**[INFO-7] No Authentication on API**
- File: `server.py`
- `X-Device-Id` used for scoping only; no authentication layer. Acceptable for local-only but dangerous if network-exposed.

**[INFO-8] Unauthenticated Google Translate Usage**
- Files: `config.py:20`, `pipeline.py:147-150`
- Uses `deep_translator.GoogleTranslator` (scraping-based) — no formal data processing agreement.

### kalima

**[WARNING-8] API Key Accepted as URL Query Parameter**
- Files: `src/kalima/api/websocket.py:106`, `README.md:82,104`, `docs/android-ime-research.md:578`
- `api_key` accepted as WebSocket URL query parameter — leaks credentials into server logs, browser history, proxy logs, referrer headers.
- Recommendation: Remove query-parameter key acceptance; enforce header-only (`X-API-Key`).

**[WARNING-9] Silent Auth Bypass When `KALIMA_API_KEYS` Is Unset**
- Files: `src/kalima/main.py:35-36`, `src/kalima/config.py:23`
- If `KALIMA_API_KEYS` is absent, all requests are accepted without authentication.
- Recommendation: Explicit startup check refusing to start (or logging a prominent warning) when unset in non-dev mode.

**[WARNING-10] Production Username and Full Path in Documentation**
- File: `docs/research/arabic-keyboard-dictation-ux.md:380`
- Full path `/home/puretensorai/kalima/src/kalima/asr/deepgram.py` — reveals OS username.
- Recommendation: Replace with a generic path.

**[INFO-9] Android Emulator IP in Documentation**
- File: `docs/android-ime-research.md:1019`
- `ws://10.0.2.2:8000` — standard Android emulator loopback alias for the host machine. Not a real internal host.

### autopen

**No credential findings.** All 6 commits scanned. No API keys, tokens, passwords, private keys, or internal IPs.

**[INFO-10] `.gitignore` Missing Some Credential File Patterns**
- Missing: `*.p12`, `*.pfx`, `credentials*`, `secrets/`, `*service_account*.json`

### bookengine

**No credential findings.** All 15 commits scanned. Single Python file using only local Ollama.

**[INFO-11] `.gitignore` Missing Some Credential File Patterns**
- Missing: `*.p12`, `*.pfx`

---

## Cross-Repo

**[INFO-12] `puretensors.com` (with "s") vs `puretensor.ai` in Git Commit Metadata**

- **Repos:** `autopen`, `bookengine`, `pureclaw`
- **Finding:** Git commit trailers contain `Co-Authored-By: HAL <hal@puretensors.com>`. The primary domain used everywhere else is `puretensor.ai` (without "s"). If `puretensors.com` is not owned by PureTensor, it could be registered by a third party for phishing or impersonation.
- **Recommendation:** Confirm ownership of `puretensors.com`. If not PureTensor-controlled, register it or correct future commit authorship.

---

## Confirmed Clean — No Findings in Current Scan

The following patterns were searched across all files in current HEAD and full git history of `pureclaw` and returned **no real credential values**:

- Anthropic API keys (`sk-ant-…`)
- OpenAI API keys (`sk-proj-…`)
- xAI API keys (`xai-…`)
- GitHub PATs (`ghp_…` / `ghs_…`)
- AWS Access Key IDs (`AKIA[A-Z0-9]{16}`)
- AWS Secret Access Keys
- Telegram bot tokens (real format — `123456789:ABCdef…` in README is clearly a placeholder)
- PEM-encoded private keys (`BEGIN RSA`, `BEGIN EC`, `BEGIN OPENSSH`)
- Google API keys (`AIza…`)
- JWT tokens
- Committed `.env` files with real secret values (root `.env` excluded by `.gitignore`)
- `k8s/secrets.yaml` — all `REPLACE_ME` placeholders (correct)
- Docker configs with embedded credentials
- CI/CD configs with hardcoded secrets (`.github/workflows/test.yml` uses no secrets)
- Phone numbers
- OAuth refresh/access tokens

---

## Summary

### Findings by Severity

| Severity | Count |
|---|---|
| CRITICAL | 3 |
| WARNING | 10 |
| INFO | 12 |
| **Total** | **25** |

### Findings by Repo

| Repo | Critical | Warning | Info |
|------|----------|---------|------|
| pureclaw | 3 | 6 | 5 |
| echo-voicememo | 0 | 1 | 3 |
| kalima | 0 | 3 | 1 |
| autopen | 0 | 0 | 1 |
| bookengine | 0 | 0 | 1 |
| cross-repo | 0 | 0 | 1 |

**Repos fully scanned this session:** 1 (`pureclaw`)
**Repos with prior findings carried forward:** 4 (`echo-voicememo`, `kalima`, `autopen`, `bookengine`)
**Repos inaccessible (network):** 9 (all external repos — network unavailable in audit environment)

---

## Priority Actions

1. **[IMMEDIATE]** Rotate the BMC/IPMI password on **all** fleet nodes (tensor-core, fox-n0, fox-n1, arx1–arx4, mon1–mon3). Commit `4cfa8e9` is permanently in git history.
2. **[IMMEDIATE]** Rotate the PostgreSQL `postgres` superuser password for the `nexus_memory` database. Commits `7016c24`/`760158a` are permanently in git history with the plaintext password `PT-db-****-secure`. Also verify the `vantage` role has been dropped or its password changed.
3. **[IMMEDIATE]** Assess whether `puretensor/pureclaw` was ever made public on GitHub. If so, assume the fleet IP map, BMC credential, and PostgreSQL password have been indexed by secret scanners (GitHub, GitGuardian, TruffleHog, etc.).
4. **[SHORT TERM]** Add `deploy/*.env` to `.gitignore`. Replace `deploy/claw-*.env` with `.env.example` templates.
5. **[SHORT TERM]** Fix kalima WebSocket to reject `api_key` query parameter; accept only via header (`X-API-Key`).
6. **[MEDIUM TERM]** Move `prompts/pureclaw_context.md` infrastructure topology out of the public repo or inject at runtime via deployment secrets/memory.
7. **[MEDIUM TERM]** Move operational email addresses out of K8s ConfigMaps into Secrets/env vars.
8. **[MEDIUM TERM]** Template echo-voicememo systemd service files to remove hardcoded username and absolute paths.
9. **[MEDIUM TERM]** Consider rotating Tailscale node keys for nodes whose Tailscale IPs are in git history (tensor-core `100.121.42.54`, fox-n0 `100.69.225.18`, fox-n1 `100.103.248.9`, mon2 `100.80.213.1`, mon1 `100.92.245.5`, SearXNG host `100.105.43.27`).
10. **[LOW PRIORITY]** Verify ownership of `puretensors.com` (appears in git commit trailers).
11. **[LOW PRIORITY]** Delete `=TLSv1.2` junk file from pureclaw root.
12. **[LOW PRIORITY]** Re-run this audit against the 9 external repos when a network-connected session is available (`bookengine`, `varangian-ai`, `voice-kb`, `ecommerce-agent`, `kalima`, `kalima-android`, `echo-voicememo`, `arabic-qa`, `autopen`).
