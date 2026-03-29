# PureTensor Public Repository Security Audit

**Date:** 2026-03-29
**Auditor:** Claude Code (automated)
**Scope:** All accessible PureTensor public repositories

---

## Repository Access Summary

| Repository | Status | Method |
|---|---|---|
| `puretensor/pureclaw` | **Scanned** | Local clone + git history |
| `puretensor/bookengine` | **Scanned** | git clone (public) |
| `puretensor/kalima` | **Scanned** | git clone (public) |
| `puretensor/echo-voicememo` | **Scanned** | git clone (public) |
| `puretensor/autopen` | **Scanned** | git clone (public) |
| `puretensor/varangian-ai` | **Private / Inaccessible** | Clones require auth |
| `puretensor/voice-kb` | **Private / Inaccessible** | Clones require auth |
| `puretensor/ecommerce-agent` | **Private / Inaccessible** | Clones require auth |
| `puretensor/kalima-android` | **Private / Inaccessible** | Clones require auth |
| `puretensor/arabic-qa` | **Private / Inaccessible** | Clones require auth |

---

## PureClaw — Findings

### CRITICAL

---

**[CRITICAL-1] BMC/IPMI Fleet Password in Plaintext**

- **File:** `prompts/claw_infra_prompt.md`, line 57
- **Also in git history:** commit `4cfa8e9` (feat: Add Claw, 2026-03-21)
- **Finding:** The shared BMC/IPMI password for the entire server fleet is committed in plaintext in a public repository.
- **Redacted value:** `cons****-craz****-curl` (3-word passphrase)
- **Impact:** Anyone with this password can authenticate to the IPMI/BMC interface on tensor-core, fox-n0, fox-n1, arx1–4, and mon1–3 — giving remote power control (on/off/reset), serial console access, and potential firmware-level access to all fleet hardware.
- **Recommendation:** **Rotate this password immediately on all nodes.** Remove the line from the prompt file and load it via an environment variable or Vault. Use per-node BMC passwords going forward.

---

### WARNING

---

**[WARNING-1] Full Fleet Internal IP Mapping Exposed**

- **File:** `prompts/claw_infra_prompt.md`, lines 51–53
- **Finding:** Every IP address for every primary node is listed in three columns (LAN, Tailscale, 200G fabric):
  - tensor-core: `<LAN_TENSOR_CORE>` / `<TS_TENSOR_CORE>` / `<FABRIC_TENSOR_CORE>`
  - fox-n0: `<LAN_FOX_N0>` / `<TS_FOX_N0>` / `<FABRIC_FOX_N0>`
  - fox-n1: `<LAN_FOX_N1>` / `<TS_FOX_N1>` / `<FABRIC_FOX_N1>`
- **Impact:** Confirms exact node locations on the LAN and 200G fabric. Combined with the BMC password (CRITICAL-1), this enables targeted attacks. Even without CRITICAL-1, it exposes network topology.
- **Recommendation:** Replace with hostname references only. Remove LAN IPs (192.168.x.x) and fabric IPs (10.200.0.x) from the prompt file — the agent can resolve hostnames at runtime.

---

**[WARNING-2] Tailscale IPs for All Internal Services Hardcoded**

- **File:** `k8s/configmap.yaml`, lines 9, 22, 45–48, 52–53, 56–57
- **Finding:** Tailscale IPs for every internal service are hardcoded:
  - tensor-core (vLLM, Vision, Whisper, TTS, Ollama, SSH): `<TS_TENSOR_CORE>`
  - mon2 (Prometheus, Alertmanager): `<TS_MON2>`
  - mon1 (Gitea): `<TS_MON1>`
  - SearXNG instance: `<TS_IP_REDACTED>`
- **Also in:** `deploy/claw-infra.env:5`, `deploy/claw-ops.env:5`, `deploy/claw-sentinel.env:5`
- **Also in:** `mesh/registry.py:47` (example comment), `tools/nexus-terminal.py:48` (`DEFAULT_HOST = "100.103.248.9"`)
- **Also in:** `prompts/claw_ops_prompt.md:46` (Ceph mgr REST URL with `100.80.213.1`)
- **Impact:** Reveals the Tailscale network layout, service ports, and which node hosts which service. While Tailscale itself requires auth, this aids reconnaissance and associates hostnames to IP addresses permanently in git history.
- **Recommendation:** Use Tailscale MagicDNS hostnames instead (not sensitive). For the deploy `.env` files specifically, see INFO-1.

---

**[WARNING-3] Deploy `.env` Files with Real Infrastructure Config Committed**

- **Files:** `deploy/claw-infra.env`, `deploy/claw-ops.env`, `deploy/claw-sentinel.env`
- **First committed:** `4cfa8e9` (2026-03-21)
- **Finding:** These are real deployment environment files — not templates — containing actual Tailscale IPs, fabric IPs, service ports, and filesystem paths for the production deployment. Although bot tokens are set to `SET_ME`, the network topology data is real.
- **Additionally:** `AUTHORIZED_USER_ID=<TELEGRAM_CHAT_ID>` appears in all three files. This is a real Telegram user ID permanently committed to git history (present in history even after any edits).
- **Recommendation:** Add `deploy/*.env` to `.gitignore`. The `.gitignore` currently excludes `.env` and `.env.*` but not `deploy/*.env`. Replace with `deploy/*.env.example` template files with placeholder values.

---

**[WARNING-4] Company Registered Address Hardcoded in Production Code**

- **Files:**
  - `prompts/pureclaw_context.md`, line 174
  - `observers/doc_compiler.py`, lines 241–242
  - `observers/daily_report.py`, lines 226–227
  - `observers/weekly_report.py`, lines 278–279
- **Finding:** The string `"131 Continental Dr, Suite 305 / Newark, DE 19713, US"` is hardcoded as a fixed literal in the document generation code and system context prompt.
- **Impact:** Exposes the registered business address directly in source code of a public repo. Minor risk on its own (address is on public filings), but unnecessary and makes the address hard to update.
- **Recommendation:** Move to a configurable env var (e.g. `COMPANY_ADDRESS`) with a blank or generic default.

---

**[WARNING-5] Comprehensive Internal Infrastructure Topology in System Prompt**

- **File:** `prompts/pureclaw_context.md` (entire file)
- **Finding:** The context document committed to the public repo contains detailed infrastructure topology:
  - Hardware specs for every node (CPU model, RAM, NVMe count, GPU model/count)
  - Tier layout (Tier 0 through Tier 3 + GCP)
  - Node roles and services hosted on each
  - Storage layout (RAID configs, ZFS pools, Ceph OSD counts and sizes)
  - SSH patterns and command examples
  - GCP instance types and hosted domains (bretalon.com, nesdia.com)
  - IMAP account handles (`hh`, `alan`, `yahoo`) — not passwords, but account names
- **Impact:** Provides a detailed attack surface map. An adversary knows exactly which node to target for which service, what hardware is present, and how storage is organized.
- **Recommendation:** Keep a generic/redacted version in the public repo and inject real infrastructure details at runtime via the deployment secrets/ConfigMap mechanism.

---

**[WARNING-6] Agent Email Addresses in K8s ConfigMap**

- **File:** `k8s/configmap.yaml`, line 61
- **Finding:** `AGENT_EMAIL: "hal@puretensors.com,hal@puretensor.com,hal@puretensor.ai"` — three real operational email addresses committed.
- **Also found in git history** in earlier versions of configmap.yaml.
- **Impact:** Links the bot identity to real email accounts. Exposes operational email addresses for the org; could facilitate targeted phishing or enumeration.
- **Recommendation:** Move to the secrets mechanism or reference a placeholder.

---

### INFO

---

**[INFO-1] `.gitignore` Missing Entry for `deploy/*.env`**

- **File:** `.gitignore`
- **Finding:** The `.gitignore` correctly excludes `.env` and `.env.*` at the root, and `failover/.env*`. However, `deploy/claw-infra.env`, `deploy/claw-ops.env`, and `deploy/claw-sentinel.env` fall outside these patterns and are tracked by git.
- **Recommendation:** Add `deploy/*.env` (and optionally `deploy/*.conf`) to `.gitignore`.

---

**[INFO-2] `observers/.state/` Directory Committed**

- **Files:** `observers/.state/health_2026-03-18.json` through recent dates
- **Finding:** Runtime state files are committed. Contents appear benign (observer run counts, timestamps). The `.gitignore` already lists `.state/` — these files were committed before that entry was added, or committed explicitly.
- **Recommendation:** Remove these files from tracking (`git rm --cached observers/.state/`) and ensure the `.gitignore` entry covers them going forward.

---

**[INFO-3] `=TLSv1.2` File at Repo Root**

- **File:** `=TLSv1.2` (root of repo)
- **Finding:** An empty file named `=TLSv1.2` exists at the repo root. Likely the result of a copy-paste error. No sensitive data.
- **Recommendation:** Delete this file.

---

**[INFO-4] Filesystem Paths Expose Deployment Layout**

- **Files:** `deploy/claw-infra.env:25`, `deploy/claw-ops.env:25`, `deploy/claw-sentinel.env:19`
- **Finding:** Absolute filesystem paths are committed: `/home/puretensorai/nexus/...`, `/opt/claw-ops/...`, `/opt/claw-sentinel/...`. These reveal the deployment username (`puretensorai`) and directory structure.
- **Impact:** Low. Useful for an attacker who already has access, not exploitable on its own.
- **Recommendation:** Moot once `deploy/*.env` files are removed from git (see WARNING-3 / INFO-1).

---

**[INFO-5] Security Policy Files Disable Private Range Blocking for Claw Agents**

- **Files:** `security/claw_infra_policy.yaml:28`, `security/claw_ops_policy.yaml:28`, `security/claw_sentinel_policy.yaml:29`
- **Finding:** `block_private_ranges: false` for all three Claw agents. This is intentional (they need LAN/Tailscale access) but means these agents are not protected by SSRF controls.
- **Recommendation:** Add a comment in each policy file explicitly documenting why this is set to false.

---

## bookengine — Findings

**No findings.** Single Python file (`bookengine.py`), README, and LICENSE. No credentials, no IP addresses, no sensitive configuration. The file uses `~/.bookengine-data/` for local storage and expects Ollama on localhost. Clean.

---

## kalima — Findings

**No findings.** Review covered all source files, `tests/`, `.env.example`, `docker-compose.yml`, `.github/` CI config, and full git history.

- `.env.example` is correctly structured with empty API key placeholders.
- Test fixtures in `tests/conftest.py` use clearly fake values (`"fake-soniox-key"`, `"fake-dg-key"`, `"fake-mistral-key"`).
- `docker-compose.yml` uses `env_file: .env` — no hardcoded credentials.
- `docs/android-ime-research.md` references `10.0.2.2` — this is the Android emulator's standard alias for the host machine's localhost, not a real internal IP.
- CI workflow (`.github/workflows/`) uses no hardcoded secrets; API keys are injected via GitHub Actions secrets.

---

## echo-voicememo — Findings

**No findings.** Full repo and git history scanned.

- `config.py` uses only `localhost` URLs.
- No API keys, tokens, or passwords in any file.
- `.gitignore` correctly excludes `.env`, audio files, and database files.
- Service files (`echo-voicememo-api.service`, `xtts-spanish.service`) contain only unit configuration — no credentials.

---

## autopen — Findings

**No findings.** Full repo and git history scanned.

- No API keys or credentials in any file.
- `requirements.txt` and `pyproject.toml` contain only package dependencies.
- `.gitignore` correctly excludes `.env`, secrets, and key files.
- `examples/` directory contains only PDF output samples.

---

## No Issues Found — Confirmed Clean Patterns (PureClaw)

The following patterns were searched and returned **no findings** in PureClaw:

- API keys matching `sk-[a-z0-9]{20,}` (Anthropic), `xai-[a-z0-9]{20,}` (xAI), `ghp_[a-z0-9]{20,}` (GitHub PAT)
- AWS credentials (`AKIA[A-Z0-9]{16}`, `aws_secret_access_key`)
- Telegram bot token format (`[0-9]{8,12}:[a-zA-Z0-9_-]{35,}`) — only match in history was a clearly fake example
- Hardcoded database connection strings with credentials
- Private keys or certificates (`.pem`, `.key`, PEM headers)
- OAuth tokens or refresh tokens committed directly
- `.env` file committed at the root (`.env.example` exists and is clean)
- `k8s/secrets.yaml` — all values are `REPLACE_ME` placeholders (correct)
- `k8s/local/secrets.yaml` — contains only `kubectl create` instructions, no real values

---

## Summary

| Severity | Count | Key Issues |
|---|---|---|
| CRITICAL | 1 | Fleet-wide BMC/IPMI password in plaintext (PureClaw) |
| WARNING | 6 | Internal IPs, deploy env files, company address, infra topology, email addresses, Tailscale IPs (all PureClaw) |
| INFO | 5 | .gitignore gaps, stale state files, junk file, path exposure, SSRF policy intent (all PureClaw) |

**Repos scanned:** 5 (`pureclaw`, `bookengine`, `kalima`, `echo-voicememo`, `autopen`)
**Repos inaccessible (private):** 5 (`varangian-ai`, `voice-kb`, `ecommerce-agent`, `kalima-android`, `arabic-qa`)
**Total findings:** 12 (1 CRITICAL, 6 WARNING, 5 INFO) — all in `puretensor/pureclaw`
**Other repos:** All clean, no issues found

### Priority Actions

1. **Immediate:** Rotate the BMC/IPMI password on all fleet nodes. It is permanently in git history and cannot be expunged without a full history rewrite. Assume it is compromised.
2. **Short term:** Add `deploy/*.env` to `.gitignore` and replace the three deploy env files with `.example` templates containing placeholder values only.
3. **Short term:** Remove specific IP addresses from `prompts/claw_infra_prompt.md` — use Tailscale MagicDNS hostnames instead.
4. **Medium term:** Consider whether `prompts/pureclaw_context.md` level of infrastructure detail is appropriate for a public repo. Move sensitive topology to a deployment-injected context file.
5. **Medium term:** Audit the 5 private repos when a session with appropriate credentials is available.
