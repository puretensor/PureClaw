# PureTensor Public Repository Security Audit

**Date:** 2026-03-27
**Auditor:** Claude Code (automated)
**Scope:** All accessible PureTensor public repositories

---

## Repository Access Summary

| Repository | Status | Method |
|---|---|---|
| `puretensor/pureclaw` | **Scanned** | Local clone + git history |
| `puretensor/bookengine` | **Inaccessible** | No network access to GitHub (no git CLI auth) |
| `puretensor/varangian-ai` | **Inaccessible** | No network access to GitHub |
| `puretensor/voice-kb` | **Inaccessible** | No network access to GitHub |
| `puretensor/ecommerce-agent` | **Inaccessible** | No network access to GitHub |
| `puretensor/kalima` | **Inaccessible** | No network access to GitHub |
| `puretensor/kalima-android` | **Inaccessible** | No network access to GitHub |
| `puretensor/echo-voicememo` | **Inaccessible** | No network access to GitHub |
| `puretensor/arabic-qa` | **Inaccessible** | No network access to GitHub |
| `puretensor/autopen` | **Inaccessible** | No network access to GitHub |

**Note:** The GitHub MCP tools available to this session are restricted to `puretensor/pureclaw` only. Direct `git clone` over HTTPS was blocked by the sandbox environment (no external network). The 9 additional repos could not be audited. A re-run with network access or wider MCP scope is recommended.

---

## PureClaw — Findings

### CRITICAL

---

**[CRITICAL-1] BMC/IPMI Fleet Password in Plaintext**

- **File:** `prompts/claw_infra_prompt.md`, line 56
- **Also in git history:** commit `4cfa8e9` (feat: Add Claw, 2026-03-21)
- **Finding:** The shared BMC/IPMI password for the entire server fleet is committed in plaintext in a public repository.
- **Redacted value:** `cons***-craz***-curl` (12 chars, 3-word passphrase)
- **Impact:** Anyone with this password can authenticate to the IPMI/BMC interface on tensor-core, fox-n0, fox-n1, arx1–4, and mon1–3 — giving remote power control (on/off/reset), serial console access, and potential firmware-level access to all fleet hardware.
- **Recommendation:** **Rotate this password immediately on all nodes.** Remove the line from the prompt file and load it via an environment variable or Vault. Use per-node BMC passwords where possible.

---

### WARNING

---

**[WARNING-1] Full Fleet Internal IP Mapping Exposed**

- **File:** `prompts/claw_infra_prompt.md`, lines 51–53
- **Finding:** Every IP address for every primary node is listed in three columns (LAN, Tailscale, 200G fabric):
  - tensor-core: `192.168.4.217` / `100.121.42.54` / `10.200.0.3`
  - fox-n0: `192.168.4.184` / `100.69.225.18` / `10.200.0.1`
  - fox-n1: `192.168.4.50` / `100.103.248.9` / `10.200.0.2`
- **Impact:** Confirms exact node locations on the LAN and 200G fabric. Combined with the BMC password (CRITICAL-1), this enables targeted attacks. Even without CRITICAL-1, it exposes network topology.
- **Recommendation:** Replace with hostname references only. Remove LAN IPs (192.168.x.x) and fabric IPs (10.200.0.x) from the prompt file — the agent can resolve hostnames at runtime.

---

**[WARNING-2] Tailscale IPs for All Internal Services in K8s ConfigMap**

- **File:** `k8s/configmap.yaml`, lines 9, 22, 45–48, 52–53, 56–57
- **Finding:** Tailscale IPs for every internal service are hardcoded:
  - tensor-core (vLLM, Vision, Whisper, TTS, Ollama, SSH): `100.121.42.54`
  - mon2 (Prometheus, Alertmanager): `100.80.213.1`
  - mon1 (Gitea): `100.92.245.5`
  - SearXNG instance: `100.105.43.27`
- **Also in:** `deploy/claw-infra.env:5`, `deploy/claw-ops.env:5`, `deploy/claw-sentinel.env:5`
- **Also in:** `mesh/registry.py:47` (example comment), `tools/nexus-terminal.py:48` (`DEFAULT_HOST`)
- **Also in:** `observers/alertmanager_monitor.py` / config references
- **Impact:** Reveals the Tailscale network layout, service ports, and which node hosts which service. While Tailscale itself requires auth, this aids reconnaissance.
- **Recommendation:** Move Tailscale IPs to the secrets mechanism or use Tailscale MagicDNS hostnames (which are not sensitive). For the deploy `.env` files specifically, see INFO-1.

---

**[WARNING-3] Deploy `.env` Files with Real Infrastructure Config Committed**

- **Files:** `deploy/claw-infra.env`, `deploy/claw-ops.env`, `deploy/claw-sentinel.env`
- **First committed:** `4cfa8e9` (2026-03-21)
- **Finding:** These are real deployment environment files — not templates — containing actual Tailscale IPs, fabric IPs, service ports, and filesystem paths for the production deployment. Although bot tokens are set to `SET_ME`, the network topology data is real.
- **Additionally:** `AUTHORIZED_USER_ID=22276981` appears in all three files. This is a real Telegram user ID permanently committed to git history (present in history even after any edits).
- **Recommendation:** Add `deploy/*.env` to `.gitignore`. The `.gitignore` currently excludes `.env` and `.env.*` but not `deploy/*.env`. Replace with `deploy/*.env.example` template files with placeholder values.

---

**[WARNING-4] Company Registered Address Hardcoded in Production Code**

- **Files:**
  - `prompts/pureclaw_context.md`, line 174
  - `observers/doc_compiler.py`, lines 241–242
  - `observers/daily_report.py`, lines 226–227
  - `observers/weekly_report.py`, lines 278–279
- **Finding:** The string `"131 Continental Dr, Suite 305 / Newark, DE 19713, US"` is hardcoded as a fixed literal in the document generation code and system context prompt.
- **Impact:** Exposes the registered business address directly in source code of a public repo. Minor risk on its own (address is likely on public filings), but unnecessary exposure. Also makes the address hard to update.
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
- **Recommendation:** This file is a system prompt for the AI agent. Consider either (a) keeping a generic/redacted version in the public repo and injecting the real details at runtime via the deployment secret mechanism, or (b) moving the infrastructure context to a file excluded by `.gitignore`.

---

**[WARNING-6] Agent Email Addresses in K8s ConfigMap**

- **File:** `k8s/configmap.yaml`, line 61
- **Finding:** `AGENT_EMAIL: "hal@puretensors.com,hal@puretensor.com,hal@puretensor.ai"` — three real operational email addresses committed.
- **Also found in git history** in earlier versions of configmap.yaml.
- **Impact:** Links the bot identity to real email accounts. Low severity but confirms operational email addresses for the org.
- **Recommendation:** Move to the secrets mechanism or load from env. Alternatively, use the `.env.example` pattern (already done for most secrets) — the configmap should reference a placeholder.

---

### INFO

---

**[INFO-1] `.gitignore` Missing Entry for `deploy/*.env`**

- **File:** `.gitignore`
- **Finding:** The `.gitignore` correctly excludes `.env` and `.env.*` at the root, and `failover/.env*`. However, `deploy/claw-infra.env`, `deploy/claw-ops.env`, and `deploy/claw-sentinel.env` fall outside these patterns and are tracked by git.
- **Recommendation:** Add `deploy/*.env` (and optionally `deploy/*.conf`) to `.gitignore`.

---

**[INFO-2] `observers/.state/` Directory Committed**

- **Files:** `observers/.state/health_2026-03-18.json` through `health_2026-03-26.json`
- **Finding:** Runtime state files are committed. The current contents appear benign (observer run counts, timestamps). However, the `.gitignore` already lists `.state/` — these files may have been committed before that entry was added, or committed explicitly.
- **Recommendation:** Remove these files from tracking (`git rm --cached observers/.state/`) and ensure the `.gitignore` entry covers them going forward.

---

**[INFO-3] `=TLSv1.2` File at Repo Root**

- **File:** `=TLSv1.2` (root of repo)
- **Finding:** An empty file named `=TLSv1.2` exists at the repo root. This is likely the result of a copy-paste error (e.g., `curl ... --tlsv1.2 = file` misinterpretation). No sensitive data, but it is clutter.
- **Recommendation:** Delete this file.

---

**[INFO-4] Failover Filesystem Paths Expose Deployment Layout**

- **Files:** `deploy/claw-infra.env:25`, `deploy/claw-ops.env:25`, `deploy/claw-sentinel.env:19`
- **Finding:** Absolute filesystem paths are committed: `/home/puretensorai/nexus/...`, `/opt/claw-ops/...`, `/opt/claw-sentinel/...`. These reveal the deployment user account name (`puretensorai`) and directory structure.
- **Impact:** Low. Useful for an attacker who already has some access, not exploitable on its own.
- **Recommendation:** These files should not be in the public repo at all (see WARNING-3). Once moved to `.gitignore`, this is moot.

---

**[INFO-5] Security Policy Files Disable Private Range Blocking for Claw Agents**

- **Files:** `security/claw_infra_policy.yaml:28`, `security/claw_ops_policy.yaml:28`, `security/claw_sentinel_policy.yaml:29`
- **Finding:** `block_private_ranges: false` for all three Claw agents. This is intentional (they need LAN/Tailscale access), but it means these agents are not protected by SSRF controls. This is documented/expected behaviour but worth flagging.
- **Recommendation:** Document explicitly in the policy files that this is intentional and why. Consider adding a comment like `# Required: agents operate on LAN/Tailscale addresses`.

---

## No Issues Found — Confirmed Clean Areas

The following patterns were searched and returned **no findings** in PureClaw:

- API keys matching `sk-[a-z0-9]{20,}` (Anthropic), `xai-[a-z0-9]{20,}` (xAI), `ghp_[a-z0-9]{20,}` (GitHub PAT)
- AWS credentials (`AKIA[A-Z0-9]{16}`, `aws_secret_access_key`)
- Telegram bot token format (`[0-9]{8,12}:[a-zA-Z0-9_-]{35,}`) — the only match in history was a clearly fake example (`123456789:ABCdef...` in a README code block)
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
| CRITICAL | 1 | Fleet-wide BMC/IPMI password in plaintext |
| WARNING | 6 | Internal IPs, deploy env files, company address, infra topology, email addresses, Tailscale IPs |
| INFO | 5 | .gitignore gaps, stale state files, junk file, path exposure, SSRF policy intent |

**Repos scanned:** 1 (`puretensor/pureclaw`)
**Repos inaccessible:** 9 (no network/MCP scope to access; require re-audit)
**Total findings:** 12 (1 CRITICAL, 6 WARNING, 5 INFO)

### Priority Actions

1. **Immediate:** Rotate the BMC/IPMI password on all fleet nodes. It is publicly exposed in git history and cannot be "removed" without a full git history rewrite.
2. **Short term:** Add `deploy/*.env` to `.gitignore`, replace deploy env files with `.example` templates.
3. **Short term:** Remove specific IP addresses from `prompts/claw_infra_prompt.md` — use hostnames only.
4. **Medium term:** Audit the 9 inaccessible repos once network/MCP access is available.
5. **Medium term:** Consider whether `prompts/pureclaw_context.md` level of detail is appropriate for a public repo.
