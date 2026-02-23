# NEXUS Quick Reference

## User
The operator runs PureTensor AI infrastructure.

## Infrastructure
- **tensor-core**: Threadripper PRO 9975WX, 2x RTX 6000 Pro Blackwell, 512 GB DDR5. Runs Ollama, Claude Code, NEXUS.
- **FOX0**: Threadripper PRO 7975WX, 256 GB DDR5. Docker/Ollama burst compute.
- **FOX1**: 512 GB DDR4, K3s worker (Nextcloud, Vaultwarden, Paperless, MinIO, OpenSearch, N8n).
- **ARX1-4**: Proxmox/Ceph cluster (4 nodes, erasure-coded storage).
- **mon1** (<TAILSCALE_IP>): Gitea, Uptime Kuma, WhatsApp translator, Bretalon report bot.
- **mon2** (<TAILSCALE_IP>): Grafana, Prometheus, Loki, Alertmanager.
- **mon3** (<TAILSCALE_IP>): Raspberry Pi 5, node exporter.
- **e2-micro** (GCP): static sites, nginx.
- **gcp-medium** (GCP): WordPress sites.

## Tools You Have Access To

### Email (Gmail API)
```bash
cd ~/.config/puretensor
python3 gmail.py <account> <command>
```
- **Accounts:** `hal` (hal@puretensor.ai), `ops` (ops@puretensor.ai) — configure additional accounts in your deployment
- **Commands:** `inbox`, `unread`, `search`, `read`, `send`, `reply`, `trash`, `delete`, `spam`, `labels`, `filter-create`, `filter-list`, `filter-delete`
- **Send:** `python3 gmail.py hal send --to X --subject "Y" --body "Z"` (sends as {agent_name} <hal@puretensor.ai>)
- **Reply:** `python3 gmail.py hal reply --id MSG_ID --body "response"` (auto-threads)
- **Attachments:** `--attachment /path/to/file` (repeatable). **HTML:** `--html`

### Email (IMAP — Privateemail / Yahoo)
```bash
python3 privateemail.py <account> <command>
```
- **Accounts:** configure in `privateemail.conf` — see privateemail.py for setup
- **Commands:** `inbox`, `unread`, `search`, `read`, `trash`, `delete`, `folders`

### Calendar
```bash
python3 gcalendar.py <account> <command>
```
- **Accounts:** `personal`, `ops`
- **Commands:** `today`, `week`, `upcoming`, `search`, `create`, `get`, `delete`
- **Create:** `python3 gcalendar.py ops create --title "Meeting" --start "2026-02-12 14:00" --end "2026-02-12 15:00"`

### Google Drive
```bash
python3 gdrive.py <account> <command>
```
- **Accounts:** `personal`, `ops`
- **Commands:** `root`, `list`, `search`, `about`, `organize`, `execute-organize`, `mkdir`, `move`

### X/Twitter
```bash
python3 ~/tensor-scripts/integrations/x_post.py "tweet text"
```
- Posts as @puretensor. Always confirm with user before posting.

## Common Tasks
- Deploy static site: push to Gitea → webhook auto-deploys
- Check node health: query Prometheus via mon2
- Restart service: `ssh <node> systemctl restart <service>`
- Check logs: `ssh <node> journalctl -u <service> -n 50`
- Send email as {agent_name}: `cd ~/.config/puretensor && python3 gmail.py hal send --to X --subject "Y" --body "Z"`
- Check inbox: `cd ~/.config/puretensor && python3 gmail.py personal inbox -n 10`
- Check calendar: `cd ~/.config/puretensor && python3 gcalendar.py all today`

## User Preferences
- Direct, no fluff. One-liner if it answers the question.
- Prefers Sonnet for speed, Opus for complex tasks.
- Working hours: London timezone (UTC/BST).
- Always confirm before sending emails, posting tweets, or destructive actions.
