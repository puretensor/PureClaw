# Hardening Rollout Notes

## Purpose
This runbook covers the deployment changes introduced by the hardening branch: authenticated ingress, restricted autonomous channels, and session-scoped backend routing.

## Required secrets
| Variable | Where it must exist | Used by |
|---|---|---|
| `MESH_SHARED_SECRET` | every mesh sender and receiver | `/mesh/message` auth/signing |
| `ALERTMANAGER_WEBHOOK_SECRET` | Alertmanager sender and every Nexus receiver handling alerts | `/webhook/alertmanager` auth/signing |
| `WA_WEBHOOK_SECRET` | `wa-bridge` and the Nexus instance receiving WhatsApp events | `/wa/incoming` auth/signing |
| `SUMMARY_BACKEND` | optional on Nexus / mesh runners | summary generation backend selection |

## Where to configure them
- Main Nexus service: `.env` or the systemd/Kubernetes env source used for the service
- Mesh runners: `deploy/claw-infra.env`, `deploy/claw-ops.env`, `deploy/claw-sentinel.env`, or equivalent deployment env sources
- WhatsApp bridge: `wa-bridge` runtime env / Kubernetes deployment env
- Alertmanager: the sender configuration that posts to Nexus

## Behavior changes to expect
- unsigned or incorrectly signed requests to `/mesh/message`, `/webhook/alertmanager`, and `/wa/incoming` now fail
- autonomous email and WhatsApp flows run with restricted tool profiles and should not perform mutating actions
- backend selection is now session-scoped; changing a backend in one chat no longer changes the process-wide backend
- summary generation now honors `SUMMARY_BACKEND` as well as `SUMMARY_MODEL`

## Rollout sequence
1. Set `MESH_SHARED_SECRET` on every mesh sender and receiver.
2. Set `ALERTMANAGER_WEBHOOK_SECRET` on Alertmanager and every receiving Nexus/mesh service.
3. Set `WA_WEBHOOK_SECRET` on `wa-bridge` and the receiving Nexus service.
4. Confirm `SUMMARY_MODEL` and, if needed, `SUMMARY_BACKEND` for the target deployment.
5. Restart or redeploy `wa-bridge`.
6. Restart or redeploy mesh agents.
7. Restart or redeploy Nexus.
8. Send one signed test request to each protected ingress path.
9. Validate that autonomous email/WhatsApp flows remain non-mutating.

## Verification checklist
- `/mesh/message` accepts a correctly signed request and rejects an unsigned one
- `/webhook/alertmanager` accepts a correctly signed request and rejects an unsigned one
- `/wa/incoming` accepts a correctly signed request from `wa-bridge`
- session backend changes stay isolated to the current chat/session
- summaries use the expected backend/model pairing

## Failure symptoms
| Symptom | Likely cause | Fix |
|---|---|---|
| 401/403 on protected ingress | missing or mismatched shared secret | set the same secret on sender and receiver, then restart/redeploy |
| mesh sender returns auth/config error | `MESH_SHARED_SECRET` missing on sender | configure sender env and redeploy |
| WhatsApp messages stop reaching Nexus | `WA_WEBHOOK_SECRET` missing or mismatched | set the secret on both `wa-bridge` and Nexus |
| alerts stop posting into Nexus | `ALERTMANAGER_WEBHOOK_SECRET` missing or mismatched | update Alertmanager and receiving service env |
| summaries use an unexpected provider | `SUMMARY_BACKEND` unset or defaulting | set `SUMMARY_BACKEND` explicitly |
| backend choice no longer affects other chats | expected behavior after hardening | no action needed |

## Post-rollout smoke test
- send a signed mesh message and confirm it is processed
- send a signed Alertmanager test alert and confirm receipt
- send a WhatsApp message through the bridge and confirm receipt
- switch backend in one chat and confirm another chat is unaffected
- trigger a summary and confirm the configured summary backend/model are used
