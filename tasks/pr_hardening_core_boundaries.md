# PR: Harden security boundaries, ingress auth, and session backend routing

## Summary
This PR hardens PureClaw's trust boundary so tool execution fails closed, machine-to-machine ingress requires shared-secret authentication, autonomous inbound channels run under restricted tool profiles, and backend selection is scoped to the active session instead of global process state.

## Why
Before this branch:
- tool policy errors could fail open
- mesh / Alertmanager / WhatsApp ingress could accept unsigned requests
- autonomous email and WhatsApp flows could reach broader tool capability than intended
- backend switching could leak across chats because it mutated process-global state
- mesh work ran inline on the request path
- a few observer / startup bugs and async test warnings remained

## What changed
### Security boundary
- tool execution now fails closed on policy-check errors
- machine ingress now validates shared-secret HMAC/auth headers
- autonomous email and WhatsApp flows use restricted tool profiles
- summaries/redaction/config behavior was tightened and documented

### Architecture
- backend selection is now session-scoped and stored per chat/session
- engine/backend resolution now uses the session backend instead of mutable global runtime state

### Reliability
- mesh work is queued off the HTTP request path
- observer success messages are delivered correctly
- WhatsApp startup wiring and terminal `/sessions` bugs were fixed
- TTS conversion now verifies ffmpeg output existence

## Operational changes
### New / now-required environment variables
- `MESH_SHARED_SECRET` — required on mesh senders and receivers
- `ALERTMANAGER_WEBHOOK_SECRET` — required on Alertmanager sender and Nexus receiver
- `WA_WEBHOOK_SECRET` — required on `wa-bridge` and Nexus `/wa/incoming`
- `SUMMARY_BACKEND` — now controls which backend performs summary generation alongside `SUMMARY_MODEL`

### Behavior changes
- unsigned requests to `/mesh/message`, `/webhook/alertmanager`, and `/wa/incoming` are rejected
- autonomous email/WhatsApp flows are non-mutating by default
- backend changes no longer affect other chats or channels

## Validation
- `python3 -m pytest -q`
- result on this branch after follow-up cleanup: `998 passed in 109.48s`
- targeted async warning gate added to CI for:
  - `tests/test_escalation.py::TestEscalationFixCallback::test_fix_timeout_handled`
  - `tests/test_file_output.py::TestScanAndSendOutputs::test_tilde_expansion`
  - `tests/test_summaries.py::TestMaybeGenerateSummary::test_double_interval_triggers`
  - `tests/test_voice_tts.py::TestTextToVoiceNote::test_success`

## Commits in this branch
- `cebf5cc` security: fail closed on tool execution
- `0aa13c2` security: authenticate ingress and restrict autonomous channels
- `8ca880f` architecture: make backend selection session scoped
- `4fd3166` reliability: queue mesh work and fix observer runtime bugs
- follow-up docs/test cleanup commits in this branch

## Reviewer checklist
- [ ] `MESH_SHARED_SECRET`, `ALERTMANAGER_WEBHOOK_SECRET`, and `WA_WEBHOOK_SECRET` are configured where needed
- [ ] unsigned ingress is rejected for mesh, Alertmanager, and WhatsApp paths
- [ ] autonomous email/WhatsApp flows cannot perform mutating tool actions
- [ ] backend changes are isolated to the active session/chat
- [ ] mesh requests stay responsive while long-running work is processed
- [ ] rollout notes in `deploy/hardening_rollout.md` are sufficient for operators

## Rollout notes
Use `deploy/hardening_rollout.md` for the exact rollout order, failure modes, and service-specific secret placement.

## Follow-ups
- keep the suite warning-free as async mocking patterns evolve
- consider brokered/fully-audited support for CLI backends before widening production use
