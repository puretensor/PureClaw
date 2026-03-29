# Codex Audit Report: PureClaw

_Date: 2026-03-29_

_Test run:_ `python3 -m pytest tests/test_security_policy.py tests/test_security_filesystem.py tests/test_security_network.py tests/test_security_redact.py tests/test_scheduler.py tests/test_session_commands.py -q` → **213 passed**

## 1. Executive Summary
PureClaw is ambitious and the core security/scheduler/session layers are better tested than many systems of this size. The biggest architectural weakness is isolation: backend selection, terminal sessions, and mesh traffic are all more global/shared than the repo description implies. The two highest-risk items are unauthenticated mesh traffic and cross-session/global backend mutation.

## 2. Architecture Assessment
The monolithic `nexus.py` + `engine.py` core is still workable, but only if state boundaries stay explicit. Right now several key decisions that should be scoped per session or per peer are stored globally, which makes cross-channel interference too easy. The observer model and backend abstraction are viable, but the mesh/security perimeter needs a harder contract.

## 3. Critical Findings

### Critical — Security — Mesh traffic is unauthenticated
- **Evidence:** `mesh/server.py:79-126`; `mesh/client.py:67-84`
- **Issue:** `/mesh/message` and the Alertmanager webhook accept plain JSON with no shared secret, signature, or mTLS. Any reachable client can inject `task`, `query`, or `alert` payloads into the mesh path.
- **Recommendation:** Add HMAC or mTLS, verify sender identity, and restrict source addresses before handing messages to the LLM/tooling path.
- **Priority:** Do now

## 4. High Priority Findings

### High — Architecture — Backend switching is global, not per session
- **Evidence:** `channels/telegram/commands.py:281-338`; `channels/discord/handlers.py:71-93`; `channels/terminal/__init__.py:155-203`
- **Issue:** Channel commands mutate `config.ENGINE_BACKEND` directly. One user/channel can flip the backend for every other channel, observer, and terminal client.
- **Recommendation:** Store backend selection in session state and route through the engine per request instead of mutating global config.
- **Priority:** Do now

### High — Reliability / Security — Terminal WebSocket clients all share one fixed session namespace
- **Evidence:** `channels/terminal/__init__.py:30-31`, `323-365`
- **Issue:** Every terminal connection uses the same `TERMINAL_CHAT_ID`, lock, and session history. Multiple clients can collide, block each other, or inherit the same conversation state.
- **Recommendation:** Allocate a unique chat/session ID per WebSocket connection or per authenticated token.
- **Priority:** Do now

### High — Security — Missing policy file falls back to permissive defaults
- **Evidence:** `security/policy.py:196-199`
- **Issue:** If the policy file is absent, the system silently loads an allow-everything policy.
- **Recommendation:** Fail closed in production mode, or require an explicit env var to allow permissive fallback.
- **Priority:** Do now

### High — Security — Network policy enforcement is incomplete
- **Evidence:** `backends/tools.py:1435-1442`
- **Issue:** `_check_security_policy()` only applies URL/domain checks to `web_fetch`. `web_search` and several other networked paths bypass the SSRF/domain layer entirely.
- **Recommendation:** Move egress checks into shared HTTP client wrappers or cover every outward-calling tool.
- **Priority:** Next sprint

## 5. Medium / Low Findings

### Medium — Reliability — Mesh request handling can block the aiohttp server
- **Evidence:** `mesh/server.py:103-106`; `engine.py:336-391`
- **Issue:** The mesh server calls synchronous LLM work directly from the request handler path.
- **Recommendation:** Offload heavyweight processing to a worker queue/task and return an async acknowledgement.
- **Priority:** Next sprint

### Medium — UX / Security — Terminal auth path collapses all errors into “Auth timeout”
- **Evidence:** `channels/terminal/__init__.py:448-465`
- **Issue:** Bad JSON, internal errors, and actual timeouts all look the same to the client.
- **Recommendation:** Separate protocol/auth failures from internal exceptions for clearer operator feedback.
- **Priority:** Backlog

## 6. Prioritized Roadmap
1. Authenticate/sign mesh traffic.
2. Remove global backend mutation; make backend routing session-scoped.
3. Give terminal clients isolated session IDs and locks.
4. Fail closed when the security policy is missing.
5. Centralize network egress enforcement across all tools.
6. Make mesh request handling asynchronous/non-blocking.
7. Add end-to-end tests for multi-client terminal behavior.
8. Audit observer network/tool paths for policy bypasses.
9. Add explicit per-channel/backend telemetry so cross-talk is visible.
10. Keep the current module layout, but harden the state boundaries.
