"""Core email loop — IMAP polling with per-recipient profile routing.

Refactored extraction of channels/email_in.py. Same 11 safety gates,
but routes auto-replies through profile-matched backends instead of
the global engine singleton.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
from datetime import timedelta
from pathlib import Path

from config import AUTHORIZED_USER_ID, AGENT_NAME, SYSTEM_PROMPT
from db import (
    is_email_seen,
    mark_email_seen,
    get_session,
    upsert_session,
    get_lock,
    content_hash,
    is_email_content_seen,
    mark_email_content_seen,
    has_reply_been_sent,
    record_reply_sent,
    count_replies_to_sender,
)
from drafts.classifier import classify_email
from drafts.queue import GMAIL_SCRIPT, GMAIL_IDENTITY

# Import utility functions from email_in (all module-level, safe to import)
from channels.email_in import (
    fetch_new_emails,
    _parse_email_age,
    _is_terminal,
    _email_chat_id,
    EMAIL_SYSTEM_PROMPT_TEMPLATE,
    AGENT_ADDRESSES,
    ACCOUNTS_FILE,
    MAX_EMAIL_AGE_HOURS,
    MAX_REPLIES_PER_SENDER_PER_HOUR,
)

# Memory injection (same pattern as engine.py:268-279)
try:
    from memory import get_memories_for_injection, get_shared_context
except ImportError:
    get_memories_for_injection = None
    get_shared_context = None

from hal_mail.profiles import load_profiles, match_profile
from hal_mail.health import HealthServer

log = logging.getLogger("hal-mail")

# Environment config
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "120"))
DRY_RUN = os.environ.get("DRY_RUN", "false").lower() in ("true", "1", "yes")
PROFILES_PATH = os.environ.get(
    "RECIPIENT_PROFILES_PATH", "/app/recipient_profiles.yaml"
)


class HalMailRunner:
    """Standalone email processing loop with per-recipient profile routing."""

    def __init__(
        self,
        profiles: dict,
        backends: dict,
        bot,
        health: HealthServer,
    ):
        self._profiles = profiles
        self._backends = backends
        self._bot = bot
        self._health = health
        self._profiles_path = Path(PROFILES_PATH)
        self._profiles_mtime: float = 0
        self._running = True

    async def run(self):
        """Main loop -- polls IMAP, processes emails."""
        log.info(
            "Runner started (poll=%ds, dry_run=%s, profiles=%s)",
            POLL_INTERVAL,
            DRY_RUN,
            self._profiles_path,
        )
        while self._running:
            try:
                self._check_profile_reload()
                await self._poll_once()
                self._health.record_poll()
            except Exception as e:
                log.exception("Poll error: %s", e)
                self._health.record_error()
            await asyncio.sleep(POLL_INTERVAL)

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------
    # Profile hot-reload
    # ------------------------------------------------------------------

    def _check_profile_reload(self):
        """Hot-reload profiles if file mtime changed."""
        try:
            mtime = self._profiles_path.stat().st_mtime
            if mtime > self._profiles_mtime:
                log.info("Reloading recipient profiles (mtime changed)")
                self._profiles = load_profiles(self._profiles_path)
                self._profiles_mtime = mtime
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------
    # IMAP polling
    # ------------------------------------------------------------------

    async def _poll_once(self):
        """Single poll cycle -- fetch from all accounts, classify, act."""
        if not ACCOUNTS_FILE.exists():
            return

        accounts = json.loads(ACCOUNTS_FILE.read_text())

        for account in accounts:
            name = account.get("name", account["username"])
            role = account.get("role", "monitor")

            # DRY_RUN: force monitor mode (no IMAP side effects)
            effective_role = "monitor" if DRY_RUN else role

            loop = asyncio.get_event_loop()
            emails = await loop.run_in_executor(
                None, fetch_new_emails, account, effective_role
            )

            if emails:
                log.info(
                    "Fetched %d emails from %s (%s)", len(emails), name, role
                )

            for em in emails:
                try:
                    await self._process_email(em, role, name)
                    self._health.record_email()
                except Exception as e:
                    log.exception(
                        "Error processing email from %s: %s",
                        em.get("from", "?"),
                        e,
                    )
                    self._health.record_error()

    # ------------------------------------------------------------------
    # Safety gate pipeline (identical to email_in.py)
    # ------------------------------------------------------------------

    async def _process_email(self, em: dict, role: str, account_name: str):
        """Process a single email through all safety gates."""
        sender_addr = (em.get("from_addr") or em["from"]).lower().strip()

        # --- Gate 1: Self-ignore (agent's own email addresses) ---
        if sender_addr in AGENT_ADDRESSES:
            return

        # --- Gate 2: Bretalon workflow routing ---
        if re.search(r"\[BRETALON\]", em.get("subject", ""), re.IGNORECASE):
            log.info(
                "Bretalon workflow email from %s -- deferring to observer",
                sender_addr,
            )
            await self._send_notification(em, tag="BRETALON")
            return

        # --- Gate 3: Message-ID dedup ---
        if is_email_seen(em["id"]):
            return

        # --- Gate 4: Content-hash dedup (cross-account) ---
        body_text = em.get("body", "")
        chash = content_hash(sender_addr, em.get("subject", ""), body_text)
        if is_email_content_seen(chash):
            log.info(
                "Content-hash duplicate from %s (account=%s), skipping",
                sender_addr,
                account_name,
            )
            return

        if not DRY_RUN:
            mark_email_content_seen(chash, em["id"], account_name)

        # --- Gate 5: Classify (role-aware) ---
        classification = classify_email(
            em["from"],
            em["subject"],
            em.get("to", ""),
            account_role=role,
        )

        if classification == "ignore":
            if DRY_RUN:
                log.info(
                    "[DRY_RUN] IGNORE: %s re: %s", sender_addr, em["subject"]
                )
            return

        # --- Gate 6: Age filter ---
        email_age = _parse_email_age(em.get("date_raw", ""))
        if email_age and email_age > timedelta(hours=MAX_EMAIL_AGE_HOURS):
            log.info(
                "Skipping aged email from %s (%s old)", sender_addr, email_age
            )
            await self._send_notification(em, tag="AGED")
            return

        # Match profile for logging
        profile = match_profile(sender_addr, self._profiles)
        backend_name = profile.get("backend", "vllm")

        if classification == "notify":
            if DRY_RUN:
                log.info(
                    "[DRY_RUN] NOTIFY: %s re: %s (backend: %s)",
                    sender_addr,
                    em["subject"],
                    backend_name,
                )
            else:
                await self._send_notification(em)
            return

        if classification == "followup":
            if DRY_RUN:
                log.info(
                    "[DRY_RUN] FOLLOW-UP: %s re: %s",
                    sender_addr,
                    em["subject"],
                )
            else:
                await self._send_notification(em, tag="FOLLOW-UP")
            return

        # --- classification == "auto_reply" from here ---
        if role != "primary":
            if DRY_RUN:
                log.info(
                    "[DRY_RUN] NOTIFY (monitor): %s re: %s",
                    sender_addr,
                    em["subject"],
                )
            else:
                await self._send_notification(em)
            return

        # --- Gate 7: Reply-sent guard (content-hash in email_replies_sent) ---
        if has_reply_been_sent(chash):
            log.info(
                "Already replied to this content from %s, skipping",
                sender_addr,
            )
            await self._send_notification(em, tag="DEDUP")
            return

        # --- Gate 8: Rate limit per sender ---
        reply_count = count_replies_to_sender(sender_addr, hours=1)
        if reply_count >= MAX_REPLIES_PER_SENDER_PER_HOUR:
            log.warning(
                "Rate limit: %d replies to %s in last hour, skipping",
                reply_count,
                sender_addr,
            )
            await self._send_notification(em, tag="RATE-LIMITED")
            return

        if DRY_RUN:
            log.info(
                "[DRY_RUN] AUTO_REPLY: %s re: %s -> backend=%s model=%s",
                sender_addr,
                em["subject"],
                backend_name,
                profile.get("model") or "default",
            )
            return

        # All gates passed -- create auto reply with profile routing
        await self._create_auto_reply(em, chash, profile)

    # ------------------------------------------------------------------
    # Auto-reply with per-recipient backend routing
    # ------------------------------------------------------------------

    async def _create_auto_reply(self, em: dict, chash: str, profile: dict):
        """Compose and send a reply using the profile-matched backend."""
        body_preview = em["body"][:3000] if em.get("body") else "(no body)"

        # Pre-filter: skip terminal one-liners
        if _is_terminal(body_preview):
            log.info(
                "Email from %s re: %s -- terminal phrase, skipping",
                em["from_addr"],
                em["subject"],
            )
            await self._send_notification(em, tag="SKIPPED")
            return

        sender_addr = em["from_addr"] or em["from"]
        backend_name = profile.get("backend", "vllm")
        model = profile.get("model") or "default"
        personality = profile.get("personality", "")

        if backend_name in {"claude_code", "codex_cli", "gemini_cli"}:
            log.error(
                "Unsafe CLI backend '%s' rejected for autonomous email reply to %s",
                backend_name,
                sender_addr,
            )
            await self._send_notification(em, tag="UNSAFE-BACKEND")
            return

        backend = self._backends.get(backend_name)
        if not backend:
            log.error(
                "Backend '%s' not available for %s, falling back to notify",
                backend_name,
                sender_addr,
            )
            await self._send_notification(em)
            return

        log.info(
            "Auto-reply to %s using backend=%s model=%s",
            sender_addr,
            backend_name,
            model,
        )

        chat_id = _email_chat_id(sender_addr)
        lock = get_lock(chat_id)

        async with lock:
            session = get_session(chat_id)
            session_id = session["session_id"] if session else None
            msg_count = session["message_count"] if session else 0
            session_backend = session.get("backend") if session else backend_name

            user_message = (
                f"[Email from {em['from']}]\n"
                f"Subject: {em['subject']}\n\n"
                f"{body_preview}"
            )

            # Build extra_system_prompt: email rules + profile personality
            extra_sp = EMAIL_SYSTEM_PROMPT_TEMPLATE.format(
                agent_name=AGENT_NAME,
                sender_name=em["from"],
                sender_addr=sender_addr,
                subject=em["subject"],
            )
            if personality:
                extra_sp += (
                    "\n\nPERSONALITY FOR THIS RECIPIENT:\n" + personality
                )

            # Memory injection (same pattern as engine.py:268-279)
            memory_ctx = self._build_memory_context()

            try:
                from backends.tools import ToolExecutionContext

                data = await backend.call_streaming(
                    user_message,
                    session_id=session_id,
                    model=model,
                    streaming_editor=None,
                    system_prompt=SYSTEM_PROMPT,
                    memory_context=memory_ctx,
                    extra_system_prompt=extra_sp,
                    tool_context=ToolExecutionContext(
                        policy_profile="reply_only",
                        session_id=session_id,
                        channel="email_auto",
                    ),
                )
                reply_body = data.get("result", "").strip()
                new_session_id = data.get("session_id", session_id)
            except Exception as e:
                log.warning(
                    "Engine call failed for %s (backend=%s): %s",
                    em["from"],
                    backend_name,
                    e,
                )
                await self._send_notification(em)
                return

            if not reply_body:
                log.warning(
                    "Empty reply for %s re: %s", sender_addr, em["subject"]
                )
                await self._send_notification(em)
                return

            # Enforce max reply length
            max_len = profile.get("max_reply_length", 3000)
            if len(reply_body) > max_len:
                reply_body = reply_body[:max_len]

            # Persist session for sender continuity
            upsert_kwargs = {"backend": session_backend} if session_backend is not None else {}
            upsert_session(chat_id, new_session_id, model, msg_count + 1, **upsert_kwargs)

        # Send reply via gmail.py
        try:
            reply_subject = em["subject"]
            if not reply_subject.lower().startswith("re:"):
                reply_subject = f"Re: {reply_subject}"

            loop = asyncio.get_event_loop()
            send_result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [
                        "python3",
                        str(GMAIL_SCRIPT),
                        GMAIL_IDENTITY,
                        "reply",
                        "--to",
                        sender_addr,
                        "--subject",
                        reply_subject,
                        "--id",
                        em["id"],
                        "--body",
                        reply_body,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                ),
            )

            if send_result.returncode == 0:
                log.info(
                    "Auto-replied to %s re: %s (backend=%s)",
                    sender_addr,
                    em["subject"],
                    backend_name,
                )
                record_reply_sent(
                    sender_addr, em["subject"], chash, reply_body[:200]
                )
                await self._send_reply_notification(em, reply_body, profile)
            else:
                error = send_result.stderr[:200] or send_result.stdout[:200]
                log.warning(
                    "Auto-reply send failed for %s: %s", sender_addr, error
                )
                await self._send_notification(em)
        except Exception as e:
            log.warning("Auto-reply error for %s: %s", sender_addr, e)
            await self._send_notification(em)

    # ------------------------------------------------------------------
    # Memory injection (replicates engine.py:268-287)
    # ------------------------------------------------------------------

    def _build_memory_context(self) -> str | None:
        """Build memory context for prompt injection."""
        parts = []
        if get_memories_for_injection:
            mem = get_memories_for_injection()
            if mem:
                parts.append(mem)
        if get_shared_context:
            shared = get_shared_context()
            if shared:
                parts.append(shared)

        memory_ctx = "\n\n".join(parts) if parts else None

        # Redact before injection
        try:
            from security.redact import redact_text

            if memory_ctx:
                memory_ctx = redact_text(memory_ctx)
        except Exception:
            pass

        return memory_ctx

    # ------------------------------------------------------------------
    # Telegram notifications (direct Bot API, no polling)
    # ------------------------------------------------------------------

    async def _send_notification(self, em: dict, tag: str = "EMAIL"):
        """Send a Telegram notification about an email."""
        if not self._bot or DRY_RUN:
            return

        text = (
            f"[{tag}] {em['date']}\n"
            f"From: {em['from']}\n"
            f"Subject: {em['subject']}\n"
        )
        if em.get("body"):
            preview = em["body"][:200].replace("\n", " ")
            text += f"\n{preview}..."

        try:
            await self._bot.send_message(
                chat_id=int(AUTHORIZED_USER_ID),
                text=text,
            )
        except Exception as e:
            log.warning("Failed to send notification: %s", e)

    async def _send_reply_notification(
        self, em: dict, reply_body: str, profile: dict
    ):
        """Send a [SENT] Telegram notification after auto-reply."""
        if not self._bot:
            return
        backend_name = profile.get("backend", "?")
        preview = (
            reply_body[:300] + "..." if len(reply_body) > 300 else reply_body
        )
        text = (
            f"[SENT] Auto-reply to {em['from']} ({backend_name})\n"
            f"Re: {em['subject']}\n\n"
            f"{preview}"
        )
        try:
            await self._bot.send_message(
                chat_id=int(AUTHORIZED_USER_ID),
                text=text,
            )
        except Exception:
            pass
