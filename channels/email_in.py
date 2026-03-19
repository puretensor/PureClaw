"""Email input channel — polls IMAP for incoming messages.

Lifecycle:
  1. Polls IMAP inbox every 2 minutes for new unread messages
  2. Classifies each message (ignore / notify / auto_reply / followup)
  3. For auto_reply: asks Claude to draft a reply, sends it immediately
  4. For notify: sends a Telegram notification
  5. Marks messages as seen in SQLite (Message-ID + content hash) to prevent duplicates
  6. Primary accounts: marks emails as SEEN in IMAP after processing
  7. Monitor accounts: read-only (PEEK), never auto-reply

Defense-in-depth: an email must pass ALL of these gates to trigger an auto-reply:
  1. Account role = primary
  2. Not from agent's own addresses
  3. Not a [BRETALON] workflow email
  4. Message-ID not in email_seen
  5. Content hash not in email_seen
  6. Email age < 24 hours
  7. Not a terminal one-liner
  8. Classifier returns auto_reply (VIP domain match)
  9. Content hash not in email_replies_sent
  10. Sender rate < 3 replies/hour
  11. Engine returns non-empty reply

Uses the same IMAP accounts file as EmailDigestObserver.
"""

import asyncio
import email
import email.header
import email.utils
import html
import imaplib
import json
import logging
import os
import re
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from channels.base import Channel
from config import AUTHORIZED_USER_ID, AGENT_NAME, log
from db import (
    is_email_seen, mark_email_seen, get_session, upsert_session, get_lock,
    content_hash, is_email_content_seen, mark_email_content_seen,
    has_reply_been_sent, record_reply_sent, count_replies_to_sender,
)
from drafts.classifier import classify_email
from drafts.queue import create_email_draft

# IMAP accounts config — shared with email_digest observer
ACCOUNTS_FILE = Path(__file__).parent.parent / "observers" / "email_accounts.json"

# Poll interval (seconds)
POLL_INTERVAL = 120  # 2 minutes

# Max emails to process per poll cycle
MAX_PER_POLL = 20

# Max age for emails to be processed (prevents backlog avalanche on restart)
MAX_EMAIL_AGE_HOURS = 24

# Rate limit: max auto-replies per sender per hour
MAX_REPLIES_PER_SENDER_PER_HOUR = 3

# Agent's own email addresses (self-ignore — prevents reply loops)
# Set AGENT_EMAIL env var to a comma-separated list of addresses
_AGENT_EMAIL_RAW = os.environ.get("AGENT_EMAIL", "")
AGENT_ADDRESSES = {a.strip().lower() for a in _AGENT_EMAIL_RAW.split(",") if a.strip()}

# Exact terminal phrases that don't warrant a reply.
# Only suppresses when the entire message body (after signature stripping) matches.
TERMINAL_PHRASES = {
    "thank you", "thanks", "thanks!", "thank you!",
    "ok", "okay", "got it", "noted", "cheers",
    "cheers!", "will do", "perfect", "great",
}

# Patterns that mark the start of an email signature
_SIG_PATTERNS = re.compile(
    r"^(?:--|___+|---+|best regards|kind regards|regards|cheers,"
    r"|thanks,|thank you,|sent from my|get outlook)",
    re.IGNORECASE,
)

# Offset email chat IDs to avoid collision with Telegram user IDs
EMAIL_CHAT_ID_OFFSET = 900_000_000_000

# System prompt template injected via extra_system_prompt (same pattern as Discord)
EMAIL_SYSTEM_PROMPT_TEMPLATE = (
    "IMPORTANT: You are {agent_name}, responding to an email (not Telegram/Discord).\n\n"
    "Sender: {sender_name} <{sender_addr}>\n"
    "Subject: {subject}\n\n"
    "CRITICAL EMAIL RULES:\n"
    "- Your text output IS the reply. It will be sent automatically to the sender.\n"
    "- NEVER use gmail.py, smtp, or any tool to send/reply to emails yourself.\n"
    "- NEVER send emails to anyone other than the sender — no CC, no BCC, no separate notifications.\n"
    "- If you need to take actions (delete WP posts, research, etc.), do that silently with tools.\n"
    "- Then write your response as a direct reply to the sender. That's all.\n"
    "- Do NOT narrate what you did to a third party. Just reply to the sender.\n\n"
    "RESPONSE STYLE:\n"
    "- Reply naturally and conversationally. Be warm, concise, helpful.\n"
    "- Plain text only (no markdown, no HTML).\n"
    "- Keep replies appropriately sized — short for simple things, detailed for complex requests.\n"
    "- Sign off as {agent_name}.\n"
    "- You are {agent_name}, not Claude. Never mention being an AI or language model.\n"
    "- This session has history — you remember previous emails with this sender."
)


def _email_chat_id(sender_email: str) -> int:
    """Deterministic chat ID for an email sender (avoids Telegram ID collision)."""
    normalized = sender_email.lower().strip()
    return EMAIL_CHAT_ID_OFFSET + abs(zlib.crc32(normalized.encode())) % (2**31)


def _is_terminal(body: str) -> bool:
    """Return True if the email body is a bare terminal phrase (after stripping signature)."""
    lines = body.strip().splitlines()
    # Drop signature block: everything from the first signature-like line onward
    content_lines = []
    for line in lines:
        if _SIG_PATTERNS.match(line.strip()):
            break
        content_lines.append(line)
    content = " ".join(content_lines).strip().lower()
    # Strip trailing punctuation for matching flexibility
    content_norm = content.rstrip("!.,")
    return content in TERMINAL_PHRASES or content_norm in TERMINAL_PHRASES


def _decode_header(raw: str) -> str:
    """Decode an email header (handles encoded words like =?UTF-8?Q?...?=)."""
    if not raw:
        return ""
    parts = email.header.decode_header(raw)
    decoded = []
    for data, charset in parts:
        if isinstance(data, bytes):
            decoded.append(data.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(data)
    return " ".join(decoded)


def _extract_email_addr(header_value: str) -> str:
    """Extract bare email address from a header like 'Name <user@example.com>'."""
    if not header_value:
        return ""
    _, addr = email.utils.parseaddr(header_value)
    return addr or header_value


def _strip_html(raw_html: str) -> str:
    """Strip HTML tags and decode entities to produce readable plain text."""
    # Remove style/script blocks entirely (content + tags)
    text = re.sub(r"<style[^>]*>.*?</style>", "", raw_html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.IGNORECASE | re.DOTALL)
    # Replace block-level tags with newlines for readability
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(?:p|div|tr|li|h[1-6])>", "\n", text, flags=re.IGNORECASE)
    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode HTML entities (&amp; &nbsp; &#8230; etc.)
    text = html.unescape(text)
    # Collapse whitespace runs (but keep single newlines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _get_body(msg) -> str:
    """Extract plain text body from an email message."""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")
        # Fallback: try text/html (strip tags to plain text)
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return _strip_html(payload.decode(charset, errors="replace"))
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            # If the single part is HTML, strip it
            if msg.get_content_type() == "text/html":
                return _strip_html(text)
            return text
    return ""


def _parse_email_age(date_str: str) -> timedelta | None:
    """Parse email Date header and return age as timedelta, or None on failure."""
    try:
        parsed = email.utils.parsedate_to_datetime(date_str)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - parsed
    except Exception:
        return None


def fetch_new_emails(account: dict, role: str = "monitor") -> list[dict]:
    """Fetch unread emails from a single IMAP account.

    For 'primary' role: opens mailbox read-write and marks emails as SEEN after fetch.
    For 'monitor' role: opens mailbox readonly with BODY.PEEK[] (no side effects).

    Returns list of dicts with: id, from, from_addr, subject, date, date_raw, to, body
    """
    server = account["server"]
    port = account.get("port", 993)
    username = account["username"]
    password = account["password"]

    is_primary = (role == "primary")

    try:
        conn = imaplib.IMAP4_SSL(server, port)
        conn.login(username, password)
    except Exception as e:
        log.warning("Email input: connection to %s failed: %s", server, e)
        return []

    try:
        conn.select("INBOX", readonly=(not is_primary))
        status, data = conn.search(None, "UNSEEN")
        if status != "OK" or not data[0]:
            return []

        uids = data[0].split()[-MAX_PER_POLL:]
        uids.reverse()

        results = []
        for uid in uids:
            try:
                # Primary: fetch full body (marks as seen implicitly via BODY[])
                # Monitor: use PEEK to avoid side effects
                fetch_cmd = "(BODY[])" if is_primary else "(BODY.PEEK[])"
                status, msg_data = conn.fetch(uid, fetch_cmd)
                if status != "OK" or not msg_data or msg_data[0] is None:
                    continue

                raw = msg_data[0][1] if isinstance(msg_data[0], tuple) else msg_data[0]
                msg = email.message_from_bytes(raw)

                from_raw = _decode_header(msg.get("From", ""))
                subject = _decode_header(msg.get("Subject", "(no subject)"))
                date_str = msg.get("Date", "")
                msg_id = msg.get("Message-ID", f"{uid.decode()}@{server}")
                to_raw = _decode_header(msg.get("To", ""))
                body = _get_body(msg)

                # Parse date for display
                try:
                    parsed = email.utils.parsedate_to_datetime(date_str)
                    date_display = parsed.strftime("%b %d %H:%M")
                except Exception:
                    date_display = date_str[:16] if date_str else "unknown"

                results.append({
                    "id": msg_id.strip(),
                    "from": from_raw,
                    "from_addr": _extract_email_addr(from_raw),
                    "subject": subject,
                    "date": date_display,
                    "date_raw": date_str,
                    "to": to_raw,
                    "body": body[:5000],  # Cap body size
                })

                # Primary accounts: explicitly mark as \Seen
                if is_primary:
                    try:
                        conn.store(uid, "+FLAGS", "\\Seen")
                    except Exception as e:
                        log.warning("Email input: failed to mark UID %s as seen: %s",
                                    uid.decode(), e)

            except Exception as e:
                log.warning("Email input: skipping UID %s: %s", uid.decode(), e)

        return results

    except Exception as e:
        log.warning("Email input: fetch error for %s: %s", server, e)
        return []
    finally:
        try:
            conn.logout()
        except Exception:
            pass


class EmailInputChannel(Channel):
    """Polls IMAP for new emails, classifies them, and creates drafts or notifications."""

    def __init__(self, bot=None):
        self._bot = bot
        self._task = None

    async def start(self):
        """Start the email polling loop."""
        self._task = asyncio.create_task(self._poll_loop())
        log.info("Email input channel started (polling every %ds)", POLL_INTERVAL)

    async def stop(self):
        """Stop the email polling loop."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("Email input channel stopped")

    async def _poll_loop(self):
        """Main loop — checks for new emails every POLL_INTERVAL seconds."""
        while True:
            try:
                await self._poll_once()
            except Exception as e:
                log.exception("Email input poll error: %s", e)
            await asyncio.sleep(POLL_INTERVAL)

    async def _poll_once(self):
        """Single poll cycle — fetch, classify, act."""
        if not ACCOUNTS_FILE.exists():
            return

        accounts = json.loads(ACCOUNTS_FILE.read_text())

        for account in accounts:
            name = account.get("name", account["username"])
            role = account.get("role", "monitor")  # safe default: monitor

            # Run IMAP fetch in thread pool (blocking I/O)
            loop = asyncio.get_event_loop()
            emails = await loop.run_in_executor(None, fetch_new_emails, account, role)

            for em in emails:
                await self._process_email(em, role, name)

    async def _process_email(self, em: dict, role: str, account_name: str):
        """Process a single email through all safety gates."""
        sender_addr = (em.get("from_addr") or em["from"]).lower().strip()

        # --- Gate 1: Self-ignore (agent's own email addresses) ---
        if sender_addr in AGENT_ADDRESSES:
            return

        # --- Gate 2: Bretalon workflow routing ---
        if re.search(r'\[BRETALON\]', em.get("subject", ""), re.IGNORECASE):
            log.info("Bretalon workflow email from %s — deferring to observer", sender_addr)
            await self._send_notification(em, tag="BRETALON")
            return

        # --- Gate 3: Message-ID dedup ---
        if is_email_seen(em["id"]):
            return

        # --- Gate 4: Content-hash dedup (cross-account) ---
        body_text = em.get("body", "")
        chash = content_hash(sender_addr, em.get("subject", ""), body_text)
        if is_email_content_seen(chash):
            log.info("Content-hash duplicate from %s (account=%s), skipping",
                     sender_addr, account_name)
            return

        # Mark as seen with both message_id and content_hash
        mark_email_content_seen(chash, em["id"], account_name)

        # --- Gate 5: Classify (role-aware) — ignore returns early with no notification ---
        classification = classify_email(
            em["from"], em["subject"], em.get("to", ""),
            account_role=role,
        )

        if classification == "ignore":
            return

        # --- Gate 6: Age filter (skip emails older than MAX_EMAIL_AGE_HOURS) ---
        # Runs AFTER classify so that ignored emails are silently dropped,
        # but old non-ignored emails get an [AGED] tag notification.
        email_age = _parse_email_age(em.get("date_raw", ""))
        if email_age and email_age > timedelta(hours=MAX_EMAIL_AGE_HOURS):
            log.info("Skipping aged email from %s (%s old)", sender_addr, email_age)
            await self._send_notification(em, tag="AGED")
            return

        if classification == "notify":
            await self._send_notification(em)
            return

        if classification == "followup":
            await self._send_notification(em, tag="FOLLOW-UP")
            return

        # --- classification == "auto_reply" from here ---
        # (classifier already gated on role=primary, but double-check)
        if role != "primary":
            await self._send_notification(em)
            return

        # --- Gate 7: Reply-sent guard (content-hash in email_replies_sent) ---
        if has_reply_been_sent(chash):
            log.info("Already replied to this content from %s, skipping", sender_addr)
            await self._send_notification(em, tag="DEDUP")
            return

        # --- Gate 8: Rate limit per sender ---
        reply_count = count_replies_to_sender(sender_addr, hours=1)
        if reply_count >= MAX_REPLIES_PER_SENDER_PER_HOUR:
            log.warning("Rate limit: %d replies to %s in last hour, skipping",
                        reply_count, sender_addr)
            await self._send_notification(em, tag="RATE-LIMITED")
            return

        # All gates passed — create auto reply
        await self._create_auto_reply(em, chash)

    async def _send_notification(self, em: dict, tag: str = "EMAIL",
                                 followup: bool = False):
        """Send a Telegram notification about an email."""
        if not self._bot:
            return

        # Backward compat: followup param overrides tag
        if followup:
            tag = "FOLLOW-UP"

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
            log.warning("Email input: failed to send notification: %s", e)

    async def _send_reply_notification(self, em: dict, reply_body: str):
        """Send a [SENT] Telegram notification after auto-reply."""
        if not self._bot:
            return
        preview = reply_body[:300] + "..." if len(reply_body) > 300 else reply_body
        text = (
            f"[SENT] Auto-reply to {em['from']}\n"
            f"Re: {em['subject']}\n\n"
            f"{preview}"
        )
        try:
            await self._bot.send_message(
                chat_id=int(AUTHORIZED_USER_ID), text=text,
            )
        except Exception:
            pass

    async def _create_auto_reply(self, em: dict, chash: str):
        """Use the full engine pipeline to compose and send a reply autonomously.

        Same call_streaming path as Telegram/Discord — full tool access,
        per-sender session continuity, memory injection, and system prompts.
        """
        import subprocess
        from engine import call_streaming
        from drafts.queue import GMAIL_SCRIPT, GMAIL_IDENTITY

        body_preview = em["body"][:3000] if em.get("body") else "(no body)"

        # Pre-filter: skip exact terminal one-liners without burning a Claude call
        if _is_terminal(body_preview):
            log.info("Email from %s re: %s — terminal phrase, skipping",
                     em["from_addr"], em["subject"])
            await self._send_notification(em, tag="SKIPPED")
            return

        sender_addr = em["from_addr"] or em["from"]
        chat_id = _email_chat_id(sender_addr)
        lock = get_lock(chat_id)

        async with lock:
            session = get_session(chat_id)
            session_id = session["session_id"] if session else None
            model = session["model"] if session else "sonnet"
            msg_count = session["message_count"] if session else 0

            user_message = (
                f"[Email from {em['from']}]\n"
                f"Subject: {em['subject']}\n\n"
                f"{body_preview}"
            )

            extra_sp = EMAIL_SYSTEM_PROMPT_TEMPLATE.format(
                agent_name=AGENT_NAME,
                sender_name=em["from"],
                sender_addr=sender_addr,
                subject=em["subject"],
            )

            try:
                data = await call_streaming(
                    user_message, session_id, model,
                    streaming_editor=None,
                    extra_system_prompt=extra_sp,
                )
                reply_body = data.get("result", "").strip()
                new_session_id = data.get("session_id", session_id)
            except Exception as e:
                log.warning("Email input: engine call failed for %s: %s", em["from"], e)
                await self._send_notification(em)
                return

            if not reply_body:
                log.warning("Empty reply for %s re: %s", sender_addr, em["subject"])
                await self._send_notification(em)
                return

            # Persist session for sender continuity
            upsert_session(chat_id, new_session_id, model, msg_count + 1)

        # Send immediately — no approval gate
        try:
            reply_subject = em["subject"]
            if not reply_subject.lower().startswith("re:"):
                reply_subject = f"Re: {reply_subject}"

            loop = asyncio.get_event_loop()
            send_result = await loop.run_in_executor(None, lambda: subprocess.run(
                [
                    "python3", str(GMAIL_SCRIPT),
                    GMAIL_IDENTITY, "reply",
                    "--to", sender_addr,
                    "--subject", reply_subject,
                    "--id", em["id"],
                    "--body", reply_body,
                ],
                capture_output=True, text=True, timeout=30,
            ))

            if send_result.returncode == 0:
                log.info("Auto-replied to %s re: %s", sender_addr, em["subject"])
                # Record reply in audit trail (dedup + rate limiting)
                record_reply_sent(sender_addr, em["subject"], chash,
                                  reply_body[:200])
                await self._send_reply_notification(em, reply_body)
            else:
                error = send_result.stderr[:200] or send_result.stdout[:200]
                log.warning("Auto-reply send failed for %s: %s", sender_addr, error)
                await self._send_notification(em)
        except Exception as e:
            log.warning("Auto-reply error for %s: %s", sender_addr, e)
            await self._send_notification(em)
