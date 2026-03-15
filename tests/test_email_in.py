"""Tests for channels/email_in.py — EmailInputChannel.

Covers:
- IMAP email fetching (mocked)
- Email classification routing (ignore/notify/auto_reply)
- Draft creation for auto_reply
- Telegram notifications for notify
- Deduplication via email_seen
- Header decoding and body extraction
"""

import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

with patch.dict("os.environ", {
    "TELEGRAM_BOT_TOKEN": "fake:token",
    "AUTHORIZED_USER_ID": "12345",
    "VIP_SENDERS": "vip-user@example.com,ops@puretensor.ai,vip-user@example.com",
}):
    from channels.email_in import (
        EmailInputChannel,
        fetch_new_emails,
        _decode_header,
        _extract_email_addr,
        _get_body,
        _email_chat_id,
        EMAIL_CHAT_ID_OFFSET,
    )
    from db import init_db, is_email_seen, mark_email_seen, get_session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Fresh temp database for each test."""
    db_file = tmp_path / "test_email_in.db"
    monkeypatch.setattr("db.DB_PATH", db_file)
    monkeypatch.setattr("drafts.queue.AUTHORIZED_USER_ID", 12345)
    init_db()
    yield db_file


def _recent_date_raw():
    """Return an RFC 2822 date string for ~5 minutes ago (always within the 24h age gate)."""
    from email.utils import format_datetime
    return format_datetime(datetime.now(timezone.utc))


@pytest.fixture
def accounts_file(tmp_path, monkeypatch):
    """Create a temporary email accounts file with primary role."""
    accts = [
        {
            "name": "test-acct",
            "server": "imap.example.com",
            "port": 993,
            "username": "test@example.com",
            "password": "secret",
            "role": "primary",
        }
    ]
    path = tmp_path / "email_accounts.json"
    path.write_text(json.dumps(accts))
    monkeypatch.setattr("channels.email_in.ACCOUNTS_FILE", path)
    return path


# ---------------------------------------------------------------------------
# Header decoding helpers
# ---------------------------------------------------------------------------


class TestDecodeHeader:

    def test_plain_header(self):
        assert _decode_header("Hello World") == "Hello World"

    def test_empty_header(self):
        assert _decode_header("") == ""

    def test_none_header(self):
        assert _decode_header(None) == ""


class TestExtractEmailAddr:

    def test_bare_address(self):
        assert _extract_email_addr("user@example.com") == "user@example.com"

    def test_display_name_with_angle_brackets(self):
        assert _extract_email_addr("John Doe <john@example.com>") == "john@example.com"

    def test_empty_string(self):
        assert _extract_email_addr("") == ""

    def test_none_value(self):
        assert _extract_email_addr(None) == ""


class TestGetBody:

    def test_plain_text_message(self):
        msg = MagicMock()
        msg.is_multipart.return_value = False
        msg.get_payload.return_value = b"Hello plain text"
        msg.get_content_charset.return_value = "utf-8"
        assert _get_body(msg) == "Hello plain text"

    def test_empty_payload(self):
        msg = MagicMock()
        msg.is_multipart.return_value = False
        msg.get_payload.return_value = None
        assert _get_body(msg) == ""


# ---------------------------------------------------------------------------
# EmailInputChannel — poll_once routing
# ---------------------------------------------------------------------------


class TestPollOnceClassification:

    @pytest.mark.asyncio
    async def test_ignore_skips_notification(self, accounts_file):
        """Emails classified as 'ignore' should not send any notification."""
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()

        channel = EmailInputChannel(bot=mock_bot)

        fake_email = {
            "id": "msg-ignore-1",
            "from": "noreply@example.com",
            "from_addr": "noreply@example.com",
            "subject": "Your order shipped",
            "date": "Feb 10 12:00",
            "date_raw": _recent_date_raw(),
            "to": "ops@puretensor.ai",
            "body": "Your package is on the way.",
        }

        with patch("channels.email_in.fetch_new_emails", return_value=[fake_email]):
            await channel._poll_once()

        # Should not send notification for ignored emails
        mock_bot.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_notify_sends_telegram(self, accounts_file):
        """Emails classified as 'notify' should send a Telegram notification."""
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()

        channel = EmailInputChannel(bot=mock_bot)

        fake_email = {
            "id": "msg-notify-1",
            "from": "receipts@stripe.com",
            "from_addr": "receipts@stripe.com",
            "subject": "Payment receipt",
            "date": "Feb 10 12:00",
            "date_raw": _recent_date_raw(),
            "to": "ops@puretensor.ai",
            "body": "You received a payment of $100.",
        }

        with patch("channels.email_in.fetch_new_emails", return_value=[fake_email]):
            await channel._poll_once()

        mock_bot.send_message.assert_awaited_once()
        call_kwargs = mock_bot.send_message.call_args[1]
        assert call_kwargs["chat_id"] == 12345
        assert "stripe.com" in call_kwargs["text"]

    @pytest.mark.asyncio
    async def test_auto_reply_sends_directly(self, accounts_file, monkeypatch):
        """Emails classified as 'auto_reply' should use call_streaming and send reply directly."""
        monkeypatch.setattr("drafts.classifier.VIP_SENDERS",
                            ["vip-user@example.com", "ops@puretensor.ai"])

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()

        channel = EmailInputChannel(bot=mock_bot)

        fake_email = {
            "id": "msg-auto-1",
            "from": "VIP User <vip-user@example.com>",
            "from_addr": "vip-user@example.com",
            "subject": "Report feedback",
            "date": "Feb 10 12:00",
            "date_raw": _recent_date_raw(),
            "to": "hal@example.com",
            "body": "The report looks good, please update section 3.",
        }

        mock_run = MagicMock()
        mock_run.returncode = 0

        mock_streaming = AsyncMock(return_value={
            "result": "Thank you for the feedback, Alan.",
            "session_id": "sess-email-001",
            "written_files": [],
        })

        with patch("channels.email_in.fetch_new_emails", return_value=[fake_email]), \
             patch("engine.call_streaming", mock_streaming), \
             patch("subprocess.run", return_value=mock_run):
            await channel._poll_once()

        # Should have sent a [SENT] notification via Telegram
        mock_bot.send_message.assert_called_once()
        call_kwargs = mock_bot.send_message.call_args[1]
        assert "[SENT]" in call_kwargs["text"]
        assert "vip-user@example.com" in call_kwargs["text"]

    @pytest.mark.asyncio
    async def test_auto_reply_fallback_on_engine_failure(self, accounts_file, monkeypatch):
        """If call_streaming fails, auto_reply should fall back to notification."""
        monkeypatch.setattr("drafts.classifier.VIP_SENDERS",
                            ["vip-user@example.com", "ops@puretensor.ai"])

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()

        channel = EmailInputChannel(bot=mock_bot)

        fake_email = {
            "id": "msg-auto-fail",
            "from": "ops@puretensor.ai",
            "from_addr": "ops@puretensor.ai",
            "subject": "Server question",
            "date": "Feb 10 12:00",
            "date_raw": _recent_date_raw(),
            "to": "hal@example.com",
            "body": "What is the status of tensor-core?",
        }

        mock_streaming = AsyncMock(side_effect=RuntimeError("Engine unavailable"))

        with patch("channels.email_in.fetch_new_emails", return_value=[fake_email]), \
             patch("engine.call_streaming", mock_streaming):
            await channel._poll_once()

        # Should fall back to notification
        mock_bot.send_message.assert_awaited_once()
        call_kwargs = mock_bot.send_message.call_args[1]
        assert "ops@puretensor.ai" in call_kwargs["text"]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:

    @pytest.mark.asyncio
    async def test_already_seen_email_skipped(self, accounts_file):
        """Emails already marked as seen should be skipped."""
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()

        # Pre-mark as seen
        mark_email_seen("msg-dup-1", "test-acct")

        channel = EmailInputChannel(bot=mock_bot)

        fake_email = {
            "id": "msg-dup-1",
            "from": "receipts@stripe.com",
            "from_addr": "receipts@stripe.com",
            "subject": "Payment receipt",
            "date": "Feb 10 12:00",
            "date_raw": _recent_date_raw(),
            "to": "ops@puretensor.ai",
            "body": "Payment.",
        }

        with patch("channels.email_in.fetch_new_emails", return_value=[fake_email]):
            await channel._poll_once()

        # Should not send notification for already-seen email
        mock_bot.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_new_email_marked_seen(self, accounts_file):
        """Processing a new email should mark it as seen."""
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()

        channel = EmailInputChannel(bot=mock_bot)

        fake_email = {
            "id": "msg-new-1",
            "from": "billing@provider.com",
            "from_addr": "billing@provider.com",
            "subject": "Your invoice",
            "date": "Feb 10 12:00",
            "date_raw": _recent_date_raw(),
            "to": "ops@puretensor.ai",
            "body": "Invoice attached.",
        }

        with patch("channels.email_in.fetch_new_emails", return_value=[fake_email]):
            await channel._poll_once()

        assert is_email_seen("msg-new-1") is True


# ---------------------------------------------------------------------------
# Channel start/stop
# ---------------------------------------------------------------------------


class TestChannelLifecycle:

    @pytest.mark.asyncio
    async def test_start_creates_task(self):
        """start() should create a polling task."""
        channel = EmailInputChannel()
        with patch.object(channel, "_poll_loop", new_callable=AsyncMock):
            await channel.start()
            assert channel._task is not None
            # Clean up
            await channel.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        """stop() should cancel the polling task."""
        channel = EmailInputChannel()
        with patch.object(channel, "_poll_loop", new_callable=AsyncMock):
            await channel.start()
            task = channel._task
            await channel.stop()
            assert task.cancelled()


# ---------------------------------------------------------------------------
# No accounts file
# ---------------------------------------------------------------------------


class TestNoAccountsFile:

    @pytest.mark.asyncio
    async def test_poll_with_no_accounts_file(self, tmp_path, monkeypatch):
        """If accounts file doesn't exist, poll_once should return silently."""
        monkeypatch.setattr(
            "channels.email_in.ACCOUNTS_FILE",
            tmp_path / "nonexistent.json",
        )
        channel = EmailInputChannel()
        # Should not raise
        await channel._poll_once()


# ---------------------------------------------------------------------------
# Notification formatting
# ---------------------------------------------------------------------------


class TestNotificationFormatting:

    @pytest.mark.asyncio
    async def test_notify_includes_from_and_subject(self, accounts_file):
        """Notification text should include sender and subject."""
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()

        channel = EmailInputChannel(bot=mock_bot)

        fake_email = {
            "id": "msg-fmt-1",
            "from": "security@bank.com",
            "from_addr": "security@bank.com",
            "subject": "Activity alert",
            "date": "Feb 10 14:30",
            "date_raw": _recent_date_raw(),
            "to": "ops@puretensor.ai",
            "body": "Unusual login detected from a new device.",
        }

        with patch("channels.email_in.fetch_new_emails", return_value=[fake_email]):
            await channel._poll_once()

        text = mock_bot.send_message.call_args[1]["text"]
        assert "security@bank.com" in text
        assert "Activity alert" in text
        assert "[EMAIL]" in text

    @pytest.mark.asyncio
    async def test_no_bot_no_notification(self, accounts_file):
        """If no bot is set, notification should be silently skipped."""
        channel = EmailInputChannel(bot=None)

        fake_email = {
            "id": "msg-nobot-1",
            "from": "billing@provider.com",
            "from_addr": "billing@provider.com",
            "subject": "Invoice",
            "date": "Feb 10 12:00",
            "date_raw": _recent_date_raw(),
            "to": "ops@puretensor.ai",
            "body": "Invoice attached.",
        }

        with patch("channels.email_in.fetch_new_emails", return_value=[fake_email]):
            # Should not raise even with no bot
            await channel._poll_once()


# ---------------------------------------------------------------------------
# Email chat ID helper
# ---------------------------------------------------------------------------


class TestEmailChatId:

    def test_deterministic(self):
        """Same email should always produce the same chat ID."""
        id1 = _email_chat_id("user@example.com")
        id2 = _email_chat_id("user@example.com")
        assert id1 == id2

    def test_case_insensitive(self):
        """Email addresses are case-insensitive."""
        assert _email_chat_id("User@Example.COM") == _email_chat_id("user@example.com")

    def test_offset_applied(self):
        """Chat ID should be above the EMAIL_CHAT_ID_OFFSET."""
        cid = _email_chat_id("test@test.com")
        assert cid >= EMAIL_CHAT_ID_OFFSET

    def test_different_senders_different_ids(self):
        """Different senders should (almost certainly) get different IDs."""
        id1 = _email_chat_id("alice@example.com")
        id2 = _email_chat_id("bob@example.com")
        assert id1 != id2


# ---------------------------------------------------------------------------
# Session persistence via auto_reply
# ---------------------------------------------------------------------------


class TestEmailSessions:

    @pytest.mark.asyncio
    async def test_auto_reply_creates_session(self, accounts_file, monkeypatch):
        """After a successful auto-reply, a session should be persisted for the sender."""
        monkeypatch.setattr("drafts.classifier.VIP_SENDERS",
                            ["vip-user@example.com", "ops@puretensor.ai"])

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()

        channel = EmailInputChannel(bot=mock_bot)

        fake_email = {
            "id": "msg-session-1",
            "from": "VIP User <vip-user@example.com>",
            "from_addr": "vip-user@example.com",
            "subject": "Quick question",
            "date": "Mar 02 10:00",
            "date_raw": _recent_date_raw(),
            "to": "hal@example.com",
            "body": "Hey HAL, what's the cluster status?",
        }

        mock_run = MagicMock()
        mock_run.returncode = 0

        mock_streaming = AsyncMock(return_value={
            "result": "All systems nominal. The cluster is healthy.",
            "session_id": "sess-email-alan-001",
            "written_files": [],
        })

        with patch("channels.email_in.fetch_new_emails", return_value=[fake_email]), \
             patch("engine.call_streaming", mock_streaming), \
             patch("subprocess.run", return_value=mock_run):
            await channel._poll_once()

        # Verify session was persisted
        chat_id = _email_chat_id("vip-user@example.com")
        session = get_session(chat_id)
        assert session is not None
        assert session["session_id"] == "sess-email-alan-001"
        assert session["message_count"] == 1

    @pytest.mark.asyncio
    async def test_auto_reply_resumes_session(self, accounts_file, monkeypatch):
        """Second email from same sender should pass existing session_id to call_streaming."""
        monkeypatch.setattr("drafts.classifier.VIP_SENDERS",
                            ["vip-user@example.com", "ops@puretensor.ai"])

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()

        channel = EmailInputChannel(bot=mock_bot)

        fake_email_1 = {
            "id": "msg-resume-1",
            "from": "ops@puretensor.ai",
            "from_addr": "ops@puretensor.ai",
            "subject": "First question",
            "date": "Mar 02 10:00",
            "date_raw": _recent_date_raw(),
            "to": "hal@example.com",
            "body": "How's the GPU utilisation?",
        }
        fake_email_2 = {
            "id": "msg-resume-2",
            "from": "ops@puretensor.ai",
            "from_addr": "ops@puretensor.ai",
            "subject": "Follow-up",
            "date": "Mar 02 10:05",
            "date_raw": _recent_date_raw(),
            "to": "hal@example.com",
            "body": "And what about memory usage?",
        }

        mock_run = MagicMock()
        mock_run.returncode = 0

        # First call returns a new session
        mock_streaming = AsyncMock(side_effect=[
            {"result": "GPU at 45%.", "session_id": "sess-ops-001", "written_files": []},
            {"result": "Memory at 60%.", "session_id": "sess-ops-001", "written_files": []},
        ])

        with patch("engine.call_streaming", mock_streaming), \
             patch("subprocess.run", return_value=mock_run):
            # First email
            with patch("channels.email_in.fetch_new_emails", return_value=[fake_email_1]):
                await channel._poll_once()
            # Second email
            with patch("channels.email_in.fetch_new_emails", return_value=[fake_email_2]):
                await channel._poll_once()

        # Second call should have received the session_id from first call
        assert mock_streaming.call_count == 2
        second_call_args = mock_streaming.call_args_list[1]
        assert second_call_args[0][1] == "sess-ops-001"  # session_id positional arg

        # Session should show 2 messages
        chat_id = _email_chat_id("ops@puretensor.ai")
        session = get_session(chat_id)
        assert session["message_count"] == 2
