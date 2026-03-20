#!/usr/bin/env python3
"""NEXUS — Multi-Channel Agentic Service.

Entry point that starts all subsystems:
  - Telegram channel (interactive bot)
  - Discord channel (interactive bot)
  - Scheduler (cron-like task runner)
  - Observer registry (cron-scheduled background observers)
  - Email input channel + draft queue
"""

import asyncio
import signal
import sys

from config import log
from db import init_db


def _build_observer_registry():
    """Create and populate the observer registry.

    Import and register all available observer classes here.
    Observers that fail to import are logged and skipped.
    """
    from observers.registry import ObserverRegistry

    registry = ObserverRegistry()

    observers = [
        # email_digest extracted to standalone K8s CronJob on arx1
        # ("observers.email_digest", "EmailDigestObserver"),
        # morning_brief extracted to standalone K8s CronJob on arx1
        # ("observers.morning_brief", "MorningBriefObserver"),
        # node_health disabled — alerting handled by Alertmanager on mon2
        # ("observers.node_health", "NodeHealthObserver"),
        # daily_snippet disabled — extracted to standalone K8s CronJob on arx2
        # ("observers.daily_snippet", "DailySnippetObserver"),
        # bretalon_review extracted to standalone K8s CronJob on arx1
        # ("observers.bretalon_review", "BretalonReviewObserver"),
        ("observers.git_push", "GitPushObserver"),
        ("observers.darwin_consumer", "DarwinConsumer"),
        ("observers.followup_reminder", "FollowupReminderObserver"),
        # alertmanager_monitor disabled — alerts suppressed from agent interface
        # ("observers.alertmanager_monitor", "AlertmanagerMonitorObserver"),
        # cyber_threat_feed extracted to standalone K8s CronJob on arx4
        # ("observers.cyber_threat_feed", "CyberThreatFeedObserver"),
        # intel_briefing disabled — replaced by intel_deep_analysis which
        # generates both full analysis articles and summary briefing cards
        # ("observers.intel_briefing", "IntelBriefingObserver"),
        # intel_deep_analysis extracted to standalone K8s CronJob on arx4
        # ("observers.intel_deep_analysis", "IntelDeepAnalysisObserver"),
        ("observers.memory_sync", "MemorySyncObserver"),
        # daily_report extracted to standalone K8s CronJob on fox-n1
        # ("observers.daily_report", "DailyReportObserver"),
        # doc_compiler extracted to standalone K8s CronJob on fox-n1
        # ("observers.doc_compiler", "DocCompilerObserver"),
        # weekly_report extracted to standalone K8s CronJob on fox-n1
        # ("observers.weekly_report", "WeeklyReportObserver"),
        # git_security_audit extracted to standalone K8s CronJob on arx3
        # ("observers.git_security_audit", "GitSecurityAuditObserver"),
        # git_auto_sync extracted to standalone K8s CronJob on arx3
        # ("observers.git_auto_sync", "GitAutoSyncObserver"),
        # github_activity extracted to standalone K8s CronJob on arx3
        # ("observers.github_activity", "GitHubActivityObserver"),
        ("observers.pipeline_watchdog", "PipelineWatchdog"),
        ("observers.heartbeat", "HeartbeatObserver"),
    ]

    import importlib
    for module_path, class_name in observers:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            registry.register(cls())
        except Exception as e:
            log.warning("Failed to register %s: %s", class_name, e)

    return registry


async def main():
    """Start NEXUS and run until interrupted."""
    log.info("NEXUS starting...")

    # Initialize database
    init_db()

    # Load security policy (after DB, before channels)
    from security.policy import load_policy, PolicyWatcher
    load_policy()

    # Import here to ensure config/db are ready
    from channels.telegram import TelegramChannel
    from config import DISCORD_BOT_TOKEN, WA_ENABLED, EMAIL_CHANNEL_ENABLED, TERMINAL_WS_ENABLED

    telegram = TelegramChannel()
    registry = _build_observer_registry()

    # Email input channel — can be disabled when hal-mail pod takes over
    email_in = None
    if EMAIL_CHANNEL_ENABLED:
        from channels.email_in import EmailInputChannel
        email_in = EmailInputChannel()
    else:
        log.info("Email channel disabled (EMAIL_CHANNEL_ENABLED=false)")

    # Discord — only start if token is configured
    discord_channel = None
    if DISCORD_BOT_TOKEN:
        from channels.discord import DiscordChannel
        discord_channel = DiscordChannel()

    # WhatsApp — only start if enabled
    wa_channel = None
    if WA_ENABLED:
        import json as _json
        from config import WA_INSTANCES, WA_ROUTING_CONFIG
        from channels.whatsapp import WhatsAppChannel

        try:
            instances = _json.loads(WA_INSTANCES)
        except Exception as e:
            log.warning("Failed to parse WA_INSTANCES: %s", e)
            instances = []

        wa_channel = WhatsAppChannel(
            instances=instances,
        )

    # Terminal WebSocket — only start if enabled
    terminal_channel = None
    if TERMINAL_WS_ENABLED:
        from channels.terminal import TerminalChannel
        terminal_channel = TerminalChannel()

    # Graceful shutdown handler
    shutdown_event = asyncio.Event()
    observer_task = None

    def handle_signal():
        log.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    # Start TC health probe
    from health_probes import get_probe
    health_probe_task = asyncio.create_task(get_probe().run_loop())

    # Start security policy hot-reload watcher
    policy_watcher = PolicyWatcher()
    policy_watcher_task = asyncio.create_task(policy_watcher.watch_loop())

    try:
        # Start Telegram channel
        await telegram.start()

        # Start email input channel (needs bot reference for notifications)
        if email_in:
            email_in._bot = telegram.app.bot
            await email_in.start()

        # Start Discord channel (if configured)
        if discord_channel:
            await discord_channel.start()

        # Start WhatsApp channel (if enabled)
        if wa_channel:
            wa_channel.set_telegram_bot(telegram.app.bot)
            await wa_channel.start()

            # Set global ref for Telegram callback handler (wa:approve/reject)
            from channels.telegram import callbacks as _tg_cb
            _tg_cb._wa_channel = wa_channel

            # Wire up the webhook handler: set wa_channel + event loop
            # on the GitPushObserver so WebhookHandler can dispatch
            for obs_instance in registry._observers:
                if hasattr(obs_instance, 'LISTEN_PORT'):  # GitPushObserver
                    obs_instance._wa_channel = wa_channel
                    obs_instance._event_loop = asyncio.get_event_loop()
                    break

        # Start Terminal WebSocket channel (if enabled)
        if terminal_channel:
            await terminal_channel.start()

        # Start observer registry loop
        observer_task = asyncio.create_task(registry.run_loop())

        log.info("NEXUS is running. Press Ctrl+C to stop.")

        # Wait for shutdown signal
        await shutdown_event.wait()

    except KeyboardInterrupt:
        log.info("Keyboard interrupt received")
    finally:
        log.info("NEXUS shutting down...")
        policy_watcher.stop()
        policy_watcher_task.cancel()
        try:
            await policy_watcher_task
        except asyncio.CancelledError:
            pass
        health_probe_task.cancel()
        try:
            await health_probe_task
        except asyncio.CancelledError:
            pass
        if observer_task:
            observer_task.cancel()
            try:
                await observer_task
            except asyncio.CancelledError:
                pass
        if terminal_channel:
            await terminal_channel.stop()
        if wa_channel:
            await wa_channel.stop()
        if discord_channel:
            await discord_channel.stop()
        if email_in:
            await email_in.stop()
        await telegram.stop()
        log.info("NEXUS stopped.")


if __name__ == "__main__":
    asyncio.run(main())
