"""hal-mail -- standalone email processing pod with per-recipient profiles.

Usage: python3 -m hal_mail
"""

import asyncio
import logging
import os
import signal
import sys

# Logging before anything else
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("hal-mail")


async def main():
    log.info("hal-mail starting...")

    # Initialize shared database (WAL mode handles concurrent access)
    from db import init_db

    init_db()

    # Load security policy
    from security.policy import load_policy, PolicyWatcher

    load_policy()

    # Load recipient profiles
    profiles_path = os.environ.get(
        "RECIPIENT_PROFILES_PATH", "/app/recipient_profiles.yaml"
    )
    from hal_mail.profiles import load_profiles, create_backends

    profiles = load_profiles(profiles_path)
    log.info(
        "Loaded %d recipient profiles + default",
        len(profiles.get("recipients", [])),
    )

    # Create backend instances (one per unique backend type in profiles)
    backends = create_backends(profiles)
    if not backends:
        log.error("No backends available -- cannot process emails")
        sys.exit(1)
    log.info("Backends ready: %s", ", ".join(sorted(backends.keys())))

    # Telegram bot for notifications (direct Bot API, no polling loop)
    from config import BOT_TOKEN
    from telegram import Bot

    bot = Bot(token=BOT_TOKEN)

    # Health server for K8s probes
    from hal_mail.health import HealthServer

    health = HealthServer()
    await health.start()

    # Create runner
    from hal_mail.runner import HalMailRunner

    runner = HalMailRunner(profiles, backends, bot, health)

    # Security policy hot-reload watcher
    policy_watcher = PolicyWatcher()
    policy_watcher_task = asyncio.create_task(policy_watcher.watch_loop())

    # Graceful shutdown
    shutdown_event = asyncio.Event()

    def handle_signal():
        log.info("Shutdown signal received")
        shutdown_event.set()
        runner.stop()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    # Start runner
    runner_task = asyncio.create_task(runner.run())

    dry_run = os.environ.get("DRY_RUN", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    log.info("hal-mail is running (dry_run=%s)", dry_run)

    try:
        await shutdown_event.wait()
    except KeyboardInterrupt:
        log.info("Keyboard interrupt")
    finally:
        log.info("hal-mail shutting down...")
        runner.stop()
        policy_watcher.stop()
        runner_task.cancel()
        policy_watcher_task.cancel()
        try:
            await runner_task
        except asyncio.CancelledError:
            pass
        try:
            await policy_watcher_task
        except asyncio.CancelledError:
            pass
        await health.stop()
        log.info("hal-mail stopped.")


if __name__ == "__main__":
    asyncio.run(main())
