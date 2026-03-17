"""
telegram_bot.py — Push notification bot for XRPL Whale Monitor

Watches data/alerts.json for new alerts and forwards them to a Telegram
channel or group chat. Runs as a standalone process alongside score_realtime.py.

Setup:
    1. Create a bot via @BotFather on Telegram → get TELEGRAM_BOT_TOKEN
    2. Add the bot to your channel/group and get the chat ID → TELEGRAM_CHAT_ID
    3. Set env vars (or create a .env file):

        TELEGRAM_BOT_TOKEN=123456:ABC-your-token
        TELEGRAM_CHAT_ID=-1001234567890
        TELEGRAM_MIN_SEVERITY=high   # low | medium | high | critical (default: high)

Usage:
    python3 telegram_bot.py

Dependencies:
    pip install python-telegram-bot
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone, timedelta

from telegram import Bot
from telegram.error import TelegramError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BOT_TOKEN    = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID      = os.environ.get("TELEGRAM_CHAT_ID", "")
MIN_SEVERITY = os.environ.get("TELEGRAM_MIN_SEVERITY", "high").lower()
ALERTS_PATH  = "data/alerts.json"
POLL_INTERVAL = 3   # seconds between checks

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("whale_bot")


# ---------------------------------------------------------------------------
# Message formatter
# ---------------------------------------------------------------------------

SEVERITY_EMOJI = {
    "low":      "🔵",
    "medium":   "🟡",
    "high":     "🟠",
    "critical": "🔴",
}

ALERT_EMOJI = {
    "LARGE_XRP_TRANSFER":       "🐋",
    "VOLUME_SPIKE":             "📈",
    "HIGH_RISK_TRANSACTION":    "⚠️",
    "TRANSACTION_BURST":        "⚡",
    "ABNORMAL_TX_RATE":         "🤖",
    "TOKEN_ACCUMULATION":       "🐋",
    "OFFER_CANCEL_SPIKE":       "👻",
    "MULTI_WHALE_CONVERGENCE":  "🔥",
    "MEMO_SPAM":                "📨",
    "URL_SPAM":                 "🔗",
    "HIGH_ENTROPY_MEMO":        "🔐",
    "WALLET_DRAIN":             "⚠️",
    "BEHAVIOUR_SHIFT":          "🔄",
    "NEW_WHALE_EMERGENCE":      "🆕",
}


def _short_addr(addr: str) -> str:
    """Truncate an XRPL address to rXXXX...XXXX format."""
    if not addr or addr == "NETWORK":
        return addr
    return f"{addr[:6]}...{addr[-4:]}" if len(addr) > 12 else addr


def _relative_time(ts_str: str) -> str:
    """Convert ISO timestamp to 'X min ago' string."""
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - ts
        secs = int(delta.total_seconds())
        if secs < 60:
            return f"{secs}s ago"
        if secs < 3600:
            return f"{secs // 60}m ago"
        return f"{secs // 3600}h ago"
    except Exception:
        return ""


def format_alert(alert: dict) -> str:
    """Format an alert dict into a Telegram Markdown v2 message."""
    alert_type = alert.get("alert_type", "UNKNOWN")
    severity   = alert.get("severity", "low")
    account    = alert.get("account", "")
    message    = alert.get("message", "")
    risk_score = alert.get("risk_score", 0.0)
    is_anomaly = alert.get("is_anomaly", 0)
    timestamp  = alert.get("timestamp", "")

    sev_emoji   = SEVERITY_EMOJI.get(severity, "⚪")
    alert_emoji = ALERT_EMOJI.get(alert_type, "🔔")
    sev_label   = severity.upper()
    time_str    = _relative_time(timestamp)
    addr_str    = _short_addr(account)
    anomaly_tag = " ⚠️ *anomaly*" if is_anomaly else ""

    lines = [
        f"{alert_emoji} *{alert_type}* — {sev_emoji} {sev_label}",
        f"",
        f"{message}",
        f"",
    ]

    if account and account != "NETWORK":
        lines.append(f"Account: `{addr_str}`")

    lines.append(f"Risk score: `{risk_score:.2f}`{anomaly_tag}")

    if time_str:
        lines.append(f"Time: {time_str}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Alert watcher
# ---------------------------------------------------------------------------

class AlertWatcher:
    """Polls alerts.json and yields new alerts that pass the severity filter."""

    def __init__(self, min_severity: str):
        self._min_rank   = SEVERITY_RANK.get(min_severity, 2)
        self._seen_ids: set[str] = set()
        # Seed with whatever is already in the file so we don't
        # flood the channel with old alerts on startup
        self._seed()

    def _seed(self):
        """Mark all existing alerts as already seen."""
        for alert in self._load():
            self._seen_ids.add(alert.get("id", ""))
        log.info(f"Seeded {len(self._seen_ids)} existing alerts (will not resend)")

    def _load(self) -> list[dict]:
        if not os.path.exists(ALERTS_PATH):
            return []
        try:
            with open(ALERTS_PATH) as f:
                return json.load(f)
        except Exception:
            return []

    def _passes_filter(self, alert: dict) -> bool:
        sev = alert.get("severity", "low")
        return SEVERITY_RANK.get(sev, 0) >= self._min_rank

    def poll(self) -> list[dict]:
        """Return new unseen alerts that pass the severity filter, oldest first."""
        all_alerts = self._load()
        new = []
        for alert in reversed(all_alerts):   # reversed = oldest first
            aid = alert.get("id", "")
            if aid not in self._seen_ids:
                self._seen_ids.add(aid)
                if self._passes_filter(alert):
                    new.append(alert)
        return new


# ---------------------------------------------------------------------------
# Bot runner
# ---------------------------------------------------------------------------

async def send_message(bot: Bot, text: str) -> bool:
    """Send a Markdown message — returns True on success."""
    try:
        await bot.send_message(
            chat_id=CHAT_ID,
            text=text,
            parse_mode="Markdown",
        )
        return True
    except TelegramError as e:
        log.error(f"Failed to send message: {e}")
        return False


async def run():
    if not BOT_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN is not set. Export it and restart.")
        return
    if not CHAT_ID:
        log.error("TELEGRAM_CHAT_ID is not set. Export it and restart.")
        return

    bot     = Bot(token=BOT_TOKEN)
    watcher = AlertWatcher(min_severity=MIN_SEVERITY)

    # Startup message
    startup_msg = (
        f"🐋 *XRPL Whale Monitor is live*\n\n"
        f"Watching for `{MIN_SEVERITY.upper()}` and above alerts\n"
        f"Polling every {POLL_INTERVAL}s · {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
    )
    await send_message(bot, startup_msg)
    log.info(f"Bot started — forwarding {MIN_SEVERITY}+ alerts to chat {CHAT_ID}")

    while True:
        await asyncio.sleep(POLL_INTERVAL)
        try:
            new_alerts = watcher.poll()
            for alert in new_alerts:
                text = format_alert(alert)
                ok   = await send_message(bot, text)
                if ok:
                    log.info(f"Sent: {alert.get('alert_type')} [{alert.get('severity')}]")
        except Exception as e:
            log.error(f"Watcher error: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("Bot stopped.")
