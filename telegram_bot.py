"""
telegram_bot.py — XRPL Whale Watch Bot (@XRPL_Whale_WatchBot)

Users open the bot via a deep link from the frontend dashboard.
The bot sends real-time whale alerts directly to their Telegram DMs.

All navigation is done through inline keyboard buttons — no typing required.

Deep link for the frontend button:
    https://t.me/XRPL_Whale_WatchBot?start=1

Setup:
    export TELEGRAM_BOT_TOKEN="8531459075:AAGt2Dab1rW98hFajCC41pYBy1uSSNjVURs"

Usage:
    python3 telegram_bot.py

Dependencies:
    pip install python-telegram-bot
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BOT_TOKEN     = os.environ.get("TELEGRAM_BOT_TOKEN",
                               "8531459075:AAGt2Dab1rW98hFajCC41pYBy1uSSNjVURs")
ALERTS_PATH   = "data/alerts.json"
SUBS_PATH     = "data/subscribers.json"
POLL_INTERVAL = 3   # seconds between alert checks

SEVERITY_RANK  = {"low": 0, "medium": 1, "high": 2, "critical": 3}
VALID_SEVERITIES = ["low", "medium", "high", "critical"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("whale_bot")


# ---------------------------------------------------------------------------
# Subscriber store  (data/subscribers.json)
# ---------------------------------------------------------------------------
# Format: { "chat_id_str": {"severity": "high", "joined": "ISO timestamp"} }

def _load_subs() -> dict:
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(SUBS_PATH):
        return {}
    try:
        with open(SUBS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_subs(subs: dict) -> None:
    os.makedirs("data", exist_ok=True)
    with open(SUBS_PATH, "w") as f:
        json.dump(subs, f, indent=2)


def subscribe(chat_id: int, severity: str = "high") -> bool:
    """Add subscriber. Returns True if new, False if already existed."""
    subs = _load_subs()
    key  = str(chat_id)
    is_new = key not in subs
    subs[key] = {
        "severity": severity,
        "joined": datetime.now(timezone.utc).isoformat(),
    }
    _save_subs(subs)
    return is_new


def unsubscribe(chat_id: int) -> bool:
    """Remove subscriber. Returns True if they existed."""
    subs = _load_subs()
    key  = str(chat_id)
    if key in subs:
        del subs[key]
        _save_subs(subs)
        return True
    return False


def set_severity(chat_id: int, severity: str) -> None:
    subs = _load_subs()
    key  = str(chat_id)
    if key in subs:
        subs[key]["severity"] = severity
        _save_subs(subs)


def get_subscriber(chat_id: int) -> dict | None:
    return _load_subs().get(str(chat_id))


# ---------------------------------------------------------------------------
# Keyboards
# ---------------------------------------------------------------------------

def main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📊 My Status",     callback_data="menu:status"),
            InlineKeyboardButton("⚙️ Set Severity",  callback_data="menu:severity"),
        ],
        [
            InlineKeyboardButton("🔕 Unsubscribe",   callback_data="menu:stop"),
        ],
    ])


def severity_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔵 Low",      callback_data="sev:low"),
            InlineKeyboardButton("🟡 Medium",   callback_data="sev:medium"),
        ],
        [
            InlineKeyboardButton("🟠 High",     callback_data="sev:high"),
            InlineKeyboardButton("🔴 Critical", callback_data="sev:critical"),
        ],
        [InlineKeyboardButton("⬅️ Back", callback_data="menu:main")],
    ])


def confirm_stop_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Yes, unsubscribe", callback_data="stop:confirm"),
            InlineKeyboardButton("❌ Cancel",           callback_data="menu:main"),
        ],
    ])


def subscribe_keyboard() -> InlineKeyboardMarkup:
    """Shown to users who aren't subscribed yet."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🐋 Subscribe to Alerts", callback_data="menu:subscribe")],
    ])


# ---------------------------------------------------------------------------
# Message formatter
# ---------------------------------------------------------------------------

SEVERITY_EMOJI = {"low": "🔵", "medium": "🟡", "high": "🟠", "critical": "🔴"}

ALERT_EMOJI = {
    "LARGE_XRP_TRANSFER":      "🐋",
    "VOLUME_SPIKE":            "📈",
    "HIGH_RISK_TRANSACTION":   "⚠️",
    "TRANSACTION_BURST":       "⚡",
    "ABNORMAL_TX_RATE":        "🤖",
    "TOKEN_ACCUMULATION":      "🐋",
    "OFFER_CANCEL_SPIKE":      "👻",
    "MULTI_WHALE_CONVERGENCE": "🔥",
    "MEMO_SPAM":               "📨",
    "URL_SPAM":                "🔗",
    "HIGH_ENTROPY_MEMO":       "🔐",
    "WALLET_DRAIN":            "⚠️",
    "BEHAVIOUR_SHIFT":         "🔄",
    "NEW_WHALE_EMERGENCE":     "🆕",
}


def _short_addr(addr: str) -> str:
    if not addr or addr == "NETWORK":
        return addr
    return f"{addr[:6]}...{addr[-4:]}" if len(addr) > 12 else addr


def _relative_time(ts_str: str) -> str:
    try:
        ts    = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        secs  = int((datetime.now(timezone.utc) - ts).total_seconds())
        if secs < 60:   return f"{secs}s ago"
        if secs < 3600: return f"{secs // 60}m ago"
        return f"{secs // 3600}h ago"
    except Exception:
        return ""


def format_alert(alert: dict) -> str:
    alert_type = alert.get("alert_type", "UNKNOWN")
    severity   = alert.get("severity", "low")
    account    = alert.get("account", "")
    message    = alert.get("message", "")
    risk_score = alert.get("risk_score", 0.0)
    is_anomaly = alert.get("is_anomaly", 0)
    timestamp  = alert.get("timestamp", "")

    sev_emoji   = SEVERITY_EMOJI.get(severity, "⚪")
    alert_emoji = ALERT_EMOJI.get(alert_type, "🔔")
    time_str    = _relative_time(timestamp)
    anomaly_tag = " ⚠️ *anomaly*" if is_anomaly else ""

    lines = [
        f"{alert_emoji} *{alert_type}* — {sev_emoji} {severity.upper()}",
        f"",
        f"{message}",
        f"",
    ]
    if account and account != "NETWORK":
        lines.append(f"Account: `{_short_addr(account)}`")
    lines.append(f"Risk score: `{risk_score:.2f}`{anomaly_tag}")
    if time_str:
        lines.append(f"_{time_str}_")

    return "\n".join(lines)


def _status_text(chat_id: int) -> str:
    sub = get_subscriber(chat_id)
    if not sub:
        return (
            "You are *not subscribed*.\n\n"
            "Tap the button below to start receiving whale alerts."
        )
    sev      = sub["severity"]
    joined   = sub.get("joined", "")[:10]
    sev_icon = SEVERITY_EMOJI.get(sev, "⚪")
    return (
        f"✅ *Subscribed*\n\n"
        f"Severity filter: {sev_icon} *{sev.upper()}* and above\n"
        f"Joined: {joined}\n\n"
        f"You will receive alerts for: "
        + ", ".join(
            f"{SEVERITY_EMOJI[s]} {s}"
            for s in VALID_SEVERITIES
            if SEVERITY_RANK[s] >= SEVERITY_RANK[sev]
        )
    )


# ---------------------------------------------------------------------------
# Command & callback handlers
# ---------------------------------------------------------------------------

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id  = update.effective_chat.id
    is_new   = subscribe(chat_id, severity="high")

    if is_new:
        text = (
            "🐋 *Welcome to XRPL Whale Watch!*\n\n"
            "You're now subscribed to real-time whale alerts.\n"
            "By default you'll receive *HIGH* and *CRITICAL* alerts.\n\n"
            "Use the buttons below to manage your subscription."
        )
        log.info(f"New subscriber: {chat_id}")
    else:
        text = (
            "🐋 *XRPL Whale Watch*\n\n"
            "You're already subscribed. Use the buttons below to manage your alerts."
        )

    await update.message.reply_text(text, parse_mode="Markdown",
                                    reply_markup=main_menu_keyboard())


async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query   = update.callback_query
    chat_id = query.message.chat_id
    action  = query.data  # e.g. "menu:status", "menu:severity"

    await query.answer()

    if action == "menu:main":
        await query.edit_message_text(
            "🐋 *XRPL Whale Watch*\n\nWhat would you like to do?",
            parse_mode="Markdown",
            reply_markup=main_menu_keyboard(),
        )

    elif action == "menu:status":
        sub = get_subscriber(chat_id)
        kb  = main_menu_keyboard() if sub else subscribe_keyboard()
        await query.edit_message_text(
            _status_text(chat_id),
            parse_mode="Markdown",
            reply_markup=kb,
        )

    elif action == "menu:subscribe":
        subscribe(chat_id, severity="high")
        await query.edit_message_text(
            "✅ *Subscribed!*\n\nYou'll now receive HIGH and CRITICAL alerts.\nUse the buttons below to adjust.",
            parse_mode="Markdown",
            reply_markup=main_menu_keyboard(),
        )

    elif action == "menu:severity":
        sub = get_subscriber(chat_id)
        if not sub:
            await query.edit_message_text(
                "You need to subscribe first.",
                reply_markup=subscribe_keyboard(),
            )
            return
        current = sub["severity"]
        sev_icon = SEVERITY_EMOJI.get(current, "⚪")
        await query.edit_message_text(
            f"⚙️ *Set minimum severity*\n\nCurrent: {sev_icon} *{current.upper()}*\n\nChoose the minimum alert level you want to receive:",
            parse_mode="Markdown",
            reply_markup=severity_keyboard(),
        )

    elif action == "menu:stop":
        await query.edit_message_text(
            "🔕 *Unsubscribe?*\n\nYou will stop receiving all whale alerts.",
            parse_mode="Markdown",
            reply_markup=confirm_stop_keyboard(),
        )


async def severity_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query    = update.callback_query
    chat_id  = query.message.chat_id
    severity = query.data.split(":")[1]   # "sev:high" → "high"

    await query.answer(f"Severity set to {severity.upper()}")
    set_severity(chat_id, severity)

    sev_icon = SEVERITY_EMOJI.get(severity, "⚪")
    await query.edit_message_text(
        f"✅ *Severity updated*\n\nYou'll now receive {sev_icon} *{severity.upper()}* and above alerts.",
        parse_mode="Markdown",
        reply_markup=main_menu_keyboard(),
    )


async def stop_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query   = update.callback_query
    chat_id = query.message.chat_id

    await query.answer()
    unsubscribe(chat_id)
    await query.edit_message_text(
        "🔕 *Unsubscribed.*\n\nYou won't receive any more alerts.\n\n"
        "To resubscribe, tap the button below or use the link on the dashboard.",
        parse_mode="Markdown",
        reply_markup=subscribe_keyboard(),
    )
    log.info(f"Unsubscribed: {chat_id}")


# ---------------------------------------------------------------------------
# Alert watcher — runs as a background asyncio task
# ---------------------------------------------------------------------------

class AlertWatcher:
    def __init__(self):
        self._seen_ids: set[str] = set()
        self._seed()

    def _seed(self):
        for alert in self._load():
            self._seen_ids.add(alert.get("id", ""))
        log.info(f"Seeded {len(self._seen_ids)} existing alerts")

    def _load(self) -> list[dict]:
        if not os.path.exists(ALERTS_PATH):
            return []
        try:
            with open(ALERTS_PATH) as f:
                return json.load(f)
        except Exception:
            return []

    def poll(self) -> list[dict]:
        """Return new unseen alerts, oldest first."""
        new = []
        for alert in reversed(self._load()):
            aid = alert.get("id", "")
            if aid not in self._seen_ids:
                self._seen_ids.add(aid)
                new.append(alert)
        return new


async def alert_dispatch_loop(bot: Bot) -> None:
    """Background task: poll alerts.json and fan out to subscribers."""
    watcher = AlertWatcher()
    log.info("Alert dispatch loop started")

    while True:
        await asyncio.sleep(POLL_INTERVAL)
        try:
            new_alerts = watcher.poll()
            if not new_alerts:
                continue

            subs = _load_subs()
            if not subs:
                continue

            for alert in new_alerts:
                alert_rank = SEVERITY_RANK.get(alert.get("severity", "low"), 0)
                text       = format_alert(alert)

                for chat_id_str, prefs in subs.items():
                    min_rank = SEVERITY_RANK.get(prefs.get("severity", "high"), 2)
                    if alert_rank < min_rank:
                        continue
                    try:
                        await bot.send_message(
                            chat_id=int(chat_id_str),
                            text=text,
                            parse_mode="Markdown",
                        )
                        log.info(f"→ {chat_id_str}: {alert.get('alert_type')} [{alert.get('severity')}]")
                    except TelegramError as e:
                        log.error(f"Failed to send to {chat_id_str}: {e}")

        except Exception as e:
            log.error(f"Dispatch loop error: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    app = Application.builder().token(BOT_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CallbackQueryHandler(menu_callback,     pattern="^menu:"))
    app.add_handler(CallbackQueryHandler(severity_callback, pattern="^sev:"))
    app.add_handler(CallbackQueryHandler(stop_callback,     pattern="^stop:confirm$"))

    # Set bot command list (shown in Telegram menu)
    await app.bot.set_my_commands([
        ("start", "Subscribe to whale alerts"),
    ])

    log.info("Starting @XRPL_Whale_WatchBot ...")

    # Start polling and the alert dispatch loop concurrently
    async with app:
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)

        # Run alert dispatch as a background task
        dispatch_task = asyncio.create_task(alert_dispatch_loop(app.bot))

        log.info("Bot is running. Press Ctrl+C to stop.")
        try:
            await asyncio.Event().wait()   # run forever
        except asyncio.CancelledError:
            pass
        finally:
            dispatch_task.cancel()
            await app.updater.stop()
            await app.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Bot stopped.")
