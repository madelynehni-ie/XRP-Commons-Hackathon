"""
alerts_writer.py — Persistent alert storage

Maintains data/alerts.json as a rolling JSON array (newest first).
Capped at MAX_ALERTS entries to keep the file bounded.

Usage:
    from alerts_writer import write_alert, read_alerts
    write_alert(alert)          # append one Alert
    alerts = read_alerts()      # list of dicts, newest first
"""

import json
import os
import tempfile
import threading
from alert_engine import Alert

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUT_DIR = "data"
ALERTS_PATH = os.path.join(OUT_DIR, "alerts.json")
MAX_ALERTS = 500          # rolling cap — oldest entries are dropped

_lock = threading.Lock()  # one writer at a time


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_alert(alert: Alert) -> None:
    """Append an alert to alerts.json (thread-safe, atomic write)."""
    os.makedirs(OUT_DIR, exist_ok=True)

    with _lock:
        existing = _load()
        existing.insert(0, alert.to_dict())   # newest first
        if len(existing) > MAX_ALERTS:
            existing = existing[:MAX_ALERTS]
        _save(existing)


def read_alerts(limit: int = MAX_ALERTS, severity: str | None = None) -> list[dict]:
    """Return up to `limit` alerts, optionally filtered by severity level."""
    alerts = _load()
    if severity:
        alerts = [a for a in alerts if a.get("severity") == severity]
    return alerts[:limit]


def clear_alerts() -> None:
    """Wipe alerts.json (useful for testing)."""
    with _lock:
        _save([])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load() -> list[dict]:
    if not os.path.exists(ALERTS_PATH):
        return []
    try:
        with open(ALERTS_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save(data: list[dict]) -> None:
    """Write atomically — write to temp file then rename."""
    dir_ = os.path.dirname(ALERTS_PATH)
    os.makedirs(dir_, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, ALERTS_PATH)
    except Exception:
        os.unlink(tmp_path)
        raise
