"""
Alert Engine — translates scored XRPL transactions into named, human-readable alerts.

Sits on top of the ML scoring pipeline. For every incoming scored transaction,
call `engine.process_transaction(scored_tx)` to get a list of fired alerts.

Architecture:
    AlertEngine
        ├── WhaleRegistry   — knows which accounts are whales
        ├── TransactionBuffer — 10-min rolling window + cooldowns
        └── Detectors (one per alert group)
            ├── Group 1: Size & Whale Movements
            ├── Group 2: Burst & Bot Activity
            ├── Group 3: DEX & Trading Activity      [coming soon]
            ├── Group 4: Memo & Spam Detection       [coming soon]
            └── Group 5: Account Behaviour Shifts    [coming soon]

Usage:
    from whale_registry import WhaleRegistry
    from transaction_buffer import TransactionBuffer
    from alert_engine import AlertEngine

    registry = WhaleRegistry.build()
    buffer   = TransactionBuffer()
    engine   = AlertEngine(registry, buffer)

    alerts = engine.process_transaction(scored_tx)   # list[Alert]
    for alert in alerts:
        print(alert.message)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from whale_registry import WhaleRegistry
from transaction_buffer import TransactionBuffer

# ---------------------------------------------------------------------------
# Thresholds  (will be user-configurable in the final product)
# ---------------------------------------------------------------------------

# Group 1
LARGE_TX_PERCENTILE     = 0.99   # tx_size_percentile above which a tx is "large"
VOLUME_SPIKE_RATIO      = 3.0    # volume_spike_ratio above which we alert
HIGH_RISK_THRESHOLD     = 0.8    # risk_score above which we fire HIGH_RISK_TRANSACTION
CRITICAL_RISK_THRESHOLD = 0.9    # risk_score above which severity is critical

# Group 2
BURST_TX_COUNT          = 50     # rolling_tx_count_5m above which we fire TRANSACTION_BURST
BURST_Z_SCORE           = 3.0    # tx_rate_z_score above which we fire ABNORMAL_TX_RATE


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    id: str                        # "alert_{unix_ts}_{account[:8]}"
    timestamp: str                 # ISO 8601 UTC
    account: str                   # XRPL address (or "NETWORK" for network-wide alerts)
    alert_type: str                # e.g. "LARGE_XRP_TRANSFER"
    severity: str                  # "low" | "medium" | "high" | "critical"
    risk_score: float              # ML model score (0–1)
    is_anomaly: int                # ML model flag (0 or 1)
    message: str                   # plain-English one-liner shown to the trader
    details: dict = field(default_factory=dict)   # alert-specific context fields

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

def _severity(risk_score: float, is_anomaly: int) -> str:
    """Map ML scores to a severity label."""
    if risk_score >= CRITICAL_RISK_THRESHOLD and is_anomaly == 1:
        return "critical"
    if risk_score >= HIGH_RISK_THRESHOLD or is_anomaly == 1:
        return "high"
    if risk_score >= 0.4:
        return "medium"
    return "low"


def _bump_severity(severity: str) -> str:
    """Bump severity up one level (used when ML corroborates a rule-based alert)."""
    order = ["low", "medium", "high", "critical"]
    idx = order.index(severity)
    return order[min(idx + 1, len(order) - 1)]


def _make_id(account: str) -> str:
    return f"alert_{int(time.time())}_{account[:8]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt_amount(amount: float, currency: str) -> str:
    """Format a token amount for display in alert messages."""
    currency = currency or "XRP"
    if amount >= 1_000_000:
        return f"{amount / 1_000_000:.2f}M {currency}"
    if amount >= 1_000:
        return f"{amount / 1_000:.2f}K {currency}"
    return f"{amount:.2f} {currency}"


# ---------------------------------------------------------------------------
# AlertEngine
# ---------------------------------------------------------------------------

class AlertEngine:
    """Processes scored transactions and returns fired alerts."""

    def __init__(self, registry: WhaleRegistry, buffer: TransactionBuffer):
        self.registry = registry
        self.buffer   = buffer

    # ── public API ────────────────────────────────────────────────────────

    def process_transaction(self, scored_tx: dict) -> list[Alert]:
        """
        Main entry point. Call once per scored transaction.

        Parameters
        ----------
        scored_tx : dict
            A scored transaction from score_realtime.py. Must contain at minimum:
            timestamp, account, tx_type, amount_xrp, currency, destination,
            risk_score, is_anomaly, tx_size_percentile, is_large_tx,
            volume_spike_ratio, tx_rate_z_score, rolling_tx_count_5m.

        Returns
        -------
        list[Alert]
            All alerts fired for this transaction. Empty list if nothing triggered.
        """
        # Add to rolling buffer first so state queries reflect this tx
        self.buffer.add(scored_tx)

        alerts: list[Alert] = []

        # Group 1 — Size & Whale Movements
        alerts += self._check_large_transfer(scored_tx)
        alerts += self._check_volume_spike(scored_tx)
        alerts += self._check_high_risk(scored_tx)

        # Group 2 — Burst & Bot Activity
        alerts += self._check_transaction_burst(scored_tx)
        alerts += self._check_abnormal_tx_rate(scored_tx)

        # Groups 3–5 will be added here in subsequent steps

        return alerts

    # ── Group 1: Size & Whale Movements ──────────────────────────────────

    def _check_large_transfer(self, tx: dict) -> list[Alert]:
        """
        LARGE_XRP_TRANSFER — a whale moved an unusually large amount.

        Fires when:
          - Account is a known whale
          - Transaction is a Payment or OfferCreate
          - Amount is in the top 1% (is_large_tx=1 or tx_size_percentile >= 0.99)
          - Not on cooldown
        """
        account  = tx.get("account", "")
        tx_type  = tx.get("tx_type", "")
        currency = tx.get("currency") or "XRP"
        amount   = float(tx.get("amount_xrp") or 0)
        pct      = float(tx.get("tx_size_percentile") or 0)
        is_large = int(tx.get("is_large_tx") or 0)
        risk     = float(tx.get("risk_score") or 0)
        anomaly  = int(tx.get("is_anomaly") or 0)
        dest     = tx.get("destination") or ""

        # Gates
        if not self.registry.is_whale(account):
            return []
        if tx_type not in ("Payment", "OfferCreate"):
            return []
        if is_large != 1 and pct < LARGE_TX_PERCENTILE:
            return []
        if self.buffer.is_on_cooldown(account, "LARGE_XRP_TRANSFER"):
            return []

        # Severity — bump if ML also flags it
        severity = _severity(risk, anomaly)
        if risk >= 0.6:
            severity = _bump_severity(severity)

        amount_str = _fmt_amount(amount, currency)
        dest_short = f"{dest[:8]}..." if dest else "unknown"
        message = f"🐋 Whale moved {amount_str} → {dest_short}"

        alert = Alert(
            id=_make_id(account),
            timestamp=_now_iso(),
            account=account,
            alert_type="LARGE_XRP_TRANSFER",
            severity=severity,
            risk_score=risk,
            is_anomaly=anomaly,
            message=message,
            details={
                "amount": amount,
                "currency": currency,
                "destination": dest,
                "tx_size_percentile": round(pct, 4),
                "is_large_tx": is_large,
            },
        )

        self.buffer.set_cooldown(account, "LARGE_XRP_TRANSFER")
        return [alert]

    def _check_volume_spike(self, tx: dict) -> list[Alert]:
        """
        VOLUME_SPIKE — network-wide XRP volume spiked above normal.

        Fires when:
          - volume_spike_ratio >= 3.0 (current 5-min volume is 3× the historical average)
          - Not on network-level cooldown

        This is a network-level alert: account field is set to "NETWORK".
        Cooldown key is ("NETWORK", "VOLUME_SPIKE") so it fires at most once
        per cooldown window regardless of which account triggered it.
        """
        spike_ratio = float(tx.get("volume_spike_ratio") or 0)
        risk        = float(tx.get("risk_score") or 0)
        anomaly     = int(tx.get("is_anomaly") or 0)
        trigger_acc = tx.get("account", "")

        if spike_ratio < VOLUME_SPIKE_RATIO:
            return []
        if self.buffer.is_on_cooldown("NETWORK", "VOLUME_SPIKE"):
            return []

        severity = _severity(risk, anomaly)
        message = (
            f"📈 Network volume spike: {spike_ratio:.1f}× above average "
            f"in the last 5 min"
        )

        alert = Alert(
            id=_make_id("NETWORK"),
            timestamp=_now_iso(),
            account="NETWORK",
            alert_type="VOLUME_SPIKE",
            severity=severity,
            risk_score=risk,
            is_anomaly=anomaly,
            message=message,
            details={
                "volume_spike_ratio": round(spike_ratio, 2),
                "total_volume_5m": float(tx.get("total_volume_5m") or 0),
                "trigger_account": trigger_acc,
            },
        )

        self.buffer.set_cooldown("NETWORK", "VOLUME_SPIKE")
        return [alert]

    def _check_high_risk(self, tx: dict) -> list[Alert]:
        """
        HIGH_RISK_TRANSACTION — the ML model flagged this transaction as highly anomalous.

        Fires when:
          - risk_score >= 0.8  AND  is_anomaly = 1
          - Not on cooldown for this account

        No whale check — any account doing something extremely unusual is worth flagging.
        Severity is always at least "high"; "critical" if risk_score >= 0.9.
        """
        account = tx.get("account", "")
        risk    = float(tx.get("risk_score") or 0)
        anomaly = int(tx.get("is_anomaly") or 0)
        tx_type = tx.get("tx_type") or "unknown"

        if risk < HIGH_RISK_THRESHOLD or anomaly != 1:
            return []
        if self.buffer.is_on_cooldown(account, "HIGH_RISK_TRANSACTION"):
            return []

        severity = "critical" if risk >= CRITICAL_RISK_THRESHOLD else "high"
        message = (
            f"⚠️ High-risk transaction detected — "
            f"risk score {risk:.2f} ({tx_type})"
        )

        alert = Alert(
            id=_make_id(account),
            timestamp=_now_iso(),
            account=account,
            alert_type="HIGH_RISK_TRANSACTION",
            severity=severity,
            risk_score=risk,
            is_anomaly=anomaly,
            message=message,
            details={
                "tx_type": tx_type,
                "tx_rate_z_score": float(tx.get("tx_rate_z_score") or 0),
                "volume_spike_ratio": float(tx.get("volume_spike_ratio") or 0),
                "rolling_tx_count_5m": float(tx.get("rolling_tx_count_5m") or 0),
                "tx_size_percentile": float(tx.get("tx_size_percentile") or 0),
            },
        )

        self.buffer.set_cooldown(account, "HIGH_RISK_TRANSACTION")
        return [alert]

    # ── Group 2: Burst & Bot Activity ────────────────────────────────────

    def _check_transaction_burst(self, tx: dict) -> list[Alert]:
        """
        TRANSACTION_BURST — a whale is sending an abnormally high number of
        transactions in a short window.

        Fires when:
          - Account is a known whale
          - rolling_tx_count_5m >= BURST_TX_COUNT (50 txs in 5 min)
          - Not on cooldown

        Uses the rolling_tx_count_5m feature computed by the ML pipeline,
        which counts how many transactions this account sent in the last 5 min.
        This is a strong signal for bots, wash trading, or market manipulation.
        """
        account  = tx.get("account", "")
        tx_count = float(tx.get("rolling_tx_count_5m") or 0)
        risk     = float(tx.get("risk_score") or 0)
        anomaly  = int(tx.get("is_anomaly") or 0)

        if not self.registry.is_whale(account):
            return []
        if tx_count < BURST_TX_COUNT:
            return []
        if self.buffer.is_on_cooldown(account, "TRANSACTION_BURST"):
            return []

        severity = _severity(risk, anomaly)
        message = (
            f"⚡ Whale burst: {int(tx_count)} transactions in 5 min "
            f"from {account[:12]}..."
        )

        alert = Alert(
            id=_make_id(account),
            timestamp=_now_iso(),
            account=account,
            alert_type="TRANSACTION_BURST",
            severity=severity,
            risk_score=risk,
            is_anomaly=anomaly,
            message=message,
            details={
                "rolling_tx_count_5m": int(tx_count),
                "burst_threshold": BURST_TX_COUNT,
                "tx_per_minute": float(tx.get("tx_per_minute") or 0),
            },
        )

        self.buffer.set_cooldown(account, "TRANSACTION_BURST")
        return [alert]

    def _check_abnormal_tx_rate(self, tx: dict) -> list[Alert]:
        """
        ABNORMAL_TX_RATE — a whale's transaction rate is statistically extreme
        compared to all other accounts in the dataset.

        Fires when:
          - Account is a known whale
          - tx_rate_z_score >= BURST_Z_SCORE (3.0 standard deviations above mean)
          - Not on cooldown

        tx_rate_z_score is computed by the ML pipeline as:
            (account_tx_per_minute - network_mean) / network_std
        A z-score of 3.0 means the account is sending transactions at a rate
        that only ~0.1% of accounts would reach by chance.
        """
        account = tx.get("account", "")
        z_score = float(tx.get("tx_rate_z_score") or 0)
        rate    = float(tx.get("tx_per_minute") or 0)
        risk    = float(tx.get("risk_score") or 0)
        anomaly = int(tx.get("is_anomaly") or 0)

        if not self.registry.is_whale(account):
            return []
        if z_score < BURST_Z_SCORE:
            return []
        if self.buffer.is_on_cooldown(account, "ABNORMAL_TX_RATE"):
            return []

        severity = _severity(risk, anomaly)
        message = (
            f"🤖 Abnormal tx rate: {account[:12]}... sending at "
            f"{rate:.1f} tx/min — {z_score:.1f}× above network average"
        )

        alert = Alert(
            id=_make_id(account),
            timestamp=_now_iso(),
            account=account,
            alert_type="ABNORMAL_TX_RATE",
            severity=severity,
            risk_score=risk,
            is_anomaly=anomaly,
            message=message,
            details={
                "tx_rate_z_score": round(z_score, 3),
                "tx_per_minute": round(rate, 2),
                "z_score_threshold": BURST_Z_SCORE,
            },
        )

        self.buffer.set_cooldown(account, "ABNORMAL_TX_RATE")
        return [alert]
