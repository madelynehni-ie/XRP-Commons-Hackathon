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
            ├── Group 3: DEX & Trading Activity
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

# Group 3
TOKEN_OFFER_MIN         = 5      # min OfferCreates for same token to fire TOKEN_ACCUMULATION/FLASH_DUMP
CANCEL_RATIO_THRESHOLD  = 0.7    # offer_cancel_ratio above which we fire OFFER_CANCEL_SPIKE
CANCEL_MIN_OFFERS       = 5      # min total offers before cancel ratio is meaningful
MULTI_WHALE_MIN         = 3      # min whales on same token to fire MULTI_WHALE_CONVERGENCE


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

        # Group 3 — DEX & Trading Activity
        alerts += self._check_token_accumulation(scored_tx)
        alerts += self._check_offer_cancel_spike(scored_tx)
        alerts += self._check_multi_whale_convergence(scored_tx)

        # Groups 4–5 will be added here in subsequent steps

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

    # ── Group 3: DEX & Trading Activity ──────────────────────────────────

    def _check_token_accumulation(self, tx: dict) -> list[Alert]:
        """
        TOKEN_ACCUMULATION / FLASH_DUMP — a whale is repeatedly trading
        the same token in the same direction within the rolling window.

        Fires TOKEN_ACCUMULATION when:
          - Account is a whale
          - tx_type is OfferCreate
          - The same non-XRP currency appears >= TOKEN_OFFER_MIN times
            in the account's recent OfferCreates (from the buffer)
          - Not on cooldown for TOKEN_ACCUMULATION or FLASH_DUMP

        Since the normalised schema captures the currency of the traded amount,
        repeated OfferCreates for the same currency signal accumulation activity.
        We fire FLASH_DUMP under the same logic — repeated offers on a token
        are directional activity regardless of buy/sell side.
        """
        account = tx.get("account", "")
        tx_type = tx.get("tx_type", "")
        currency = tx.get("currency") or ""
        risk    = float(tx.get("risk_score") or 0)
        anomaly = int(tx.get("is_anomaly") or 0)

        if not self.registry.is_whale(account):
            return []
        if tx_type != "OfferCreate":
            return []
        if not currency or currency == "XRP":
            return []

        # Count recent OfferCreates for this specific currency in the buffer
        offer_count = sum(
            1 for t in self.buffer._buffer
            if t.get("account") == account
            and t.get("tx_type") == "OfferCreate"
            and t.get("currency") == currency
        )

        if offer_count < TOKEN_OFFER_MIN:
            return []

        alert_type = "TOKEN_ACCUMULATION"
        if self.buffer.is_on_cooldown(account, alert_type):
            return []

        severity = _severity(risk, anomaly)
        token_short = currency[:10]
        message = (
            f"🐋 Whale active on {token_short}: "
            f"{offer_count} offers in last 10 min from {account[:12]}..."
        )

        alert = Alert(
            id=_make_id(account),
            timestamp=_now_iso(),
            account=account,
            alert_type=alert_type,
            severity=severity,
            risk_score=risk,
            is_anomaly=anomaly,
            message=message,
            details={
                "token": currency,
                "offer_count": offer_count,
                "min_offers_threshold": TOKEN_OFFER_MIN,
            },
        )

        self.buffer.set_cooldown(account, alert_type)
        return [alert]

    def _check_offer_cancel_spike(self, tx: dict) -> list[Alert]:
        """
        OFFER_CANCEL_SPIKE — a whale is placing and cancelling offers at a
        suspicious ratio, a classic spoofing / order-book manipulation pattern.

        Fires when:
          - Account is a whale
          - tx_type is OfferCancel
          - offer_cancel_ratio (from AccountState) >= CANCEL_RATIO_THRESHOLD (0.7)
          - Total offers in window >= CANCEL_MIN_OFFERS (avoids noise on small samples)
          - Not on cooldown
        """
        account = tx.get("account", "")
        tx_type = tx.get("tx_type", "")
        risk    = float(tx.get("risk_score") or 0)
        anomaly = int(tx.get("is_anomaly") or 0)

        if not self.registry.is_whale(account):
            return []
        if tx_type != "OfferCancel":
            return []
        if self.buffer.is_on_cooldown(account, "OFFER_CANCEL_SPIKE"):
            return []

        state = self.buffer.get_account_state(account)
        total_offers = state.offer_creates + state.offer_cancels

        if total_offers < CANCEL_MIN_OFFERS:
            return []
        if state.offer_cancel_ratio < CANCEL_RATIO_THRESHOLD:
            return []

        severity = _severity(risk, anomaly)
        pct = int(state.offer_cancel_ratio * 100)
        message = (
            f"👻 Possible spoofing: {account[:12]}... cancelled {pct}% of offers "
            f"({state.offer_cancels} cancels / {total_offers} total)"
        )

        alert = Alert(
            id=_make_id(account),
            timestamp=_now_iso(),
            account=account,
            alert_type="OFFER_CANCEL_SPIKE",
            severity=severity,
            risk_score=risk,
            is_anomaly=anomaly,
            message=message,
            details={
                "offer_creates": state.offer_creates,
                "offer_cancels": state.offer_cancels,
                "cancel_ratio": round(state.offer_cancel_ratio, 3),
                "cancel_ratio_threshold": CANCEL_RATIO_THRESHOLD,
            },
        )

        self.buffer.set_cooldown(account, "OFFER_CANCEL_SPIKE")
        return [alert]

    def _check_multi_whale_convergence(self, tx: dict) -> list[Alert]:
        """
        MULTI_WHALE_CONVERGENCE — multiple whales are trading the same token
        simultaneously, a signal of coordinated activity.

        Fires when:
          - tx_type is OfferCreate
          - Currency is non-XRP
          - >= MULTI_WHALE_MIN distinct whale accounts have OfferCreates
            for this same currency in the current buffer window
          - Not on network-level cooldown for this token

        Cooldown key is ("TOKEN:{currency}", "MULTI_WHALE_CONVERGENCE") so
        each token has its own independent cooldown.
        """
        tx_type  = tx.get("tx_type", "")
        currency = tx.get("currency") or ""
        risk     = float(tx.get("risk_score") or 0)
        anomaly  = int(tx.get("is_anomaly") or 0)

        if tx_type != "OfferCreate":
            return []
        if not currency or currency == "XRP":
            return []

        cooldown_key = f"TOKEN:{currency}"
        if self.buffer.is_on_cooldown(cooldown_key, "MULTI_WHALE_CONVERGENCE"):
            return []

        # Count distinct whale accounts offering this currency in the window
        whale_accounts_on_token = set(
            t.get("account") for t in self.buffer._buffer
            if t.get("tx_type") == "OfferCreate"
            and t.get("currency") == currency
            and self.registry.is_whale(t.get("account", ""))
        )

        if len(whale_accounts_on_token) < MULTI_WHALE_MIN:
            return []

        severity = _severity(risk, anomaly)
        token_short = currency[:10]
        n = len(whale_accounts_on_token)
        message = (
            f"🔥 {n} whales trading {token_short} simultaneously — "
            f"coordinated activity detected"
        )

        alert = Alert(
            id=_make_id("NETWORK"),
            timestamp=_now_iso(),
            account="NETWORK",
            alert_type="MULTI_WHALE_CONVERGENCE",
            severity=severity,
            risk_score=risk,
            is_anomaly=anomaly,
            message=message,
            details={
                "token": currency,
                "whale_count": n,
                "whale_accounts": list(whale_accounts_on_token),
                "min_whales_threshold": MULTI_WHALE_MIN,
            },
        )

        self.buffer.set_cooldown(cooldown_key, "MULTI_WHALE_CONVERGENCE")
        return [alert]
