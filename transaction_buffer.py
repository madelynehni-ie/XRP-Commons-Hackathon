"""
Transaction Buffer — rolling in-memory state for the alert engine.

Maintains a 10-minute sliding window of scored transactions and exposes
per-account and network-wide state snapshots that alert detectors read.
Also manages per-alert cooldowns so the same alert doesn't fire repeatedly.

Usage:
    from transaction_buffer import TransactionBuffer

    buffer = TransactionBuffer()
    buffer.add(scored_tx)                              # called per transaction

    state = buffer.get_account_state("rXXX...")        # AccountState
    net   = buffer.get_network_state()                 # NetworkState

    if not buffer.is_on_cooldown("rXXX...", "TRANSACTION_BURST"):
        buffer.set_cooldown("rXXX...", "TRANSACTION_BURST")
        # fire the alert
"""

from __future__ import annotations

from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Config  (will be made user-configurable later)
# ---------------------------------------------------------------------------

WINDOW_SECONDS  = 600   # 10-minute rolling window
COOLDOWN_SECONDS = 120  # 2-minute cooldown per alert type per account


# ---------------------------------------------------------------------------
# State snapshots (read-only views returned to alert detectors)
# ---------------------------------------------------------------------------

@dataclass
class AccountState:
    """Rolling stats for a single account within the current window."""
    account: str
    tx_count: int               # total txs in window
    xrp_out: float              # XRP sent in window
    xrp_in: float               # XRP received (as destination) in window
    offer_creates: int          # OfferCreate count in window
    offer_cancels: int          # OfferCancel count in window
    tx_types: dict              # {"Payment": 3, "OfferCreate": 12, ...}
    memo_texts: list            # decoded memo strings seen in window
    risk_scores: list           # ML risk_score per tx in window
    tokens: dict                # {currency: {"buys": N, "sells": N}}
    last_seen: Optional[datetime]  # timestamp of most recent tx

    @property
    def avg_risk_score(self) -> float:
        return sum(self.risk_scores) / len(self.risk_scores) if self.risk_scores else 0.0

    @property
    def max_risk_score(self) -> float:
        return max(self.risk_scores) if self.risk_scores else 0.0

    @property
    def offer_cancel_ratio(self) -> float:
        total = self.offer_creates + self.offer_cancels
        return self.offer_cancels / total if total > 0 else 0.0

    @property
    def net_xrp(self) -> float:
        """Positive = net receiver, negative = net sender."""
        return self.xrp_in - self.xrp_out


@dataclass
class NetworkState:
    """Rolling stats across the entire network within the current window."""
    tx_count: int
    active_accounts: int
    total_xrp_volume: float
    avg_risk_score: float
    anomaly_count: int          # transactions flagged is_anomaly=1


# ---------------------------------------------------------------------------
# TransactionBuffer
# ---------------------------------------------------------------------------

class TransactionBuffer:
    """Sliding-window buffer of scored transactions with cooldown tracking."""

    def __init__(
        self,
        window_seconds: int = WINDOW_SECONDS,
        cooldown_seconds: int = COOLDOWN_SECONDS,
    ):
        self.window_seconds  = window_seconds
        self.cooldown_seconds = cooldown_seconds

        # Chronological deque of scored transaction dicts
        self._buffer: deque[dict] = deque()

        # Cooldown tracker: (account, alert_type) → UTC datetime last fired
        self._cooldowns: dict[tuple[str, str], datetime] = {}

    # ── ingestion ─────────────────────────────────────────────────────────

    def add(self, scored_tx: dict) -> None:
        """Add a scored transaction to the buffer and prune expired entries."""
        self._buffer.append(scored_tx)
        self._prune()

    def _prune(self) -> None:
        """Remove transactions older than window_seconds from the front."""
        if not self._buffer:
            return
        # Use the most recent tx's timestamp as "now"
        now = self._parse_ts(self._buffer[-1].get("timestamp"))
        if now is None:
            return
        cutoff = now - pd.Timedelta(seconds=self.window_seconds)
        while self._buffer:
            ts = self._parse_ts(self._buffer[0].get("timestamp"))
            if ts is not None and ts < cutoff:
                self._buffer.popleft()
            else:
                break

    @staticmethod
    def _parse_ts(raw) -> Optional[pd.Timestamp]:
        if raw is None:
            return None
        if isinstance(raw, pd.Timestamp):
            return raw if raw.tzinfo is not None else raw.tz_localize("UTC")
        if isinstance(raw, datetime):
            return pd.Timestamp(raw)   # preserves existing tzinfo
        try:
            return pd.Timestamp(raw, tz="UTC")
        except Exception:
            return None

    # ── account state ─────────────────────────────────────────────────────

    def get_account_state(self, account: str) -> AccountState:
        """Compute rolling stats for one account from the current buffer."""
        tx_count      = 0
        xrp_out       = 0.0
        xrp_in        = 0.0
        offer_creates = 0
        offer_cancels = 0
        tx_types: dict[str, int]         = defaultdict(int)
        memo_texts: list[str]            = []
        risk_scores: list[float]         = []
        tokens: dict[str, dict]          = defaultdict(lambda: {"buys": 0, "sells": 0})
        last_seen: Optional[pd.Timestamp] = None

        for tx in self._buffer:
            if tx.get("account") != account:
                # Still check if this account was a destination (XRP received)
                if tx.get("destination") == account:
                    currency = tx.get("currency", "XRP")
                    if currency == "XRP":
                        xrp_in += float(tx.get("amount_xrp") or 0)
                continue

            tx_count += 1
            tx_type   = tx.get("tx_type") or ""
            currency  = tx.get("currency") or "XRP"
            amount    = float(tx.get("amount_xrp") or 0)

            tx_types[tx_type] += 1

            if currency == "XRP":
                xrp_out += amount

            if tx_type == "OfferCreate":
                offer_creates += 1
                if currency != "XRP":
                    tokens[currency]["buys"] += 1
            elif tx_type == "OfferCancel":
                offer_cancels += 1

            memo = tx.get("memo_text") or ""
            if memo:
                memo_texts.append(memo)

            rs = tx.get("risk_score")
            if rs is not None:
                risk_scores.append(float(rs))

            ts = self._parse_ts(tx.get("timestamp"))
            if ts is not None:
                if last_seen is None or ts > last_seen:
                    last_seen = ts

        return AccountState(
            account=account,
            tx_count=tx_count,
            xrp_out=xrp_out,
            xrp_in=xrp_in,
            offer_creates=offer_creates,
            offer_cancels=offer_cancels,
            tx_types=dict(tx_types),
            memo_texts=memo_texts,
            risk_scores=risk_scores,
            tokens=dict(tokens),
            last_seen=last_seen,
        )

    # ── network state ─────────────────────────────────────────────────────

    def get_network_state(self) -> NetworkState:
        """Compute network-wide rolling stats from the current buffer."""
        if not self._buffer:
            return NetworkState(0, 0, 0.0, 0.0, 0)

        accounts      = set()
        total_xrp     = 0.0
        risk_sum      = 0.0
        risk_count    = 0
        anomaly_count = 0

        for tx in self._buffer:
            acc = tx.get("account")
            if acc:
                accounts.add(acc)

            currency = tx.get("currency") or "XRP"
            if currency == "XRP":
                total_xrp += float(tx.get("amount_xrp") or 0)

            rs = tx.get("risk_score")
            if rs is not None:
                risk_sum   += float(rs)
                risk_count += 1

            if tx.get("is_anomaly") == 1:
                anomaly_count += 1

        return NetworkState(
            tx_count=len(self._buffer),
            active_accounts=len(accounts),
            total_xrp_volume=total_xrp,
            avg_risk_score=risk_sum / risk_count if risk_count > 0 else 0.0,
            anomaly_count=anomaly_count,
        )

    # ── cooldown management ───────────────────────────────────────────────

    def is_on_cooldown(self, account: str, alert_type: str) -> bool:
        """Return True if this alert type fired recently for this account."""
        key = (account, alert_type)
        last_fired = self._cooldowns.get(key)
        if last_fired is None:
            return False
        elapsed = (datetime.now(timezone.utc) - last_fired).total_seconds()
        return elapsed < self.cooldown_seconds

    def set_cooldown(self, account: str, alert_type: str) -> None:
        """Record that this alert just fired for this account."""
        self._cooldowns[(account, alert_type)] = datetime.now(timezone.utc)

    def clear_cooldown(self, account: str, alert_type: str) -> None:
        """Manually clear a cooldown (useful for testing)."""
        self._cooldowns.pop((account, alert_type), None)

    # ── introspection ─────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def active_accounts(self) -> set[str]:
        return {tx["account"] for tx in self._buffer if tx.get("account")}
