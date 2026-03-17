"""
Whale Registry — identifies whale accounts from historical transaction data.

A whale is defined as any account in the top 5% by total transaction count.
XRP volume is also tracked, but since most transactions in the dataset are
token-based (not native XRP), transaction count is the primary signal.

Usage:
    from whale_registry import WhaleRegistry
    registry = WhaleRegistry.build()
    registry.is_whale("rXXX...")          # True / False
    registry.get_stats("rXXX...")         # dict of per-account stats
    registry.percentile("rXXX...")        # 0.0 – 1.0
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths — prefer featured CSV (faster, already normalised); fall back to raw
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent
_FEATURED_CSV = _REPO_ROOT / "data" / "featured_transactions.csv"
_RAW_CSV = _REPO_ROOT / "ie_dataset" / "transactions.csv"
_RAW_CSV_FALLBACK = Path(
    "/Users/artemabesadze/Desktop/XRPL Commons Work/"
    "XRPL_ML_WhaleProject/ie_dataset/transactions.csv"
)

# Top-N% by transaction count defines a whale
WHALE_PERCENTILE = 0.95


# ---------------------------------------------------------------------------
# Per-account stats
# ---------------------------------------------------------------------------

@dataclass
class AccountStats:
    account: str
    tx_count: int = 0
    xrp_sent: float = 0.0
    xrp_received: float = 0.0
    unique_destinations: int = 0
    tx_types: dict = field(default_factory=dict)   # {"Payment": 3, ...}
    tokens_traded: set = field(default_factory=set) # {"SOLO", "RLUSD", ...}
    percentile: float = 0.0   # 0–1 within the full account population
    is_whale: bool = False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class WhaleRegistry:
    """Pre-computed whale index built from historical transaction data."""

    def __init__(
        self,
        stats: dict[str, AccountStats],
        whale_tx_threshold: int,
        total_accounts: int,
    ):
        self._stats = stats
        self.whale_tx_threshold = whale_tx_threshold
        self.total_accounts = total_accounts
        self._whale_set: set[str] = {
            acc for acc, s in stats.items() if s.is_whale
        }

    # ── public API ────────────────────────────────────────────────────────

    def is_whale(self, account: str) -> bool:
        """Return True if this account is in the top 5% by tx count."""
        return account in self._whale_set

    def get_stats(self, account: str) -> Optional[AccountStats]:
        """Return AccountStats for this account, or None if unknown."""
        return self._stats.get(account)

    def percentile(self, account: str) -> float:
        """Return the account's percentile rank (0–1). 0 if unknown."""
        s = self._stats.get(account)
        return s.percentile if s else 0.0

    @property
    def whale_accounts(self) -> set[str]:
        return self._whale_set

    @property
    def whale_count(self) -> int:
        return len(self._whale_set)

    def summary(self) -> dict:
        return {
            "total_accounts": self.total_accounts,
            "whale_accounts": self.whale_count,
            "whale_pct": round(self.whale_count / max(self.total_accounts, 1) * 100, 2),
            "whale_tx_threshold": self.whale_tx_threshold,
        }

    # ── factory ───────────────────────────────────────────────────────────

    @classmethod
    def build(cls, verbose: bool = True) -> "WhaleRegistry":
        """Load transaction data and compute the whale registry."""

        df = _load_data(verbose)
        return _compute_registry(df, verbose)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data(verbose: bool) -> pd.DataFrame:
    """Load the minimal columns needed from whichever CSV exists."""

    COLS_FEATURED = ["account", "destination", "tx_type", "amount_xrp", "currency"]
    COLS_RAW = ["account", "destination", "transaction_type", "amount", "fee"]

    # 1. Try featured CSV first (already normalised)
    if _FEATURED_CSV.exists():
        if verbose:
            print(f"[WhaleRegistry] Loading featured CSV: {_FEATURED_CSV}")
        df = pd.read_csv(_FEATURED_CSV, usecols=COLS_FEATURED, low_memory=False)
        df = df.rename(columns={"tx_type": "transaction_type"})
        return df

    # 2. Try raw CSV relative to repo
    raw_path = _RAW_CSV if _RAW_CSV.exists() else _RAW_CSV_FALLBACK
    if not raw_path.exists():
        raise FileNotFoundError(
            f"No transaction data found. Checked:\n"
            f"  {_FEATURED_CSV}\n  {_RAW_CSV}\n  {_RAW_CSV_FALLBACK}"
        )

    if verbose:
        print(f"[WhaleRegistry] Loading raw CSV: {raw_path}")

    df = pd.read_csv(raw_path, usecols=COLS_RAW, low_memory=False)

    # Parse amount field to get XRP values and currency
    if verbose:
        print("[WhaleRegistry] Parsing amount field ...")
    parsed = df["amount"].apply(_parse_amount_raw)
    df["amount_xrp"] = parsed.apply(lambda x: x[0])
    df["currency"]   = parsed.apply(lambda x: x[1])
    df = df.rename(columns={"transaction_type": "transaction_type"})

    return df


def _parse_amount_raw(val) -> tuple[float, str]:
    """Parse raw amount field → (xrp_value, currency)."""
    if pd.isna(val):
        return 0.0, "XRP"
    try:
        # Plain drops → native XRP
        return float(val) / 1_000_000, "XRP"
    except (ValueError, TypeError):
        pass
    try:
        d = ast.literal_eval(str(val))
        if isinstance(d, dict):
            currency = d.get("currency", "XRP")
            value = float(d.get("value", 0))
            # Drop amounts are treated as XRP only for native XRP
            if currency == "XRP":
                return value / 1_000_000, "XRP"
            return value, currency
    except Exception:
        pass
    return 0.0, "XRP"


# ---------------------------------------------------------------------------
# Registry computation
# ---------------------------------------------------------------------------

def _compute_registry(df: pd.DataFrame, verbose: bool) -> WhaleRegistry:
    """Compute per-account stats and identify whales."""

    if verbose:
        print(f"[WhaleRegistry] Computing stats for {len(df):,} transactions ...")

    # --- Transaction count per account ---
    tx_counts = df.groupby("account").size().rename("tx_count")

    # --- XRP sent per account (only native XRP rows) ---
    xrp_mask = df["currency"] == "XRP"
    xrp_sent = (
        df[xrp_mask]
        .groupby("account")["amount_xrp"]
        .sum()
        .rename("xrp_sent")
    )

    # --- XRP received per account (as destination) ---
    xrp_received = (
        df[xrp_mask & df["destination"].notna()]
        .groupby("destination")["amount_xrp"]
        .sum()
        .rename("xrp_received")
    )

    # --- Unique destinations per account ---
    unique_dests = (
        df[df["destination"].notna()]
        .groupby("account")["destination"]
        .nunique()
        .rename("unique_destinations")
    )

    # --- Transaction type breakdown per account ---
    tx_type_col = "tx_type" if "tx_type" in df.columns else "transaction_type"
    tx_type_counts = (
        df.groupby(["account", tx_type_col])
        .size()
        .unstack(fill_value=0)
    )

    # --- Tokens traded (non-XRP currencies) ---
    token_df = df[df["currency"] != "XRP"][["account", "currency"]]
    tokens_by_account = token_df.groupby("account")["currency"].apply(set)

    # --- Percentile rank by tx count ---
    sorted_counts = tx_counts.sort_values()
    percentile_rank = pd.Series(
        np.searchsorted(sorted_counts.values, tx_counts.values, side="left") / len(tx_counts),
        index=tx_counts.index,
    )

    # --- Whale threshold ---
    threshold = int(tx_counts.quantile(WHALE_PERCENTILE))

    # --- Build AccountStats objects ---
    all_accounts = tx_counts.index.tolist()
    stats: dict[str, AccountStats] = {}

    for acc in all_accounts:
        count = int(tx_counts.get(acc, 0))
        pct = float(percentile_rank.get(acc, 0.0))

        # tx type breakdown as plain dict
        if acc in tx_type_counts.index:
            tx_types = {
                col: int(tx_type_counts.loc[acc, col])
                for col in tx_type_counts.columns
                if tx_type_counts.loc[acc, col] > 0
            }
        else:
            tx_types = {}

        stats[acc] = AccountStats(
            account=acc,
            tx_count=count,
            xrp_sent=float(xrp_sent.get(acc, 0.0)),
            xrp_received=float(xrp_received.get(acc, 0.0)),
            unique_destinations=int(unique_dests.get(acc, 0)),
            tx_types=tx_types,
            tokens_traded=tokens_by_account.get(acc, set()),
            percentile=pct,
            is_whale=(count >= threshold),
        )

    if verbose:
        n_whales = sum(1 for s in stats.values() if s.is_whale)
        print(
            f"[WhaleRegistry] Done. "
            f"{len(stats):,} accounts | "
            f"{n_whales} whales (top 5%, threshold: {threshold} txs)"
        )

    return WhaleRegistry(stats, threshold, len(stats))


# ---------------------------------------------------------------------------
# CLI — quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    registry = WhaleRegistry.build(verbose=True)
    print()
    print("Summary:", registry.summary())
    print()
    print("Top 10 whales by tx count:")
    top = sorted(
        [s for s in registry._stats.values() if s.is_whale],
        key=lambda s: s.tx_count,
        reverse=True,
    )[:10]
    for s in top:
        print(
            f"  {s.account}  txs={s.tx_count:>6,}  "
            f"pct={s.percentile:.3f}  "
            f"xrp_sent={s.xrp_sent:>12,.2f}  "
            f"tokens={len(s.tokens_traded)}"
        )
