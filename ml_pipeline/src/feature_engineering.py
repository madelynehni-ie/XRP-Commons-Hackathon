"""Feature engineering for XRPL risk / fraud detection."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from price_service import get_xrp_price_usd

FEATURE_COLUMNS = [
    "tx_size_percentile",
    "is_large_tx",
    "wallet_balance_change",
    "total_volume_5m",
    "memo_entropy",
    "memo_length",
    "duplicate_memo_count",
    "contains_url",
    "rolling_tx_count_5m",
    "tx_per_minute",
    "tx_rate_z_score",
    "volume_spike_ratio",
]

URL_REGEX = re.compile(r"(https?://|www\.)", re.IGNORECASE)
WHALE_PERCENTILE_THRESHOLD = 0.95


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    defaults = {
        "timestamp": pd.NaT,
        "ledger_index": np.nan,
        "tx_hash": "",
        "tx_type": "",
        "account": "",
        "destination": "",
        "amount_xrp": 0.0,
        "currency": "",
        "issuer": "",
    }

    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["ledger_index"] = pd.to_numeric(out["ledger_index"], errors="coerce")
    out["amount_xrp"] = pd.to_numeric(out["amount_xrp"], errors="coerce").fillna(0.0)
    out["tx_type"] = out["tx_type"].fillna("").astype(str)
    out["account"] = out["account"].fillna("").astype(str)
    out["destination"] = out["destination"].fillna("").astype(str)
    out["tx_hash"] = out["tx_hash"].fillna("").astype(str)
    out["currency"] = out["currency"].fillna("").astype(str)
    out["issuer"] = out["issuer"].fillna("").astype(str)

    return out


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


def contains_url(text: str) -> int:
    if not text:
        return 0
    return 1 if URL_REGEX.search(text) else 0


def safe_div(a: float, b: float) -> float:
    if b == 0 or pd.isna(b):
        return 0.0
    return float(a) / float(b)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_columns(df)

    # Sort in a stable order for rolling/account-based features
    df = df.sort_values(
        ["account", "timestamp", "ledger_index"],
        na_position="last"
    ).reset_index(drop=True)

    # USD conversion for XRP only
    xrp_price = get_xrp_price_usd()
    print(f"Using XRP price: ${xrp_price}")

    df["amount_usd"] = df["amount_xrp"] * xrp_price
    df.loc[df["currency"] != "XRP", "amount_usd"] = 0.0

    # 1. tx_size_percentile
    # Token-aware percentile using currency+issuer grouping, ranked on USD-equivalent.
    # XRP forms its own group; non-XRP currently rank at 0 if amount_usd is 0.
    df["tx_size_percentile"] = (
        df.groupby(["currency", "issuer"])["amount_usd"]
        .rank(pct=True)
        .fillna(0.0)
    )

    # 2. is_large_tx
    df["is_large_tx"] = (
        df["tx_size_percentile"] >= WHALE_PERCENTILE_THRESHOLD
    ).astype(int)

    # 3. wallet_balance_change
    # Not available from current normalized schema yet
    df["wallet_balance_change"] = 0.0

    # 4. total_volume_5m
    # Use USD-equivalent for volume so large XRP transfers are comparable.
    df["total_volume_5m"] = (
        df.groupby("account")
        .rolling("5min", on="timestamp")["amount_usd"]
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    # 5. memo_entropy
    # Not available from current normalized schema yet
    df["memo_entropy"] = 0.0

    # 6. memo_length
    df["memo_length"] = 0.0

    # 7. duplicate_memo_count
    df["duplicate_memo_count"] = 0.0

    # 8. contains_url
    df["contains_url"] = 0

    # 9. rolling_tx_count_5m
    df["rolling_tx_count_5m"] = (
        df.groupby("account")
        .rolling("5min", on="timestamp")["tx_hash"]
        .count()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    # 10. tx_per_minute
    df["tx_per_minute"] = df["rolling_tx_count_5m"] / 5.0

    # 11. tx_rate_z_score
    rolling_mean = (
        df.groupby("account")["tx_per_minute"]
        .transform(lambda s: s.rolling(window=20, min_periods=3).mean())
        .fillna(0.0)
    )
    rolling_std = (
        df.groupby("account")["tx_per_minute"]
        .transform(lambda s: s.rolling(window=20, min_periods=3).std())
        .fillna(0.0)
    )

    df["tx_rate_z_score"] = (
        (df["tx_per_minute"] - rolling_mean) / rolling_std.replace(0, np.nan)
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # 12. volume_spike_ratio
    volume_baseline = (
        df.groupby("account")["total_volume_5m"]
        .transform(lambda s: s.rolling(window=20, min_periods=3).mean())
        .fillna(0.0)
    )

    df["volume_spike_ratio"] = [
        safe_div(curr, base)
        for curr, base in zip(df["total_volume_5m"], volume_baseline)
    ]

    keep_cols = [
        "timestamp",
        "ledger_index",
        "tx_hash",
        "tx_type",
        "account",
        "destination",
        "currency",
        "issuer",
        "amount_xrp",
        "amount_usd",
    ] + FEATURE_COLUMNS

    out = df[keep_cols].copy()
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return out


if __name__ == "__main__":
    from ml_data import load_all_transactions

    raw = load_all_transactions()
    feats = build_features(raw)
    print(feats.head(20).to_string())
    print(f"\nFeature rows: {len(feats)}")