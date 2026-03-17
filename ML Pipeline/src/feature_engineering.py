import numpy as np
import pandas as pd

from src.utils import contains_url, safe_div, shannon_entropy


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


def build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    df = transactions.copy()

    df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
    df = df.sort_values(["account", "ledger_index", "close_time"]).reset_index(drop=True)

    df["destination"] = df["destination"].fillna("")
    df["memo_text"] = df["memo_text"].fillna("")
    df["amount_value"] = pd.to_numeric(df["amount_value"], errors="coerce").fillna(0.0)
    df["balance_xrp"] = pd.to_numeric(df["balance_xrp"], errors="coerce").fillna(0.0)
    df["success"] = df["success"].fillna(False).astype(bool)

    # ---------- Whale / large-volume features ----------

    # Percentile rank of transaction size globally
    df["tx_size_percentile"] = df["amount_value"].rank(pct=True).fillna(0.0)

    # Large transaction flag
    df["is_large_tx"] = (df["tx_size_percentile"] >= 0.95).astype(int)

    # Change in wallet balance over time per account
    df["wallet_balance_change"] = (
        df.groupby("account")["balance_xrp"].diff().fillna(0.0)
    )

    # 5-minute rolling transaction volume per account
    df["total_volume_5m"] = (
        df.groupby("account")
        .rolling("5min", on="close_time")["amount_value"]
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    # ---------- Memo spam features ----------

    df["memo_entropy"] = df["memo_text"].apply(shannon_entropy)
    df["memo_length"] = df["memo_text"].str.len().fillna(0).astype(float)
    df["contains_url"] = df["memo_text"].apply(contains_url).astype(int)

    # Count how many times the same memo appears in recent account history
    df["duplicate_memo_count"] = (
        df.groupby(["account", "memo_text"]).cumcount()
    ).astype(float)

    # ---------- Burst / activity features ----------

    # 5-minute rolling tx count per account
    df["rolling_tx_count_5m"] = (
        df.groupby("account")
        .rolling("5min", on="close_time")["hash"]
        .count()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    # Transactions per minute over rolling 5m window
    df["tx_per_minute"] = df["rolling_tx_count_5m"] / 5.0

    # Rolling mean/std for tx_per_minute per account
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

    # Volume spike ratio: current 5m volume vs recent rolling baseline
    volume_baseline = (
        df.groupby("account")["total_volume_5m"]
        .transform(lambda s: s.rolling(window=20, min_periods=3).mean())
        .fillna(0.0)
    )

    df["volume_spike_ratio"] = [
        safe_div(curr, base) for curr, base in zip(df["total_volume_5m"], volume_baseline)
    ]

    # ---------- Final output ----------

    out_cols = ["account", "ledger_index"] + FEATURE_COLUMNS

    if "label" in df.columns:
        out_cols.append("label")

    features = df[out_cols].copy()
    features = features.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return features