import numpy as np
import pandas as pd

from src.utils import contains_url, safe_div, shannon_entropy


FEATURE_COLUMNS = [
    # Whale / volume
    "tx_size_percentile_local",
    "is_large_tx",
    "total_volume_5m",
    "volume_spike_ratio",
    "fee_rate",
    # Burst / activity
    "rolling_tx_count_5m",
    "tx_per_minute",
    "tx_rate_z_score",
    "tx_rate_mad_z",
    # Memo spam
    "memo_entropy",
    "memo_length",
    "duplicate_memo_count",
    "contains_url",
    # Destination behaviour
    "dest_concentration",
    "unique_dest_ratio",
    # Transaction type mix
    "offer_cancel_ratio",
    "failed_tx_ratio",
    # Dormancy
    "dormancy_gap",
]


def build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from a normalised transactions DataFrame.

    Accepts both normalize.py schema (timestamp, tx_hash, tx_type, amount_xrp)
    and queries.py schema (close_time, hash, amount_value, balance_xrp, success).
    """
    df = transactions.copy()

    # ── Unify column names ────────────────────────────────────────────────────
    if "timestamp" in df.columns and "close_time" not in df.columns:
        df["close_time"] = df["timestamp"]
    if "tx_hash" in df.columns and "hash" not in df.columns:
        df["hash"] = df["tx_hash"]
    if "amount_xrp" in df.columns and "amount_value" not in df.columns:
        df["amount_value"] = df["amount_xrp"]
    if "tx_type" in df.columns and "transaction_type" not in df.columns:
        df["transaction_type"] = df["tx_type"]
    if "success" not in df.columns:
        df["success"] = df["result"].eq("tesSUCCESS") if "result" in df.columns else True
    if "memo_text" not in df.columns:
        df["memo_text"] = ""

    # ── Type coercion ─────────────────────────────────────────────────────────
    df["close_time"]       = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
    df["destination"]      = df["destination"].fillna("").astype(str)
    df["memo_text"]        = df["memo_text"].fillna("").astype(str)
    df["amount_value"]     = pd.to_numeric(df["amount_value"], errors="coerce").fillna(0.0)
    df["success"]          = df["success"].fillna(True).astype(bool)
    df["transaction_type"] = df.get("transaction_type", pd.Series("", index=df.index)).fillna("").astype(str)
    df["fee"]              = pd.to_numeric(df.get("fee", pd.Series(0, index=df.index)), errors="coerce").fillna(0.0)

    df = df.sort_values(["account", "ledger_index", "close_time"]).reset_index(drop=True)

    # ── WHALE / VOLUME ────────────────────────────────────────────────────────

    # Per-account percentile — no leakage across accounts
    df["tx_size_percentile_local"] = (
        df.groupby("account")["amount_value"].rank(pct=True).fillna(0.0)
    )
    df["is_large_tx"] = (df["tx_size_percentile_local"] >= 0.95).astype(int)

    df["total_volume_5m"] = (
        df.groupby("account")
        .rolling("5min", on="close_time")["amount_value"]
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    volume_baseline = (
        df.groupby("account")["total_volume_5m"]
        .transform(lambda s: s.rolling(20, min_periods=3).mean())
        .fillna(0.0)
    )
    df["volume_spike_ratio"] = [
        safe_div(c, b) for c, b in zip(df["total_volume_5m"], volume_baseline)
    ]

    # Fee relative to amount — high = suspicious micro-tx with outsized fee
    df["fee_rate"] = df.apply(
        lambda r: safe_div(r["fee"], r["amount_value"] * 1e6) if r["amount_value"] > 0 else 0.0,
        axis=1,
    )

    # ── BURST / ACTIVITY ──────────────────────────────────────────────────────

    df["rolling_tx_count_5m"] = (
        df.groupby("account")
        .rolling("5min", on="close_time")["hash"]
        .count()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )
    df["tx_per_minute"] = df["rolling_tx_count_5m"] / 5.0

    rolling_mean = (
        df.groupby("account")["tx_per_minute"]
        .transform(lambda s: s.rolling(20, min_periods=3).mean())
        .fillna(0.0)
    )
    rolling_std = (
        df.groupby("account")["tx_per_minute"]
        .transform(lambda s: s.rolling(20, min_periods=3).std())
        .fillna(1.0)
    )
    df["tx_rate_z_score"] = (
        (df["tx_per_minute"] - rolling_mean) / rolling_std.replace(0, np.nan)
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # MAD z-score — robust to single spikes polluting the baseline
    def mad_zscore_transform(s: pd.Series, window: int = 20) -> pd.Series:
        med  = s.rolling(window, min_periods=3).median()
        mad  = s.rolling(window, min_periods=3).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True
        ).replace(0, np.nan)
        return ((s - med) / (1.4826 * mad)).fillna(0.0).clip(-10, 10)

    df["tx_rate_mad_z"] = (
        df.groupby("account")["tx_per_minute"]
        .transform(mad_zscore_transform)
        .fillna(0.0)
    )

    # ── MEMO SPAM ─────────────────────────────────────────────────────────────

    df["memo_entropy"]       = df["memo_text"].apply(shannon_entropy)
    df["memo_length"]        = df["memo_text"].str.len().fillna(0).astype(float)
    df["contains_url"]       = df["memo_text"].apply(contains_url).astype(int)
    df["duplicate_memo_count"] = (
        df.groupby(["account", "memo_text"]).cumcount().astype(float)
    )

    # ── DESTINATION BEHAVIOUR ─────────────────────────────────────────────────

    # Encode destination as int so pandas rolling works on it
    df["_dest_int"] = pd.Categorical(df["destination"]).codes.astype(float)

    rolling_unique_dests = (
        df.groupby("account")
        .rolling("5min", on="close_time")["_dest_int"]
        .apply(lambda x: float(len(set(x))), raw=True)
        .reset_index(level=0, drop=True)
        .fillna(1.0)
    )

    # dest_concentration: 1 / unique_dests — high means always same dest
    df["dest_concentration"] = (1.0 / rolling_unique_dests.clip(lower=1)).fillna(1.0)

    # unique_dest_ratio: unique dests / tx count — low means fan-out
    df["unique_dest_ratio"] = [
        safe_div(u, t)
        for u, t in zip(rolling_unique_dests, df["rolling_tx_count_5m"].clip(lower=1))
    ]

    df.drop(columns=["_dest_int"], inplace=True)

    # ── TRANSACTION TYPE MIX ─────────────────────────────────────────────────

    df["_is_cancel"] = (df["transaction_type"] == "OfferCancel").astype(float)
    df["_is_offer"]  = df["transaction_type"].isin(["OfferCreate", "OfferCancel"]).astype(float)

    rolling_cancels = (
        df.groupby("account")
        .rolling("5min", on="close_time")["_is_cancel"]
        .sum().reset_index(level=0, drop=True).fillna(0.0)
    )
    rolling_offers = (
        df.groupby("account")
        .rolling("5min", on="close_time")["_is_offer"]
        .sum().reset_index(level=0, drop=True).fillna(0.0)
    )
    df["offer_cancel_ratio"] = [
        safe_div(c, o) for c, o in zip(rolling_cancels, rolling_offers.clip(lower=1))
    ]

    df["_failed"] = (~df["success"]).astype(float)
    rolling_failed = (
        df.groupby("account")
        .rolling("5min", on="close_time")["_failed"]
        .sum().reset_index(level=0, drop=True).fillna(0.0)
    )
    df["failed_tx_ratio"] = [
        safe_div(f, t) for f, t in zip(rolling_failed, df["rolling_tx_count_5m"].clip(lower=1))
    ]

    df.drop(columns=["_is_cancel", "_is_offer", "_failed"], inplace=True)

    # ── DORMANCY ──────────────────────────────────────────────────────────────

    df["dormancy_gap"] = (
        df.groupby("account")["ledger_index"]
        .diff()
        .fillna(0.0)
        .clip(upper=100_000)
    )

    # ── OUTPUT ────────────────────────────────────────────────────────────────

    out_cols = ["account", "ledger_index", "close_time"] + FEATURE_COLUMNS
    if "label" in df.columns:
        out_cols.append("label")

    features = df[out_cols].copy()
    features = features.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return features
