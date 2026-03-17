"""
Feature engineering for XRPL transaction anomaly detection.

Reads the RAW transaction CSV (ie_dataset/transactions.csv) and computes
12 features per transaction that capture transaction size, memo content,
account activity patterns, and volume dynamics.

Usage:
    python feature_engineering.py
    # -> saves data/featured_transactions.csv
"""

import json
import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from normalize import _parse_amount, _pick_amount, normalize_transaction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FALLBACK_CSV = "/Users/artemabesadze/Desktop/XRPL Commons Work/XRPL_ML_WhaleProject/ie_dataset/transactions.csv"
RAW_CSV = "ie_dataset/transactions.csv" if __import__("os").path.exists("ie_dataset/transactions.csv") else _FALLBACK_CSV
OUTPUT_CSV = "data/featured_transactions.csv"

FEATURE_COLUMNS = [
    "tx_size_percentile",      # percentile rank of amount_xrp among all txs
    "is_large_tx",             # 1 if amount_xrp > 99th percentile
    "wallet_balance_change",   # net XRP change for the account (sent - received)
    "total_volume_5m",         # total XRP volume in a 5-minute rolling window
    "memo_entropy",            # Shannon entropy of the decoded memo text
    "memo_length",             # character length of the decoded memo text
    "duplicate_memo_count",    # how many times the same memo appears in dataset
    "contains_url",            # 1 if memo contains http/https/www
    "rolling_tx_count_5m",     # txs from the same account in a 5-min window
    "tx_per_minute",           # average transaction rate for the account
    "tx_rate_z_score",         # z-score of account's tx rate vs all accounts
    "volume_spike_ratio",      # ratio of current 5m volume to average 5m volume
]


# ---------------------------------------------------------------------------
# Helper: decode hex-encoded memo data
# ---------------------------------------------------------------------------

def _decode_memos(raw_memos) -> str:
    """Parse the memos column and return the decoded memo text.

    The memos column is a JSON string like:
        [{"memoData": "48656C6C6F", "memoType": "..."}]
    where memoData is hex-encoded.  We decode it and concatenate all memo
    texts in the list, separated by spaces.

    Returns an empty string when there is no memo.
    """
    # Handle NaN / None / empty
    if raw_memos is None or (isinstance(raw_memos, float) and math.isnan(raw_memos)):
        return ""
    if isinstance(raw_memos, str):
        raw_memos = raw_memos.strip()
        if not raw_memos:
            return ""

    try:
        memo_list = json.loads(raw_memos) if isinstance(raw_memos, str) else raw_memos
    except (json.JSONDecodeError, TypeError):
        return ""

    if not isinstance(memo_list, list):
        return ""

    parts = []
    for memo in memo_list:
        hex_data = memo.get("memoData", "") if isinstance(memo, dict) else ""
        if hex_data:
            try:
                # Decode hex -> bytes -> UTF-8 text (replace errors gracefully)
                decoded = bytes.fromhex(hex_data).decode("utf-8", errors="replace")
                parts.append(decoded)
            except (ValueError, TypeError):
                parts.append("")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Helper: Shannon entropy of a string
# ---------------------------------------------------------------------------

def _shannon_entropy(text: str) -> float:
    """Compute Shannon entropy (in bits) of a string.

    Higher entropy = more random-looking text; 0 for empty strings.
    """
    if not text:
        return 0.0
    freq = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


# ---------------------------------------------------------------------------
# Main feature-engineering function
# ---------------------------------------------------------------------------

def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Take a raw CSV DataFrame and return it with normalized columns + 12 features.

    Steps:
        1. Normalize every row (same schema as normalize.py)
        2. Decode memos from the raw data
        3. Sort by timestamp
        4. Compute the 12 feature columns
    """

    # ── Step 1: normalise to the 10-column schema ─────────────────────
    print("[1/4] Normalising transactions ...")
    records = [normalize_transaction(row) for row in raw_df.to_dict(orient="records")]
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["tx_type"] = df["tx_type"].astype("category")

    # ── Step 2: decode memos from the RAW data ────────────────────────
    print("[2/4] Decoding memo data ...")
    df["memo_text"] = raw_df["memos"].apply(_decode_memos).values

    # ── Step 3: sort by timestamp ─────────────────────────────────────
    print("[3/4] Sorting by timestamp ...")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Step 4: compute the 12 features ───────────────────────────────
    print("[4/4] Computing features ...")

    # Only use XRP-denominated amounts for numeric features.
    # Token amounts (ETH, USDC, etc.) are on a different scale and would
    # distort volume/size features.
    xrp_amount = df["amount_xrp"].where(df["currency"] == "XRP", 0.0).fillna(0.0).astype(float)

    # Cap XRP amounts at 100,000 XRP. Anything larger is almost certainly
    # a trust line limit or internal ledger operation, not a real transfer.
    # This keeps percentiles and volume features meaningful.
    MAX_XRP = 100_000
    amount = xrp_amount.clip(upper=MAX_XRP)

    # --- tx_size_percentile ---
    # Percentile rank: what fraction of transactions have a smaller amount?
    df["tx_size_percentile"] = amount.rank(pct=True)

    # --- is_large_tx ---
    # Flag transactions whose amount exceeds the 99th percentile
    p99 = amount.quantile(0.99)
    df["is_large_tx"] = (amount > p99).astype(int)

    # --- wallet_balance_change ---
    # Approximate net XRP change per account.
    # Sent amounts are negative, received amounts are positive.
    # Uses XRP-only amounts to avoid mixing currency scales.
    df["_xrp_amount"] = amount
    account_sent = df.groupby("account")["_xrp_amount"].transform("sum")
    recv_lookup = df.groupby("destination")["_xrp_amount"].sum()
    df["wallet_balance_change"] = df["account"].map(recv_lookup).fillna(0) - account_sent
    df.drop(columns=["_xrp_amount"], inplace=True)

    # --- total_volume_5m (rolling 5-min total XRP volume) ---
    # We use a time-based rolling window on the sorted data.
    # Uses XRP-only amounts (stored in `amount` Series).
    df_indexed = df.set_index("timestamp")
    rolling_vol = amount.set_axis(df_indexed.index).rolling("5min").sum()
    df["total_volume_5m"] = rolling_vol.values

    # --- memo features ---
    df["memo_entropy"] = df["memo_text"].apply(_shannon_entropy)
    df["memo_length"] = df["memo_text"].apply(len)

    # duplicate_memo_count: how many times the exact memo appears in the dataset
    # (count only non-empty memos; empty memos get 0)
    memo_counts = df.loc[df["memo_text"] != "", "memo_text"].value_counts()
    df["duplicate_memo_count"] = df["memo_text"].map(memo_counts).fillna(0).astype(int)

    # contains_url: does the memo text contain a URL pattern?
    url_pattern = re.compile(r"https?://|www\.", re.IGNORECASE)
    df["contains_url"] = df["memo_text"].apply(
        lambda t: 1 if url_pattern.search(t) else 0
    )

    # --- rolling_tx_count_5m (per-account rolling 5-min tx count) ---
    # For each account, count how many transactions occurred in the past 5 min.
    # We do this by iterating over groups (efficient enough for ~272K rows).
    df["_ones"] = 1
    df_indexed2 = df.set_index("timestamp")
    rolling_counts = (
        df_indexed2.groupby("account")["_ones"]
        .rolling("5min")
        .sum()
    )
    # rolling_counts has a multi-index (account, timestamp); align back
    rolling_counts = rolling_counts.droplevel(0).sort_index()
    # Because there can be duplicate timestamps per account, use positional align
    df["rolling_tx_count_5m"] = rolling_counts.values
    df.drop(columns=["_ones"], inplace=True)

    # --- tx_per_minute (average tx rate per account) ---
    # = total txs from this account / time span in minutes for the account
    acct_counts = df["account"].value_counts()
    acct_time_span = df.groupby("account")["timestamp"].agg(lambda s: (s.max() - s.min()).total_seconds() / 60.0)
    # Avoid division by zero: if an account has only 1 tx, time span is 0 -> rate = 0
    acct_rate = (acct_counts / acct_time_span.clip(lower=1.0))
    df["tx_per_minute"] = df["account"].map(acct_rate).fillna(0)

    # --- tx_rate_z_score ---
    # z-score of the account's tx_per_minute relative to all accounts
    mean_rate = acct_rate.mean()
    std_rate = acct_rate.std()
    if std_rate > 0:
        acct_z = (acct_rate - mean_rate) / std_rate
    else:
        acct_z = acct_rate * 0  # all zeros if no variance
    df["tx_rate_z_score"] = df["account"].map(acct_z).fillna(0)

    # --- volume_spike_ratio ---
    # Ratio of the current 5-min volume to the overall average 5-min volume.
    avg_5m_volume = df["total_volume_5m"].mean()
    if avg_5m_volume > 0:
        df["volume_spike_ratio"] = df["total_volume_5m"] / avg_5m_volume
    else:
        df["volume_spike_ratio"] = 0.0

    # Drop the temporary memo_text helper column
    df.drop(columns=["memo_text"], inplace=True)

    return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    print(f"Loading raw CSV from {RAW_CSV} ...")
    raw = pd.read_csv(RAW_CSV)
    print(f"  -> {len(raw):,} rows loaded.\n")

    featured = engineer_features(raw)

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    featured.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved featured transactions to {OUTPUT_CSV}")
    print(f"  -> {len(featured):,} rows, {len(featured.columns)} columns\n")

    # ── Summary stats ─────────────────────────────────────────────────
    print("=" * 60)
    print("Feature summary statistics")
    print("=" * 60)
    print(featured[FEATURE_COLUMNS].describe().round(4).to_string())
    print()

    # Show a few rows that have non-zero memo features
    memo_rows = featured[featured["memo_length"] > 0].head(5)
    if not memo_rows.empty:
        print("Sample rows with memos:")
        print(memo_rows[["account", "tx_type", "amount_xrp", "memo_entropy",
                          "memo_length", "duplicate_memo_count", "contains_url"]].to_string())
