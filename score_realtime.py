"""Score real-time XRPL transactions using the trained Isolation Forest model.

This script:
  1. Loads the trained model from models/isolation_forest.pkl
  2. Loads historical stats (percentiles, account rates, etc.) so it can
     compute features for new transactions in the same way as training
  3. Connects to the XRPL websocket and streams live transactions
  4. For each transaction: normalises → computes features → scores with model
  5. Prints the risk score and writes scored transactions to
     data/realtime_scored.csv in real time

Usage:
    # First, make sure the model is trained:
    python3 feature_engineering.py && python3 model.py

    # Then run this:
    python3 score_realtime.py
"""

import asyncio
import csv
import json
import math
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from xrpl.asyncio.clients import AsyncWebsocketClient
from xrpl.models.requests import Subscribe

from model import load_model, FEATURE_COLUMNS, score_transactions
from normalize import normalize_transaction
from whale_registry import WhaleRegistry
from transaction_buffer import TransactionBuffer
from alert_engine import AlertEngine
from alerts_writer import write_alert
from supabase_uploader import init_supabase, upload_scored_row

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUT_DIR = "data"
OUT_PATH = f"{OUT_DIR}/realtime_scored.csv"
FEATURED_CSV = "data/featured_transactions.csv"

# Output columns: the 10 normalised columns + 12 features + risk_score + is_anomaly
OUTPUT_COLUMNS = [
    "timestamp", "ledger_index", "tx_hash", "tx_type",
    "account", "destination", "fee", "amount_xrp", "currency", "issuer",
    *FEATURE_COLUMNS,
    "risk_score", "is_anomaly",
]


# ---------------------------------------------------------------------------
# Load historical stats needed to compute features for new transactions
# ---------------------------------------------------------------------------

def _load_historical_stats():
    """Load pre-computed stats from the training data.

    When we trained the model, features like tx_size_percentile and
    tx_rate_z_score were computed relative to the full historical dataset.
    To score new transactions consistently, we need those same reference
    values (e.g. the 99th percentile of amount_xrp, the mean tx rate).
    """
    print(f"Loading historical stats from {FEATURED_CSV} ...")
    hist = pd.read_csv(FEATURED_CSV)

    # Only XRP amounts (same filter as feature_engineering.py)
    xrp_amounts = hist["amount_xrp"].where(hist["currency"] == "XRP", 0.0).fillna(0.0)

    stats = {
        # For tx_size_percentile: we need the sorted distribution of amounts
        "amount_values": np.sort(xrp_amounts.values),
        # For is_large_tx: the 99th percentile threshold
        "p99_amount": xrp_amounts.quantile(0.99),
        # For volume_spike_ratio: the average 5-min volume from training
        "avg_5m_volume": hist["total_volume_5m"].mean(),
        # For tx_rate_z_score: mean and std of per-account tx rates
        "mean_tx_rate": hist["tx_per_minute"].mean(),
        "std_tx_rate": hist["tx_per_minute"].std(),
        # For duplicate_memo_count: known memo frequencies from training data
        # (realtime memos not seen in training get count = 0)
    }
    print(f"  -> loaded stats from {len(hist):,} historical transactions.\n")
    return stats


# ---------------------------------------------------------------------------
# Feature engineering for a single realtime transaction
# ---------------------------------------------------------------------------

# We keep a rolling buffer of recent transactions to compute time-window
# features (rolling_tx_count_5m, total_volume_5m, tx_per_minute, etc.)
_recent_buffer: list[dict] = []
_BUFFER_WINDOW_SECONDS = 300  # 5 minutes


def _decode_memos_from_ws(tx: dict) -> str:
    """Decode memo data from a websocket transaction message.

    The websocket sends memos as a list of dicts:
        [{"Memo": {"MemoData": "48656C6C6F", "MemoType": "..."}}]
    Note: PascalCase keys, nested under "Memo".
    """
    memos = tx.get("Memos") or tx.get("memos")
    if not memos:
        return ""
    if isinstance(memos, str):
        try:
            memos = json.loads(memos)
        except (json.JSONDecodeError, TypeError):
            return ""

    parts = []
    for entry in memos:
        # Websocket format nests under "Memo" key
        memo = entry.get("Memo", entry) if isinstance(entry, dict) else {}
        # Try both PascalCase (websocket) and camelCase (CSV)
        hex_data = memo.get("MemoData") or memo.get("memoData", "")
        if hex_data:
            try:
                decoded = bytes.fromhex(hex_data).decode("utf-8", errors="replace")
                parts.append(decoded)
            except (ValueError, TypeError):
                pass
    return " ".join(parts)


def _shannon_entropy(text: str) -> float:
    """Shannon entropy of a string. 0 for empty strings."""
    if not text:
        return 0.0
    freq = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def _compute_features(normalised: dict, memo_text: str, stats: dict) -> dict:
    """Compute the 12 features for a single transaction.

    Uses the historical stats for relative features (percentiles, z-scores)
    and the rolling buffer for time-window features.
    """
    now = normalised["timestamp"]
    amount = normalised["amount_xrp"] or 0.0
    account = normalised["account"]
    currency = normalised["currency"] or "XRP"

    # Only count XRP amounts for size features (same as training)
    xrp_amount = amount if currency == "XRP" else 0.0

    # --- Add this transaction to the rolling buffer ---
    _recent_buffer.append({
        "timestamp": now,
        "account": account,
        "xrp_amount": xrp_amount,
    })

    # --- Prune buffer: remove transactions older than 5 minutes ---
    if isinstance(now, pd.Timestamp) and not pd.isna(now):
        cutoff = now - pd.Timedelta(seconds=_BUFFER_WINDOW_SECONDS)
        while _recent_buffer and _recent_buffer[0]["timestamp"] < cutoff:
            _recent_buffer.pop(0)

    # --- tx_size_percentile ---
    # Where does this amount rank in the historical distribution?
    sorted_amounts = stats["amount_values"]
    percentile = np.searchsorted(sorted_amounts, xrp_amount) / len(sorted_amounts)

    # --- is_large_tx ---
    is_large = 1 if xrp_amount > stats["p99_amount"] else 0

    # --- wallet_balance_change ---
    # Approximate: sum of XRP this account has sent in the buffer window
    # (negative = net sender). Simplified for realtime.
    account_sent = sum(
        r["xrp_amount"] for r in _recent_buffer if r["account"] == account
    )
    wallet_balance_change = -account_sent

    # --- total_volume_5m ---
    # Total XRP volume across all transactions in the 5-min window
    total_volume_5m = sum(r["xrp_amount"] for r in _recent_buffer)

    # --- memo features ---
    memo_entropy = _shannon_entropy(memo_text)
    memo_length = len(memo_text)
    # For realtime, duplicate_memo_count is 0 (we don't track memo history)
    duplicate_memo_count = 0
    url_pattern = re.compile(r"https?://|www\.", re.IGNORECASE)
    contains_url = 1 if memo_text and url_pattern.search(memo_text) else 0

    # --- rolling_tx_count_5m ---
    # How many transactions from this account in the 5-min window?
    rolling_tx_count = sum(
        1 for r in _recent_buffer if r["account"] == account
    )

    # --- tx_per_minute ---
    # Average rate for this account in the current window
    if len(_recent_buffer) >= 2:
        window_minutes = max(
            (_recent_buffer[-1]["timestamp"] - _recent_buffer[0]["timestamp"]).total_seconds() / 60.0,
            1.0,
        )
    else:
        window_minutes = 1.0
    tx_per_minute = rolling_tx_count / window_minutes

    # --- tx_rate_z_score ---
    # How does this account's rate compare to the historical average?
    std = stats["std_tx_rate"]
    if std > 0:
        tx_rate_z_score = (tx_per_minute - stats["mean_tx_rate"]) / std
    else:
        tx_rate_z_score = 0.0

    # --- volume_spike_ratio ---
    avg_vol = stats["avg_5m_volume"]
    volume_spike_ratio = total_volume_5m / avg_vol if avg_vol > 0 else 0.0

    return {
        "tx_size_percentile": percentile,
        "is_large_tx": is_large,
        "wallet_balance_change": wallet_balance_change,
        "total_volume_5m": total_volume_5m,
        "memo_entropy": memo_entropy,
        "memo_length": memo_length,
        "duplicate_memo_count": duplicate_memo_count,
        "contains_url": contains_url,
        "rolling_tx_count_5m": rolling_tx_count,
        "tx_per_minute": tx_per_minute,
        "tx_rate_z_score": tx_rate_z_score,
        "volume_spike_ratio": volume_spike_ratio,
    }


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _init_csv():
    """Create output CSV with headers if it doesn't exist."""
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(OUT_PATH):
        with open(OUT_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()


def _append_scored_row(row: dict):
    """Append a single scored transaction to the CSV immediately."""
    with open(OUT_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writerow({k: row.get(k) for k in OUTPUT_COLUMNS})


# ---------------------------------------------------------------------------
# Main streaming + scoring loop
# ---------------------------------------------------------------------------

async def stream_and_score(model, stats, supabase_client=None):
    """Connect to the XRPL websocket, stream transactions, and score each one.

    For each incoming transaction:
      1. Normalise it (same as xrpl_realtime.py)
      2. Decode memos from the websocket message
      3. Compute the 12 features using historical stats + rolling buffer
      4. Score it with the trained Isolation Forest
      5. Print the result and append to CSV
    """
    async with AsyncWebsocketClient("wss://xrplcluster.com/") as client:
        # Subscribe to live transactions
        await client.send(Subscribe(streams=["transactions"]))

        count = 0
        async for msg in client:
            # Skip non-transaction messages (e.g. subscription confirmation)
            tx = msg.get("tx_json") or msg.get("transaction")
            if tx is None:
                continue

            # --- Step 1: Normalise the transaction ---
            raw = {
                **tx,
                "ledger_index": msg.get("ledger_index", tx.get("ledger_index")),
                "ledger_close_time_human": msg.get("close_time_iso") or msg.get("date"),
                "hash": msg.get("hash") or tx.get("hash"),
            }
            normalised = normalize_transaction(raw)

            # --- Step 2: Decode memos ---
            memo_text = _decode_memos_from_ws(tx)

            # --- Step 3: Compute features ---
            features = _compute_features(normalised, memo_text, stats)

            # --- Step 4: Score with the model ---
            # Build a single-row DataFrame for the model
            feature_row = pd.DataFrame([features])
            X = feature_row[FEATURE_COLUMNS].fillna(0).values

            # decision_function: more negative = more anomalous
            raw_score = model.decision_function(X)[0]
            prediction = model.predict(X)[0]  # -1 = anomaly, 1 = normal

            # Scale to 0-1 (using approximate mapping from training)
            # Scores typically range from about -0.3 (very anomalous) to 0.3 (very normal)
            risk_score = max(0.0, min(1.0, 0.5 - raw_score))
            is_anomaly = 1 if prediction == -1 else 0

            # --- Step 5: Output ---
            scored_row = {
                **normalised,
                **features,
                "risk_score": round(risk_score, 4),
                "is_anomaly": is_anomaly,
            }

            _append_scored_row(scored_row)
            upload_scored_row(supabase_client, scored_row)

            # --- Step 6: Run alert engine ---
            fired_alerts = _alert_engine.process_transaction(scored_row)
            for alert in fired_alerts:
                write_alert(alert)
                sev_colours = {"low": "\033[93m", "medium": "\033[93m",
                               "high": "\033[91m", "critical": "\033[95m"}
                col = sev_colours.get(alert.severity, "")
                print(f"  🚨 {col}[{alert.severity.upper()}]\033[0m "
                      f"{alert.alert_type} — {alert.message}")

            count += 1
            # Color-code: red for anomalies, green for normal
            risk_label = "\033[91mANOMALY\033[0m" if is_anomaly else "normal "
            print(
                f"[{count:>5}] "
                f"{risk_label}  "
                f"risk={risk_score:.3f}  "
                f"{normalised['tx_type']:20s}  "
                f"amt={normalised['amount_xrp'] or 0:>14}  "
                f"{normalised['currency'] or 'XRP':>5}  "
                f"acct={str(normalised['account'])[:12]}..."
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load the trained model
    model = load_model()

    # Load historical stats for feature computation
    stats = _load_historical_stats()

    # Build whale registry and alert engine
    print("Building whale registry...")
    _whale_registry = WhaleRegistry.build(verbose=False)
    _tx_buffer = TransactionBuffer()
    _alert_engine = AlertEngine(_whale_registry, _tx_buffer)
    print(f"  {len(_whale_registry.whale_accounts)} whales identified\n")

    # Set up outputs: CSV + Supabase
    _init_csv()
    supabase_client = init_supabase()
    print(f"Scoring live transactions → {OUT_PATH}")
    if supabase_client:
        print("Also uploading to Supabase.")
    print("Press Ctrl+C to stop.\n")

    try:
        asyncio.run(stream_and_score(model, stats, supabase_client))
    except KeyboardInterrupt:
        print(f"\nStopped. Scored data saved in {OUT_PATH}")
