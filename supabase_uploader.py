"""Upload scored transactions to Supabase so the Lovable frontend can read them.

This module provides a simple interface to push scored rows to a Supabase
table called `scored_transactions`. It reads credentials from environment
variables (see .env.example).

Usage from other modules:
    from supabase_uploader import init_supabase, upload_scored_row

    client = init_supabase()          # returns None if env vars missing
    upload_scored_row(client, row)     # no-op if client is None
"""

import os

from dotenv import load_dotenv

load_dotenv()

_client = None


def init_supabase():
    """Initialise and return a Supabase client, or None if not configured."""
    global _client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        print("[supabase] SUPABASE_URL / SUPABASE_KEY not set — skipping Supabase upload.")
        return None

    from supabase import create_client

    _client = create_client(url, key)
    print(f"[supabase] Connected to {url}")
    return _client


def upload_scored_row(client, row: dict):
    """Insert a single scored transaction row into Supabase.

    Silently skips if client is None (Supabase not configured).
    """
    if client is None:
        return

    # Build the payload — same columns as the CSV output
    payload = {
        "timestamp": str(row.get("timestamp", "")),
        "ledger_index": row.get("ledger_index"),
        "tx_hash": row.get("tx_hash"),
        "tx_type": row.get("tx_type"),
        "account": row.get("account"),
        "destination": row.get("destination"),
        "fee": row.get("fee"),
        "amount_xrp": row.get("amount_xrp"),
        "currency": row.get("currency"),
        "issuer": row.get("issuer"),
        "tx_size_percentile": row.get("tx_size_percentile"),
        "is_large_tx": row.get("is_large_tx"),
        "wallet_balance_change": row.get("wallet_balance_change"),
        "total_volume_5m": row.get("total_volume_5m"),
        "memo_entropy": row.get("memo_entropy"),
        "memo_length": row.get("memo_length"),
        "duplicate_memo_count": row.get("duplicate_memo_count"),
        "contains_url": row.get("contains_url"),
        "rolling_tx_count_5m": row.get("rolling_tx_count_5m"),
        "tx_per_minute": row.get("tx_per_minute"),
        "tx_rate_z_score": row.get("tx_rate_z_score"),
        "volume_spike_ratio": row.get("volume_spike_ratio"),
        "risk_score": row.get("risk_score"),
        "is_anomaly": row.get("is_anomaly"),
    }

    # Convert any NaN / None numeric values to None for JSON serialisation
    for k, v in payload.items():
        try:
            if v != v:  # NaN check
                payload[k] = None
        except TypeError:
            pass

    try:
        client.table("scored_transactions").insert(payload).execute()
    except Exception as e:
        print(f"[supabase] Insert failed: {e}")
