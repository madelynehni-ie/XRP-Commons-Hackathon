"""
Shared normalisation logic.

Both the historical CSV loader and the real-time websocket stream call
`normalize_transaction(row)` so they produce identical DataFrames that
can be fed straight into feature engineering / model training.

Unified schema
--------------
timestamp      : datetime64[ns, UTC]  – ledger close time
ledger_index   : int                  – ledger sequence number
tx_hash        : str                  – transaction hash
tx_type        : str                  – e.g. Payment, OfferCreate
account        : str                  – sender / initiator
destination    : str | NaN            – receiver (Payments only)
fee            : int                  – fee in drops
amount_xrp     : float | NaN         – value in XRP (drops ÷ 1e6 for native)
currency       : str                  – "XRP" for native, token code otherwise
issuer         : str | NaN            – token issuer (NaN for XRP)
"""

import json
import math

import pandas as pd

DROPS_PER_XRP = 1_000_000


# ── helpers ──────────────────────────────────────────────────────────

def _parse_amount(raw) -> tuple[float | None, str, str | None]:
    """Return (value_in_xrp, currency, issuer) from an XRPL amount field.

    XRPL amounts are either:
      - a string of drops  ("1000000" = 1 XRP)   ← native
      - a dict / JSON with value, currency, issuer ← issued token
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None, "XRP", None

    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("{"):
            raw = json.loads(raw)
        else:
            # plain drops string
            try:
                return int(raw) / DROPS_PER_XRP, "XRP", None
            except ValueError:
                return None, "XRP", None

    if isinstance(raw, (int, float)):
        return float(raw) / DROPS_PER_XRP, "XRP", None

    if isinstance(raw, dict):
        value = raw.get("value")
        currency = raw.get("currency", "XRP")
        issuer = raw.get("issuer")
        if currency == "XRP" or (issuer is None and currency == "XRP"):
            try:
                return float(value) / DROPS_PER_XRP, "XRP", None
            except (TypeError, ValueError):
                return None, "XRP", None
        try:
            return float(value), currency, issuer
        except (TypeError, ValueError):
            return None, currency, issuer

    return None, "XRP", None


def _pick_amount(row: dict):
    """Choose the best amount field depending on transaction type.
    Checks both snake_case (CSV) and PascalCase (websocket) field names."""
    for field in ("amount", "Amount", "deliver_max", "DeliverMax",
                  "taker_gets", "TakerGets"):
        val = row.get(field)
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            return val
    return None


def _nonan(val):
    """Return None if val is NaN, otherwise val."""
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


# ── public API ───────────────────────────────────────────────────────

def normalize_transaction(row: dict) -> dict:
    """Turn a single raw transaction (CSV row or websocket msg) into the
    unified flat schema described above."""
    amount_xrp, currency, issuer = _parse_amount(_pick_amount(row))

    # Timestamp: CSV has "ledger_close_time_human", websocket has "date" or
    # we fall back to the outer message's close time.
    ts_raw = (
        row.get("ledger_close_time_human")
        or row.get("date")
        or row.get("timestamp")
    )
    try:
        ts = pd.Timestamp(ts_raw, tz="UTC")
    except Exception:
        ts = pd.NaT

    return {
        "timestamp": ts,
        "ledger_index": int(row.get("ledger_index", 0)),
        "tx_hash": _nonan(row.get("hash")) or _nonan(row.get("tx_hash")) or row.get("txn_signature"),
        "tx_type": row.get("transaction_type") or row.get("TransactionType"),
        "account": row.get("account") or row.get("Account"),
        "destination": row.get("destination") or row.get("Destination"),
        "fee": int(row.get("fee") or row.get("Fee") or 0),
        "amount_xrp": amount_xrp,
        "currency": currency,
        "issuer": issuer,
    }


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise every row of a raw CSV DataFrame."""
    records = [normalize_transaction(row) for row in df.to_dict(orient="records")]
    out = pd.DataFrame(records)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["tx_type"] = out["tx_type"].astype("category")
    return out
