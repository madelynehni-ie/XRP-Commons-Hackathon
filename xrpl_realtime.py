"""Stream real-time XRPL transactions and normalise them into the same
unified ML-ready schema as the historical data (see normalize.py).

Each transaction is appended to the CSV as it arrives, so the ML pipeline
can read from data/realtime_transactions.csv at any time — even while this
script is still running.

Usage:
    python3 xrpl_realtime.py        # streams until you press Ctrl+C
"""

import asyncio
import csv
import os

import pandas as pd
from xrpl.asyncio.clients import AsyncWebsocketClient
from xrpl.models.requests import Subscribe

from normalize import normalize_transaction

OUT_DIR = "data"
OUT_PATH = f"{OUT_DIR}/realtime_transactions.csv"

# These are the column names that match the historical CSV schema exactly,
# so both files can be read and concatenated by the ML pipeline.
COLUMNS = [
    "timestamp", "ledger_index", "tx_hash", "tx_type",
    "account", "destination", "fee", "amount_xrp", "currency", "issuer",
]


def _init_csv():
    """Create the output directory and CSV file with headers if it doesn't
    exist yet. If it already exists, we leave it alone so new rows get
    appended to previous runs."""
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(OUT_PATH):
        with open(OUT_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()


def _append_row(row: dict):
    """Append a single normalised transaction to the CSV file immediately.
    This is what allows the ML pipeline to read the file while the stream
    is still running — each row is flushed to disk as it arrives."""
    with open(OUT_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writerow(row)


async def stream():
    """Connect to the XRPL websocket and stream live transactions.

    For each incoming transaction:
      1. Extract the transaction fields from the websocket message
      2. Normalise it into the unified 10-column schema (via normalize.py)
      3. Append it to the CSV immediately (so the ML pipeline can read it)
      4. Print a summary line to the terminal
    """
    async with AsyncWebsocketClient("wss://xrplcluster.com/") as client:
        # Subscribe to the "transactions" stream — this tells the XRPL node
        # to push every validated transaction to us in real time.
        await client.send(Subscribe(streams=["transactions"]))

        count = 0
        async for msg in client:
            # Each websocket message wraps the transaction inside a
            # "transaction" key. Skip messages that aren't transactions
            # (e.g. subscription confirmations).
            tx = msg.get("tx_json") or msg.get("transaction")
            if tx is None:
                continue

            # The websocket nests tx fields inside msg["transaction"],
            # but puts some fields (ledger_index, hash, date) on the
            # outer message. Merge them into a flat dict so the
            # normaliser can find everything.
            raw = {
                **tx,
                "ledger_index": msg.get("ledger_index", tx.get("ledger_index")),
                "ledger_close_time_human": msg.get("close_time_iso") or msg.get("date"),
                "hash": msg.get("hash") or tx.get("hash"),
            }

            # Convert to the unified 10-column schema (same as historical)
            normalised = normalize_transaction(raw)

            # Write to CSV immediately so the ML pipeline can read it
            _append_row(normalised)

            count += 1
            print(
                f"[{count}] "
                f"{normalised['timestamp']}  "
                f"{normalised['tx_type']:20s}  "
                f"{normalised['amount_xrp'] or '':>14}  "
                f"{normalised['currency']:>5}"
            )


def get_realtime_df() -> pd.DataFrame:
    """Read the realtime CSV back as a DataFrame.
    Can be called from the ML pipeline at any time, even while the
    stream is running, because rows are flushed to disk on arrival."""
    if not os.path.exists(OUT_PATH):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(OUT_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["tx_type"] = df["tx_type"].astype("category")
    return df


if __name__ == "__main__":
    # Create the CSV with headers if it doesn't exist yet
    _init_csv()
    print(f"Streaming live transactions → {OUT_PATH}")
    print("Press Ctrl+C to stop.\n")
    try:
        asyncio.run(stream())
    except KeyboardInterrupt:
        print(f"\nStopped. Data saved in {OUT_PATH}")
