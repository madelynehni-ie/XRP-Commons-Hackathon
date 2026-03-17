"""Load historical XRPL data from the CSV files and normalise it into
the unified ML-ready schema (see normalize.py).

Output: data/historical_transactions.csv
"""

import os

import pandas as pd
from normalize import normalize_dataframe

DATA_DIR = "ie_dataset"
OUT_DIR = "data"
OUT_PATH = f"{OUT_DIR}/historical_transactions.csv"


def load_historical() -> pd.DataFrame:
    """Load and normalise historical transactions. Returns a DataFrame."""
    raw_transactions = pd.read_csv(f"{DATA_DIR}/transactions.csv")
    return normalize_dataframe(raw_transactions)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    transactions = load_historical()
    transactions.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(transactions)} transactions → {OUT_PATH}")
    print(f"Time range : {transactions['timestamp'].min()} → {transactions['timestamp'].max()}")
    print(f"Ledger range: {transactions['ledger_index'].min()} – {transactions['ledger_index'].max()}")
    print(f"\nTransaction types:\n{transactions['tx_type'].value_counts()}")
