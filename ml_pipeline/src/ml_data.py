"""Load normalized historical + realtime XRPL CSV data for ML."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
HIST_PATH = DATA_DIR / "historical_transactions.csv"
REALTIME_PATH = DATA_DIR / "realtime_transactions.csv"


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    if "ledger_index" in df.columns:
        df["ledger_index"] = pd.to_numeric(df["ledger_index"], errors="coerce")

    if "amount_xrp" in df.columns:
        df["amount_xrp"] = pd.to_numeric(df["amount_xrp"], errors="coerce")

    return df


def load_all_transactions() -> pd.DataFrame:
    """Load and combine historical + realtime normalized transaction files."""
    hist = _read_csv_if_exists(HIST_PATH)
    realtime = _read_csv_if_exists(REALTIME_PATH)

    combined = pd.concat([hist, realtime], ignore_index=True)

    if "tx_hash" in combined.columns:
        combined = combined.drop_duplicates(subset=["tx_hash"], keep="last")

    sort_cols = [c for c in ["timestamp", "ledger_index"] if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols, na_position="last")

    return combined.reset_index(drop=True)


if __name__ == "__main__":
    df = load_all_transactions()
    print(df.head())
    print(f"\nLoaded {len(df)} rows")