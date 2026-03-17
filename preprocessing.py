"""
Preprocessing and train/test splitting.

Key change from original:
- Replaced random train_test_split with a TEMPORAL split
- For time-series data, random splitting leaks the future into training
- We train on the first 80% of ledger history, test on the last 20%
- No labels needed — IForest is unsupervised, split_xy is kept for
  compatibility if pseudo-labels are ever added
"""
from typing import Tuple
import numpy as np
import pandas as pd


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


def temporal_train_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by time, not randomly.

    Sorts by ledger_index (chronological order) then takes the first
    train_frac as train and the remainder as test.

    This is the correct way to split time-series data — the model learns
    from the past and is evaluated on the future, which is how it will
    actually be used in production.
    """
    df_sorted = df.sort_values("ledger_index").reset_index(drop=True)
    cutoff = int(len(df_sorted) * train_frac)
    train = df_sorted.iloc[:cutoff].copy()
    test  = df_sorted.iloc[cutoff:].copy()
    return train, test


def get_X(df: pd.DataFrame) -> np.ndarray:
    """Extract feature matrix as numpy array."""
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    return df[cols].fillna(0.0).replace([np.inf, -np.inf], 0.0).values


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Kept for compatibility if pseudo-labels are added later.
    Raises clearly if label column is missing.
    """
    if "label" not in df.columns:
        raise ValueError(
            "No 'label' column found. For unsupervised anomaly detection "
            "use get_X() instead. If you want supervised training, add "
            "pseudo-labels by running the IForest first and using its "
            "top anomaly scores as positive examples."
        )
    x = df[[c for c in FEATURE_COLUMNS if c in df.columns]].copy()
    y = df["label"].astype(int)
    return x, y


def make_train_test(x: pd.DataFrame, y: pd.Series):
    """
    Kept for compatibility. Use temporal_train_test_split instead for
    any time-ordered data.
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(
        x, y, test_size=0.2, random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )
