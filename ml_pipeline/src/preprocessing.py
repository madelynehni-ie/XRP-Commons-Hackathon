from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


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


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if "label" not in df.columns:
        raise ValueError("Training data must include a 'label' column.")

    x = df[FEATURE_COLUMNS].copy()
    y = df["label"].astype(int)
    return x, y


def make_train_test(x: pd.DataFrame, y: pd.Series):
    return train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )