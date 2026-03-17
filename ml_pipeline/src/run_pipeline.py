"""Combine normalized XRPL data and build ML-ready features."""

from __future__ import annotations

from pathlib import Path

from ml_data import load_all_transactions
from feature_engineering import build_features

DATA_DIR = Path("data")
FEATURES_PATH = DATA_DIR / "features.csv"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    raw_df = load_all_transactions()
    print(f"Loaded combined rows: {len(raw_df)}")

    features = build_features(raw_df)
    features.to_csv(FEATURES_PATH, index=False)

    print(f"Saved features to {FEATURES_PATH}")
    print(features.head(20).to_string())


if __name__ == "__main__":
    main()