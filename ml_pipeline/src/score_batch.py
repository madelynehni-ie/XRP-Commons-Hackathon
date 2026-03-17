"""Score XRPL transactions using the saved model."""

from __future__ import annotations

import joblib
import pandas as pd

from ml_data import load_all_transactions
from feature_engineering import build_features, FEATURE_COLUMNS

MODEL_PATH = "models/risk_model.joblib"


def main() -> None:
    model = joblib.load(MODEL_PATH)

    raw_df = load_all_transactions()
    feature_df = build_features(raw_df)

    x = feature_df[FEATURE_COLUMNS].copy()
    probs = model.predict_proba(x)[:, 1]

    results = feature_df[[
        "timestamp",
        "ledger_index",
        "tx_hash",
        "account",
        "currency",
        "issuer",
    ]].copy()

    results["risk_score"] = probs
    results["risk_flag"] = results["risk_score"] >= 0.8

    print(results.tail(20).to_string(index=False))


if __name__ == "__main__":
    main()