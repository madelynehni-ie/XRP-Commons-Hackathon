"""Score XRPL transactions using the saved Isolation Forest model."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from ml_data import load_all_transactions
from feature_engineering import build_features, FEATURE_COLUMNS

MODEL_PATH = "models/risk_model.joblib"


def normalize_risk_scores(scores: np.ndarray) -> np.ndarray:
    """
    Isolation Forest decision_function:
    - higher = more normal
    - lower = more anomalous

    Convert to 0-1 risk score where:
    - 1.0 = most risky
    - 0.0 = least risky
    """
    if len(scores) == 0:
        return scores

    max_score = scores.max()
    min_score = scores.min()

    if max_score == min_score:
        return np.zeros_like(scores, dtype=float)

    return (max_score - scores) / (max_score - min_score)


def main() -> None:
    model = joblib.load(MODEL_PATH)

    raw_df = load_all_transactions()
    feature_df = build_features(raw_df)

    x = feature_df[FEATURE_COLUMNS].copy()

    # Isolation Forest outputs
    anomaly_scores = model.decision_function(x)   # lower = more anomalous
    anomaly_preds = model.predict(x)              # -1 = anomaly, 1 = normal

    risk_scores = normalize_risk_scores(anomaly_scores)

    results = feature_df[
        [
            "timestamp",
            "ledger_index",
            "tx_hash",
            "account",
            "currency",
            "issuer",
        ]
    ].copy()

    results["anomaly_score_raw"] = anomaly_scores
    results["risk_score"] = risk_scores
    results["risk_flag"] = anomaly_preds == -1
    results["is_anomaly"] = anomaly_preds == -1

    # Optional: sort riskiest first
    results = results.sort_values("risk_score", ascending=False)

    print(results.head(20).to_string(index=False))


if __name__ == "__main__":
    main()