"""
Score a batch of transactions and print the highest-risk ones.

Works with the new IsolationForestPipeline — no labels needed.
predict_proba(x)[:, 1] returns anomaly score in [0, 1].
"""
import joblib
import pandas as pd

from src.queries import load_transactions
from src.feature_engineering import build_features
from src.preprocessing import FEATURE_COLUMNS, get_X
from src.config import Config


def score(raw_df: pd.DataFrame | None = None, limit: int = 5000) -> pd.DataFrame:
    """
    Score transactions and return a DataFrame with anomaly_score and risk_flag.
    Pass raw_df directly to avoid a database query (useful for live stream scoring).
    """
    config = Config()
    bundle = joblib.load(config.model_path)
    model  = bundle["model"]

    if raw_df is None:
        raw_df = load_transactions(limit=limit)

    feature_df = build_features(raw_df)
    X = get_X(feature_df)

    scores = model.decision_function(X)          # [0, 1] — higher = more anomalous
    flags  = model.predict(X)                    # 1 = anomaly, 0 = normal

    results = feature_df[["account", "ledger_index", "close_time"]].copy()
    results["anomaly_score"] = scores
    results["risk_flag"]     = flags == 1

    # Add top contributing features for explainability
    feat_cols = [c for c in FEATURE_COLUMNS if c in feature_df.columns]
    results[feat_cols] = feature_df[feat_cols].values

    return results.sort_values("anomaly_score", ascending=False)


def main():
    results = score()
    flagged = results[results["risk_flag"]]
    print(f"Total scored: {len(results)}  |  Flagged: {len(flagged)}")
    print(f"\nTop 20 anomalies:")
    print(
        flagged.head(20)[
            ["account", "ledger_index", "anomaly_score",
             "tx_rate_mad_z", "volume_spike_ratio", "dest_concentration",
             "offer_cancel_ratio", "dormancy_gap"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
