"""
Isolation Forest anomaly detection for XRPL transactions.

Trains an Isolation Forest model on the 12 engineered features, assigns
a risk_score (0-1, where 1 = highest risk) and an is_anomaly binary flag
to every transaction.

Usage:
    python model.py
    # -> trains model, saves to models/isolation_forest.pkl
    # -> saves scored transactions to data/scored_transactions.csv
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURED_CSV = "data/featured_transactions.csv"
SCORED_CSV = "data/scored_transactions.csv"
MODEL_PATH = "models/isolation_forest.pkl"

# These must match the features produced by feature_engineering.py
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(df: pd.DataFrame, contamination: float = 0.05, random_state: int = 42):
    """Train an Isolation Forest on the 12 feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all columns listed in FEATURE_COLUMNS.
    contamination : float
        Expected proportion of anomalies (default 5%).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    model : IsolationForest
        The fitted model.
    """
    print(f"Training Isolation Forest (contamination={contamination}) ...")

    # Extract the feature matrix, replacing any remaining NaN with 0
    X = df[FEATURE_COLUMNS].fillna(0).values

    model = IsolationForest(
        n_estimators=200,          # more trees = more stable results
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,                 # use all CPU cores
    )
    model.fit(X)

    print("  -> model trained on {:,} transactions with {} features.".format(
        X.shape[0], X.shape[1]
    ))
    return model


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_transactions(df: pd.DataFrame, model: IsolationForest) -> pd.DataFrame:
    """Score every transaction with the trained Isolation Forest.

    Adds two columns to the DataFrame:
        - risk_score : float in [0, 1], where 1 = most anomalous
        - is_anomaly : int, 1 if the model flags it as anomalous

    Returns
    -------
    df : pd.DataFrame
        The input DataFrame with two new columns appended.
    """
    X = df[FEATURE_COLUMNS].fillna(0).values

    # decision_function returns the anomaly score:
    #   - more negative = more anomalous
    #   - positive = normal
    raw_scores = model.decision_function(X)

    # Invert and scale to [0, 1] so that 1 = highest risk
    # (raw_scores are more negative for anomalies, so we negate them)
    inverted = -raw_scores
    scaler = MinMaxScaler(feature_range=(0, 1))
    risk_scores = scaler.fit_transform(inverted.reshape(-1, 1)).flatten()

    # predict() returns -1 for anomalies, 1 for normal
    predictions = model.predict(X)

    df = df.copy()
    df["risk_score"] = risk_scores
    df["is_anomaly"] = (predictions == -1).astype(int)

    return df


# ---------------------------------------------------------------------------
# Loading a saved model
# ---------------------------------------------------------------------------

def load_model(path: str = MODEL_PATH) -> IsolationForest:
    """Load a previously saved Isolation Forest model from disk.

    Parameters
    ----------
    path : str
        Path to the .pkl file (default: models/isolation_forest.pkl).

    Returns
    -------
    model : IsolationForest
    """
    print(f"Loading model from {path} ...")
    model = joblib.load(path)
    print("  -> model loaded.")
    return model


# ---------------------------------------------------------------------------
# CLI entry-point: full train + score pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── 1. Load featured transactions ─────────────────────────────────
    print(f"Loading featured transactions from {FEATURED_CSV} ...")
    df = pd.read_csv(FEATURED_CSV)
    print(f"  -> {len(df):,} rows loaded.\n")

    # ── 2. Train the model ────────────────────────────────────────────
    model = train_model(df)

    # ── 3. Save the model ─────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"  -> model saved to {MODEL_PATH}\n")

    # ── 4. Score all transactions ─────────────────────────────────────
    scored = score_transactions(df, model)

    # ── 5. Save scored transactions ───────────────────────────────────
    os.makedirs(os.path.dirname(SCORED_CSV), exist_ok=True)
    scored.to_csv(SCORED_CSV, index=False)
    print(f"Saved scored transactions to {SCORED_CSV}")
    print(f"  -> {len(scored):,} rows\n")

    # ── 6. Print summary ──────────────────────────────────────────────
    n_anomalies = scored["is_anomaly"].sum()
    n_total = len(scored)
    print("=" * 60)
    print("Anomaly Detection Summary")
    print("=" * 60)
    print(f"Total transactions : {n_total:,}")
    print(f"Flagged as anomaly : {n_anomalies:,} ({100 * n_anomalies / n_total:.2f}%)")
    print(f"Normal             : {n_total - n_anomalies:,}")
    print()

    # Risk score distribution
    print("Risk score distribution:")
    print(scored["risk_score"].describe().round(4).to_string())
    print()

    # Histogram-style buckets
    print("Risk score buckets:")
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    scored["_bucket"] = pd.cut(scored["risk_score"], bins=bins, labels=labels, include_lowest=True)
    print(scored["_bucket"].value_counts().sort_index().to_string())
    print()

    # Top 10 riskiest transactions
    print("Top 10 riskiest transactions:")
    top10 = scored.nlargest(10, "risk_score")
    display_cols = ["tx_hash", "account", "tx_type", "amount_xrp", "risk_score", "is_anomaly"]
    # Only show columns that exist
    display_cols = [c for c in display_cols if c in top10.columns]
    print(top10[display_cols].to_string(index=False))
