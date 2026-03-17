"""
Training script for the Isolation Forest anomaly detector.

Key changes from original:
- No labels required — IForest is unsupervised
- Uses temporal train/test split instead of random split
- Validation uses anomaly rate stability across folds + KS drift test
  instead of classification_report (which needs labels)
- Saves contamination and feature list alongside the model
"""
import os
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

from src.queries import load_transactions
from src.feature_engineering import build_features
from src.preprocessing import FEATURE_COLUMNS, temporal_train_test_split, get_X
from src.model import build_model, calibrate_contamination
from src.config import Config


def validate_model(model, X_train: np.ndarray, X_test: np.ndarray) -> dict:
    """
    Validate an unsupervised model without labels.

    Three checks:
    1. Anomaly rate stability — does the model flag a consistent fraction
       on unseen data? Should be close to the calibrated contamination.
    2. Score separation — anomalies should score clearly higher than normals.
    3. KS drift — are train and test from the same distribution?
       If not, the model may not generalise.
    """
    train_scores = model.decision_function(X_train)
    test_scores  = model.decision_function(X_test)

    train_flag_rate = (model.predict(X_train) == 1).mean()
    test_flag_rate  = (model.predict(X_test)  == 1).mean()

    ks_stat, ks_p = stats.ks_2samp(train_scores, test_scores)

    # Score separation: how far are the top 5% from the median?
    top5_mean   = np.percentile(test_scores, 95)
    median_score = np.median(test_scores)
    separation  = top5_mean - median_score

    return {
        "train_flag_rate":  round(float(train_flag_rate), 4),
        "test_flag_rate":   round(float(test_flag_rate),  4),
        "rate_drift":       round(abs(train_flag_rate - test_flag_rate), 4),
        "ks_stat":          round(float(ks_stat), 4),
        "ks_p_value":       round(float(ks_p),    4),
        "score_separation": round(float(separation), 4),
        "test_score_mean":  round(float(test_scores.mean()), 4),
        "test_score_std":   round(float(test_scores.std()),  4),
    }


def timeseries_cv(feature_df: pd.DataFrame, n_splits: int = 5) -> dict:
    """
    Time-series cross-validation across n_splits folds.
    Reports contamination stability — if the model self-calibrates to
    very different contamination values across folds, it's unstable.
    """
    X = get_X(feature_df)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_conts, fold_rates = [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        cont = calibrate_contamination(X_tr_s)
        m = build_model(contamination=cont)
        m.fit(X_tr)
        rate = (m.predict(X_te) == 1).mean()
        fold_conts.append(cont)
        fold_rates.append(rate)
        print(f"  Fold {fold+1}: cont={cont:.4f}  test_flag_rate={rate:.3f}")

    return {
        "mean_contamination": round(float(np.mean(fold_conts)), 4),
        "std_contamination":  round(float(np.std(fold_conts)),  4),
        "mean_flag_rate":     round(float(np.mean(fold_rates)),  4),
        "std_flag_rate":      round(float(np.std(fold_rates)),   4),
        "stable":             bool(np.std(fold_rates) < 0.025),
    }


def main():
    config = Config()

    print("Loading transactions...")
    raw_df = load_transactions()

    print("Building features...")
    feature_df = build_features(raw_df)
    print(f"  {len(feature_df)} rows  |  {len(FEATURE_COLUMNS)} features")

    # ── Temporal split ────────────────────────────────────────────────────────
    print("\nSplitting (temporal — train on past, test on future)...")
    train_df, test_df = temporal_train_test_split(feature_df, train_frac=0.8)
    X_train = get_X(train_df)
    X_test  = get_X(test_df)
    print(f"  Train: ledger {train_df.ledger_index.min()} → {train_df.ledger_index.max()}  ({len(train_df)} rows)")
    print(f"  Test:  ledger {test_df.ledger_index.min()}  → {test_df.ledger_index.max()}  ({len(test_df)} rows)")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nTraining Isolation Forest (contamination = self-calibrated)...")
    model = build_model()
    model.fit(X_train)
    print(f"  Calibrated contamination: {model.contamination_used_:.4f} ({model.contamination_used_*100:.1f}%)")

    # ── Validate ──────────────────────────────────────────────────────────────
    print("\nValidating...")
    metrics = validate_model(model, X_train, X_test)
    for k, v in metrics.items():
        flag = ""
        if k == "rate_drift"  and v > 0.05:  flag = "  ⚠ high drift"
        if k == "ks_p_value"  and v < 0.05:  flag = "  ⚠ distribution shift"
        if k == "score_separation" and v < 0.1: flag = "  ⚠ low separation"
        print(f"  {k:22}: {v}{flag}")

    # ── Time-series CV ────────────────────────────────────────────────────────
    print("\nTime-series cross-validation (5 folds)...")
    cv_results = timeseries_cv(feature_df)
    print(f"  Contamination: {cv_results['mean_contamination']:.4f} ± {cv_results['std_contamination']:.4f}")
    print(f"  Flag rate:     {cv_results['mean_flag_rate']:.3f} ± {cv_results['std_flag_rate']:.4f}")
    print(f"  Stable:        {'✓ YES' if cv_results['stable'] else '⚠ NO — consider more data'}")

    # ── Top anomalies preview ─────────────────────────────────────────────────
    print("\nTop anomalies in test set:")
    test_scores = model.decision_function(X_test)
    test_df = test_df.copy()
    test_df["anomaly_score"] = test_scores
    test_df["flagged"] = model.predict(X_test)
    top = test_df[test_df["flagged"] == 1].sort_values("anomaly_score", ascending=False).head(10)
    if len(top):
        print(top[["account", "ledger_index", "anomaly_score"] +
                   [c for c in FEATURE_COLUMNS if c in top.columns][:4]].to_string(index=False))
    else:
        print("  No anomalies flagged in test set")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
    joblib.dump({
        "model":               model,
        "feature_columns":     FEATURE_COLUMNS,
        "contamination":       model.contamination_used_,
        "validation_metrics":  metrics,
        "cv_results":          cv_results,
        "trained_on_rows":     len(train_df),
        "trained_at":          pd.Timestamp.utcnow().isoformat(),
    }, config.model_path)
    print(f"\nSaved → {config.model_path}")


if __name__ == "__main__":
    main()
