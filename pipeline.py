"""
XRPL Anomaly Detection — Full ML Pipeline
==========================================
Runs end-to-end from raw transaction data to formatted alert output.

Usage
-----
# From CSV files (your dataset):
    python3 pipeline.py --transactions ie_dataset/transactions.csv

# From the live realtime CSV (while xrpl_realtime.py is running):
    python3 pipeline.py --transactions data/realtime_transactions.csv --live

# Save alerts to JSON or CSV:
    python3 pipeline.py --transactions ie_dataset/transactions.csv --out alerts.json
    python3 pipeline.py --transactions ie_dataset/transactions.csv --out alerts.csv

# Save trained model for reuse:
    python3 pipeline.py --transactions ie_dataset/transactions.csv --save-model models/risk_model.joblib

# Load a previously trained model instead of retraining:
    python3 pipeline.py --transactions data/realtime_transactions.csv --load-model models/risk_model.joblib
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

# ── Make src imports work from any working directory ──────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize_dataframe
from src.feature_engineering import build_features, FEATURE_COLUMNS
from src.model import build_model, calibrate_contamination, IsolationForestPipeline
from src.preprocessing import temporal_train_test_split, get_X
from src.utils import shannon_entropy, contains_url


# ── Signal explainers — translate feature numbers into plain English ──────────
SIGNAL_EXPLAINERS = {
    "tx_rate_z_score":        lambda v: f"Tx rate {v:.1f}σ above account's normal" if abs(v) > 2 else None,
    "tx_rate_mad_z":          lambda v: f"Tx rate MAD z-score {v:.1f} (robust spike)" if abs(v) > 2 else None,
    "volume_spike_ratio":     lambda v: f"Volume {v:.1f}× above 5-min baseline" if v > 5 else None,
    "offer_cancel_ratio":     lambda v: f"Offer cancel ratio {v:.0%} — possible spoofing" if v > 0.5 else None,
    "dest_concentration":     lambda v: f"Fan-out to many destinations (conc={v:.2f})" if v < 0.3 else None,
    "failed_tx_ratio":        lambda v: f"Failed tx ratio {v:.0%} — probing or bot" if v > 0.2 else None,
    "dormancy_gap":           lambda v: f"Dormant {v:.0f} ledgers then suddenly active" if v > 2000 else None,
    "memo_entropy":           lambda v: f"Memo entropy {v:.2f} — repetitive spam content" if v < 0.5 and v > 0 else (
                                        f"Memo entropy {v:.2f} — possible encoded payload" if v > 4.5 else None),
    "contains_url":           lambda v: "Memo contains URL" if v == 1 else None,
    "duplicate_memo_count":   lambda v: f"Same memo repeated {v:.0f} times" if v > 3 else None,
    "tx_size_percentile_local": lambda v: f"Tx size at {v:.0%} of this account's history" if v > 0.95 else None,
    "fee_rate":               lambda v: f"Fee rate {v:.4f} — disproportionate to amount" if v > 0.001 else None,
}

SEVERITY_COLOR = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
RISK_LABELS = [(0.8, "Extreme"), (0.6, "Very high"), (0.4, "High"), (0.2, "Moderate"), (0.0, "Low")]

ACTION = {
    "HIGH":   "Investigate immediately → https://livenet.xrpl.org/accounts/{account}",
    "MEDIUM": "Monitor closely. Flag for review if pattern persists in next 30 minutes.",
    "LOW":    "Note for context. No immediate action required.",
}


def risk_label(score: float) -> str:
    for threshold, label in RISK_LABELS:
        if score >= threshold:
            return label
    return "Low"


def build_signals(row: dict) -> list[str]:
    signals = []
    for feat, fn in SIGNAL_EXPLAINERS.items():
        val = row.get(feat)
        if val is None:
            continue
        try:
            result = fn(float(val))
            if result:
                signals.append(result)
        except (TypeError, ValueError):
            pass
    return signals


# ── Step 1: Load & normalise ───────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print(f"\n[1/5] Loading transactions from {path}...")
    raw = pd.read_csv(path)
    print(f"      Raw rows: {len(raw)}  columns: {len(raw.columns)}")

    # normalize_dataframe handles both CSV schemas automatically
    df = normalize_dataframe(raw)
    df = df.dropna(subset=["account", "timestamp"]).reset_index(drop=True)
    df["ledger_index"] = df["ledger_index"].fillna(0).astype(int)
    # Convert category dtypes to str so feature_engineering can process them
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str)

    print(f"      Normalised: {len(df)} rows")
    print(f"      Time range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"      Accounts:   {df['account'].nunique()}")
    print(f"      Tx types:   {df['tx_type'].value_counts().to_dict()}")
    return df


# ── Step 2: Feature engineering ───────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[2/5] Building features...")
    features = build_features(df)
    print(f"      Feature rows: {len(features)}")
    print(f"      Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")

    # Sanity check
    nan_count = features[FEATURE_COLUMNS].isna().sum().sum()
    inf_count = np.isinf(features[FEATURE_COLUMNS].values).sum()
    print(f"      NaN: {nan_count}  Inf: {inf_count}  (both should be 0)")
    return features


# ── Step 3: Train model ────────────────────────────────────────────────────────

def train(features: pd.DataFrame, model_path: str | None = None) -> tuple:
    print(f"\n[3/5] Training Isolation Forest...")

    train_df, test_df = temporal_train_test_split(features, train_frac=0.8)
    X_train = get_X(train_df)
    X_test  = get_X(test_df)

    print(f"      Train: {len(train_df)} rows  "
          f"(ledger {train_df['ledger_index'].min()} → {train_df['ledger_index'].max()})")
    print(f"      Test:  {len(test_df)} rows  "
          f"(ledger {test_df['ledger_index'].min()} → {test_df['ledger_index'].max()})")

    model = build_model()
    model.fit(X_train)
    print(f"      Self-calibrated contamination: {model.contamination_used_:.4f} "
          f"({model.contamination_used_*100:.1f}% of windows expected anomalous)")

    # Validation
    train_scores = model.decision_function(X_train)
    test_scores  = model.decision_function(X_test)
    train_rate   = (model.predict(X_train) == 1).mean()
    test_rate    = (model.predict(X_test)  == 1).mean()
    ks_stat, ks_p = stats.ks_2samp(train_scores, test_scores)
    separation   = np.percentile(test_scores, 95) - np.median(test_scores)

    print(f"\n      Validation:")
    print(f"        Train flag rate : {train_rate:.3f}")
    print(f"        Test flag rate  : {test_rate:.3f}  "
          f"{'✓' if abs(train_rate - test_rate) < 0.05 else '⚠ drift'}")
    print(f"        KS p-value      : {ks_p:.4f}  "
          f"{'✓ same distribution' if ks_p > 0.05 else '⚠ distribution shift'}")
    print(f"        Score separation: {separation:.4f}  "
          f"{'✓' if separation > 0.05 else '⚠ low — anomalies not well separated'}")

    # 5-fold time-series CV
    print(f"\n      Time-series CV (5 folds):")
    X_all = get_X(features)
    fold_conts, fold_rates = [], []
    for fold, (tr, te) in enumerate(TimeSeriesSplit(n_splits=5).split(X_all)):
        from sklearn.preprocessing import RobustScaler
        sc = RobustScaler()
        Xtr = sc.fit_transform(X_all[tr])
        cont = calibrate_contamination(Xtr)
        m = build_model(contamination=cont)
        m.fit(X_all[tr])
        rate = (m.predict(X_all[te]) == 1).mean()
        fold_conts.append(cont); fold_rates.append(rate)
        print(f"        Fold {fold+1}: contamination={cont:.4f}  flag_rate={rate:.3f}")
    print(f"        Mean contamination: {np.mean(fold_conts):.4f} ± {np.std(fold_conts):.4f}")
    print(f"        Mean flag rate:     {np.mean(fold_rates):.3f} ± {np.std(fold_rates):.4f}")
    print(f"        Stable across time: {'✓ YES' if np.std(fold_rates) < 0.025 else '⚠ NO'}")

    if model_path:
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)
        joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS,
                     "contamination": model.contamination_used_}, model_path)
        print(f"\n      Model saved → {model_path}")

    return model, test_df


def load_trained_model(model_path: str):
    print(f"\n[3/5] Loading model from {model_path}...")
    bundle = joblib.load(model_path)
    print(f"      contamination={bundle['contamination']:.4f}  "
          f"features={len(bundle['feature_columns'])}")
    return bundle["model"]


# ── Step 4: Score & generate alerts ───────────────────────────────────────────

def score_and_alert(
    model: IsolationForestPipeline,
    features: pd.DataFrame,
    score_df: pd.DataFrame | None = None,
) -> list[dict]:
    print(f"\n[4/5] Scoring and generating alerts...")

    target = score_df if score_df is not None else features
    X = get_X(target)

    scores = model.decision_function(X)
    flags  = model.predict(X)

    target = target.copy()
    target["anomaly_score"] = scores
    target["is_anomalous"]  = flags == 1

    # Assign severity using model's own contamination — no hardcoded thresholds
    cont = model.contamination_used_
    def assign_severity(score, is_anomalous):
        if not is_anomalous:
            return None
        if score >= 1 - cont * 0.2:
            return "HIGH"
        if score >= 1 - cont:
            return "MEDIUM"
        return "LOW"

    target["severity"] = target.apply(
        lambda r: assign_severity(r["anomaly_score"], r["is_anomalous"]), axis=1
    )

    anomalies = target[target["severity"].notna()].sort_values(
        ["severity", "anomaly_score"], ascending=[True, False]
    )

    # Build alert objects
    alerts = []
    for _, row in anomalies.iterrows():
        row_dict  = row.to_dict()
        signals   = build_signals(row_dict)
        sev       = row_dict["severity"]
        score_val = float(row_dict["anomaly_score"])
        account   = str(row_dict.get("account", ""))

        if not signals:
            signals = [f"Multivariate anomaly — score {score_val:.3f} "
                       f"(unusual combination of {len(FEATURE_COLUMNS)} features)"]

        alerts.append({
            "severity":      sev,
            "anomaly_score": round(score_val, 4),
            "risk_label":    risk_label(score_val),
            "account":       account,
            "ledger_index":  int(row_dict.get("ledger_index", 0)),
            "close_time":    str(row_dict.get("close_time", "")),
            "signals":       signals,
            "action":        ACTION[sev].replace("{account}", account),
            "explorer_url":  f"https://livenet.xrpl.org/accounts/{account}",
            "features": {
                k: round(float(row_dict[k]), 4) if row_dict.get(k) is not None else None
                for k in FEATURE_COLUMNS if k in row_dict
            },
        })

    high   = sum(1 for a in alerts if a["severity"] == "HIGH")
    medium = sum(1 for a in alerts if a["severity"] == "MEDIUM")
    low    = sum(1 for a in alerts if a["severity"] == "LOW")
    print(f"      Scored {len(target)} rows → {len(alerts)} alerts  "
          f"(HIGH={high}  MEDIUM={medium}  LOW={low})")
    return alerts


# ── Step 5: Format output ──────────────────────────────────────────────────────

def print_alerts(alerts: list[dict]):
    print(f"\n[5/5] Alert report")
    print(f"\n{'='*70}")
    print(f"  XRPL SENTINEL — ANOMALY REPORT")
    print(f"  {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*70}")

    high   = [a for a in alerts if a["severity"] == "HIGH"]
    medium = [a for a in alerts if a["severity"] == "MEDIUM"]
    low    = [a for a in alerts if a["severity"] == "LOW"]
    print(f"  Summary: {len(high)} HIGH  {len(medium)} MEDIUM  {len(low)} LOW")
    print(f"{'='*70}\n")

    for alert in alerts:
        sev   = alert["severity"]
        icon  = SEVERITY_COLOR.get(sev, "")
        print(f"  {icon} {sev}  [{alert['risk_label']}]  score={alert['anomaly_score']:.3f}")
        print(f"  Account      : {alert['account']}")
        print(f"  Ledger       : {alert['ledger_index']}")
        print(f"  Time         : {alert['close_time']}")
        print(f"  Explorer     : {alert['explorer_url']}")
        print()
        print(f"  Why flagged:")
        for s in alert["signals"]:
            print(f"    • {s}")
        print()

        # Show most relevant features
        f = alert["features"]
        relevant = {k: v for k, v in f.items() if v is not None and v != 0.0}
        if relevant:
            top_feats = list(relevant.items())[:6]
            feat_str  = "  ".join(f"{k}={v}" for k, v in top_feats)
            print(f"  Features: {feat_str}")
            print()

        print(f"  Action: {alert['action']}")
        print(f"  {'-'*66}\n")


def save_alerts(alerts: list[dict], path: str):
    if path.endswith(".csv"):
        rows = []
        for a in alerts:
            row = {k: v for k, v in a.items() if k != "features"}
            row["signals"] = " | ".join(a["signals"])
            row.update({f"feat_{k}": v for k, v in a["features"].items()})
            rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        with open(path, "w") as f:
            json.dump(alerts, f, indent=2, default=str)
    print(f"\n  Saved {len(alerts)} alerts → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run(
    transactions_path: str,
    out_path: str | None       = None,
    save_model_path: str | None = None,
    load_model_path: str | None = None,
    live: bool                  = False,
) -> list[dict]:

    # 1. Load
    df = load_data(transactions_path)

    # 2. Features
    features = engineer_features(df)

    # 3. Train or load model
    if load_model_path:
        model = load_trained_model(load_model_path)
        score_df = features   # score everything
    else:
        model, test_df = train(features, model_path=save_model_path)
        # In live mode score everything; otherwise score only the test set
        score_df = features if live else test_df

    # 4. Score & alert
    alerts = score_and_alert(model, features, score_df=score_df)

    # 5. Output
    print_alerts(alerts)
    if out_path:
        save_alerts(alerts, out_path)

    return alerts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XRPL anomaly detection pipeline")
    parser.add_argument("--transactions",  required=True,        help="Path to transactions CSV")
    parser.add_argument("--out",           default=None,         help="Save alerts to .json or .csv")
    parser.add_argument("--save-model",    default=None,         help="Save trained model to .joblib")
    parser.add_argument("--load-model",    default=None,         help="Load existing model instead of training")
    parser.add_argument("--live",          action="store_true",  help="Score all rows, not just test set")
    args = parser.parse_args()

    run(
        transactions_path=args.transactions,
        out_path=args.out,
        save_model_path=args.save_model,
        load_model_path=args.load_model,
        live=args.live,
    )
