"""Train a simple v1 XRPL risk model."""

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml_data import load_all_transactions
from feature_engineering import build_features, FEATURE_COLUMNS

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "risk_model.joblib"


def main() -> None:
    raw_df = load_all_transactions()
    feature_df = build_features(raw_df)

    if "label" not in raw_df.columns and "label" not in feature_df.columns:
        print("No 'label' column found.")
        print("Features were built successfully, but supervised training cannot run yet.")
        print("\nPreview:")
        print(feature_df.head(20).to_string())
        return

    if "label" not in feature_df.columns and "label" in raw_df.columns:
        feature_df["label"] = raw_df["label"].values

    x = feature_df[FEATURE_COLUMNS].copy()
    y = feature_df["label"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, preds, digits=4))

    try:
        auc = roc_auc_score(y_test, probs)
        print(f"ROC AUC: {auc:.4f}")
    except ValueError:
        print("ROC AUC unavailable.")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()