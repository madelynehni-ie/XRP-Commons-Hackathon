import joblib

from src.queries import load_transactions
from src.feature_engineering import build_features
from src.preprocessing import FEATURE_COLUMNS
from src.config import Config


def main():
    config = Config()

    model = joblib.load(config.model_path)

    raw_df = load_transactions(limit=5000)
    feature_df = build_features(raw_df)

    x = feature_df[FEATURE_COLUMNS].copy()
    probs = model.predict_proba(x)[:, 1]

    results = feature_df[["account", "ledger_index"]].copy()
    results["risk_score"] = probs
    results["risk_flag"] = results["risk_score"] >= config.high_risk_threshold

    print(results.tail(20).to_string(index=False))


if __name__ == "__main__":
    main()