import os
import joblib
from sklearn.metrics import classification_report, roc_auc_score

from src.queries import load_transactions
from src.feature_engineering import build_features
from src.preprocessing import split_xy, make_train_test
from src.model import build_model
from src.config import Config


def main():
    config = Config()

    print("Loading transactions...")
    raw_df = load_transactions()

    print("Building features...")
    feature_df = build_features(raw_df)

    if "label" not in feature_df.columns:
        raise ValueError("No 'label' column found. Add labels before training.")

    x, y = split_xy(feature_df)
    x_train, x_test, y_train, y_test = make_train_test(x, y)

    print("Training model...")
    model = build_model()
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, preds, digits=4))

    try:
        auc = roc_auc_score(y_test, probs)
        print(f"ROC AUC: {auc:.4f}")
    except ValueError:
        print("ROC AUC unavailable for this dataset.")

    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
    joblib.dump(model, config.model_path)
    print(f"Saved model to {config.model_path}")


if __name__ == "__main__":
    main()