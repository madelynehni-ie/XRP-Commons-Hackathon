from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest


def build_model() -> Pipeline:
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "isolation_forest",
                IsolationForest(
                    n_estimators=200,
                    contamination=0.05,  # ~5% anomalies (tune this)
                    random_state=42,
                ),
            ),
        ]
    )
    return model