
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from pyod.models.iforest import IForest


def calibrate_contamination(X: np.ndarray) -> float:
    """
    Estimate the fraction of anomalies from the data itself using a
    Tukey fence on the first principal component.

    This means we never hardcode a threshold — the model looks at the
    data's own distribution and decides how many points look like outliers.

    Returns a float in [0.01, 0.15].
    """
    if X.shape[0] < 20:
        return 0.05

    pca = PCA(n_components=1, random_state=42)
    pc1 = pca.fit_transform(X).flatten()

    q1, q3 = np.percentile(pc1, [25, 75])
    iqr = q3 - q1
    n_outliers = int(np.sum((pc1 < q1 - 3 * iqr) | (pc1 > q3 + 3 * iqr)))
    contamination = float(np.clip(n_outliers / len(pc1), 0.01, 0.15))
    return contamination


class IsolationForestPipeline:
    """
    Wraps RobustScaler + IForest in a sklearn-compatible interface.

    Why RobustScaler instead of StandardScaler:
    - Uses median and IQR instead of mean and std
    - A single spike doesn't distort the scaling for everything else
    - Important here because transaction amounts are heavily right-skewed

    Why IForest instead of RandomForestClassifier:
    - We have no labels — this is unsupervised
    - IForest learns what "normal" looks like from the data itself
    - Anomaly score = how quickly a point gets isolated by random cuts
    - No hardcoded thresholds needed
    """

    def __init__(self, contamination: float | None = None, random_state: int = 42):
        self.contamination = contamination  # None = auto-calibrate at fit time
        self.random_state = random_state
        self.scaler_ = RobustScaler()
        self.model_  = None
        self.contamination_used_ = None

    def fit(self, X: np.ndarray) -> "IsolationForestPipeline":
        X_scaled = self.scaler_.fit_transform(X)

        # Self-calibrate contamination if not provided
        cont = self.contamination if self.contamination is not None \
               else calibrate_contamination(X_scaled)
        self.contamination_used_ = cont

        self.model_ = IForest(
            contamination=cont,
            n_estimators=200,
            max_samples="auto",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(X_scaled)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Returns anomaly scores in [0, 1].
        Higher = more anomalous.
        """
        X_scaled = self.scaler_.transform(X)
        raw = self.model_.decision_function(X_scaled)
        # PyOD decision_function: higher = more anomalous (already normalised)
        return np.clip(raw, 0.0, 1.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns 1 (anomaly) or 0 (normal).
        Uses the model's own fitted boundary — no external threshold.
        """
        X_scaled = self.scaler_.transform(X)
        preds = self.model_.predict(X_scaled)
        # PyOD returns 1=anomaly, 0=normal (opposite of sklearn's -1/1)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns array of shape (n, 2): [P(normal), P(anomaly)].
        Keeps compatibility with score_batch.py which calls predict_proba(x)[:, 1].
        """
        scores = self.decision_function(X)
        return np.column_stack([1 - scores, scores])

    # sklearn Pipeline compatibility
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).predict(X)


def build_model(contamination: float | None = None, random_state: int = 42) -> IsolationForestPipeline:
    """
    Drop-in replacement for the old build_model().
    Returns an IsolationForestPipeline with the same .fit() / .predict() / .predict_proba() interface.
    """
    return IsolationForestPipeline(contamination=contamination, random_state=random_state)
