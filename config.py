from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/xrpl_risk_monitor",
    )
    model_path: str = os.getenv("ML_MODEL_PATH", "./models/risk_model.joblib")
    random_state: int = int(os.getenv("ML_RANDOM_STATE", "42"))

    # Score above this = HIGH risk flag in score_batch output
    # This is used only for labelling output, not for the model's own decision boundary
    high_risk_threshold: float = float(os.getenv("HIGH_RISK_THRESHOLD", "0.7"))

    # If set, override the self-calibrated contamination
    # Leave as None (default) to let the model calibrate itself
    contamination_override: float | None = (
        float(os.getenv("CONTAMINATION_OVERRIDE"))
        if os.getenv("CONTAMINATION_OVERRIDE") else None
    )
