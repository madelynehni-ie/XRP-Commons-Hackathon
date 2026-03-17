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
    high_risk_threshold: float = float(os.getenv("HIGH_RISK_THRESHOLD", "0.8"))
    whale_percentile_threshold: float = float(os.getenv("WHALE_PERCENTILE_THRESHOLD", "0.95"))