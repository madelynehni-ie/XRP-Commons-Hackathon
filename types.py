from dataclasses import dataclass
from typing import Optional


@dataclass
class FeatureRow:
    account: str
    ledger_index: int
    tx_count_last_5_ledgers: float
    tx_count_last_20_ledgers: float
    destination_concentration: float
    reserve_utilization_delta: float
    failed_tx_ratio: float
    memo_entropy: float
    label: Optional[int] = None