import pandas as pd
from sqlalchemy import text
from src.db import engine


def load_transactions(limit: int | None = None) -> pd.DataFrame:
    sql = """
    SELECT
        hash,
        account,
        destination,
        ledger_index,
        close_time,
        amount_value,
        currency,
        success,
        memo_text,
        balance_xrp,
        label
    FROM transactions
    ORDER BY ledger_index ASC, close_time ASC
    """

    if limit:
        sql += f" LIMIT {int(limit)}"

    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)

    return df