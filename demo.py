"""Live demo: colour-coded terminal output of scored XRPL transactions.

This is a thin wrapper around score_realtime.py that replaces the plain
print output with a formatted, colour-coded table. Delete this file once
the web dashboard is ready.

Usage:
    python3 demo.py
"""

import asyncio
import os

from score_realtime import (
    _compute_features,
    _decode_memos_from_ws,
    _init_csv,
    _append_scored_row,
    _load_historical_stats,
    OUTPUT_COLUMNS,
    OUT_PATH,
)
from model import load_model, FEATURE_COLUMNS
from normalize import normalize_transaction

import pandas as pd
from xrpl.asyncio.clients import AsyncWebsocketClient
from xrpl.models.requests import Subscribe


# ---------------------------------------------------------------------------
# ANSI colour codes for terminal output
# ---------------------------------------------------------------------------

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def _risk_colour(score: float) -> str:
    """Return ANSI colour based on risk score."""
    if score >= 0.7:
        return RED
    elif score >= 0.4:
        return YELLOW
    return GREEN


def _risk_bar(score: float, width: int = 20) -> str:
    """Return a visual bar like [████████░░░░░░░░░░░░]"""
    filled = int(score * width)
    empty = width - filled
    colour = _risk_colour(score)
    return f"{colour}{'█' * filled}{'░' * empty}{RESET}"


def _format_xrp(amount) -> str:
    """Format XRP amount for display."""
    if amount is None or amount == 0 or amount == "":
        return "          -"
    try:
        val = float(amount)
        if val >= 1_000_000:
            return f"{val/1_000_000:>10.2f}M"
        elif val >= 1_000:
            return f"{val/1_000:>10.2f}K"
        else:
            return f"{val:>11.4f}"
    except (ValueError, TypeError):
        return "          -"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def stream_demo(model, stats):
    """Stream, score, and display transactions with colour-coded risk."""

    # Print header
    print()
    print(f"{BOLD}{'':>6} {'RISK':>6} {'':>22} {'TYPE':<20} {'AMOUNT XRP':>11} {'CURR':>5}  ACCOUNT{RESET}")
    print(f"{DIM}{'─' * 95}{RESET}")

    async with AsyncWebsocketClient("wss://xrplcluster.com/") as client:
        await client.send(Subscribe(streams=["transactions"]))

        count = 0
        anomaly_count = 0

        async for msg in client:
            tx = msg.get("tx_json") or msg.get("transaction")
            if tx is None:
                continue

            # Normalise
            raw = {
                **tx,
                "ledger_index": msg.get("ledger_index", tx.get("ledger_index")),
                "ledger_close_time_human": msg.get("close_time_iso") or msg.get("date"),
                "hash": msg.get("hash") or tx.get("hash"),
            }
            normalised = normalize_transaction(raw)

            # Decode memos and compute features
            memo_text = _decode_memos_from_ws(tx)
            features = _compute_features(normalised, memo_text, stats)

            # Score
            feature_row = pd.DataFrame([features])
            X = feature_row[FEATURE_COLUMNS].fillna(0).values
            raw_score = model.decision_function(X)[0]
            prediction = model.predict(X)[0]
            risk_score = max(0.0, min(1.0, 0.5 - raw_score))
            is_anomaly = 1 if prediction == -1 else 0

            # Save to CSV
            scored_row = {**normalised, **features, "risk_score": round(risk_score, 4), "is_anomaly": is_anomaly}
            _append_scored_row(scored_row)

            count += 1
            if is_anomaly:
                anomaly_count += 1

            # Format output
            colour = _risk_colour(risk_score)
            bar = _risk_bar(risk_score)
            label = f"{RED}{BOLD}⚠ RISK{RESET}" if is_anomaly else f"{GREEN}  OK  {RESET}"
            tx_type = (normalised.get("tx_type") or "Unknown")[:20]
            amount_str = _format_xrp(normalised.get("amount_xrp"))
            currency = (normalised.get("currency") or "XRP")[:5]
            account = str(normalised.get("account") or "")[:16]

            print(
                f"{label} "
                f"{colour}{risk_score:.3f}{RESET} "
                f"{bar} "
                f"{tx_type:<20s} "
                f"{amount_str} "
                f"{currency:>5}  "
                f"{DIM}{account}...{RESET}"
            )

            # Every 50 transactions, print a summary line
            if count % 50 == 0:
                pct = (anomaly_count / count) * 100
                print(f"{DIM}{'─' * 95}{RESET}")
                print(f"{DIM}  [{count} transactions scored | {anomaly_count} anomalies ({pct:.1f}%)]{RESET}")
                print(f"{DIM}{'─' * 95}{RESET}")


if __name__ == "__main__":
    model = load_model()
    stats = _load_historical_stats()
    _init_csv()

    print(f"\n{BOLD}🔍 XRPL Risk Monitor — Live Demo{RESET}")
    print(f"{DIM}Streaming from wss://xrplcluster.com/{RESET}")
    print(f"{DIM}Scored transactions → {OUT_PATH}{RESET}")
    print(f"{DIM}Press Ctrl+C to stop.{RESET}")

    try:
        asyncio.run(stream_demo(model, stats))
    except KeyboardInterrupt:
        print(f"\n{BOLD}Demo stopped.{RESET} Data saved in {OUT_PATH}")
