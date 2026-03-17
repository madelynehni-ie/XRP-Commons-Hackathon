"""
Tests for alert_engine.py — Group 2: Burst & Bot Activity

Run with:
    python3 test_alert_engine_group2.py
"""

from datetime import datetime, timezone
from whale_registry import WhaleRegistry
from transaction_buffer import TransactionBuffer
from alert_engine import AlertEngine, BURST_TX_COUNT, BURST_Z_SCORE

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  {PASS} {name}")
        passed += 1
    else:
        print(f"  {FAIL} {name}" + (f"  →  {detail}" if detail else ""))
        failed += 1


print("\nBuilding whale registry...")
registry = WhaleRegistry.build(verbose=False)
WHALE     = next(iter(registry.whale_accounts))
NON_WHALE = next(acc for acc in registry._stats if not registry.is_whale(acc))
print(f"  whale: {WHALE}  |  non-whale: {NON_WHALE}\n")


def make_engine():
    return AlertEngine(registry, TransactionBuffer())


def make_tx(account=None, tx_type="Payment", risk_score=0.5, is_anomaly=0,
            rolling_tx_count_5m=10, tx_rate_z_score=1.0, tx_per_minute=5.0,
            is_large_tx=0, tx_size_percentile=0.3, volume_spike_ratio=1.0):
    return {
        "timestamp": datetime.now(timezone.utc),
        "account": account or WHALE,
        "tx_type": tx_type,
        "amount_xrp": 100.0,
        "currency": "XRP",
        "destination": "rDEST",
        "risk_score": risk_score,
        "is_anomaly": is_anomaly,
        "rolling_tx_count_5m": rolling_tx_count_5m,
        "tx_rate_z_score": tx_rate_z_score,
        "tx_per_minute": tx_per_minute,
        "is_large_tx": is_large_tx,
        "tx_size_percentile": tx_size_percentile,
        "volume_spike_ratio": volume_spike_ratio,
        "total_volume_5m": 100000.0,
    }


# ---------------------------------------------------------------------------
print("── Test 1: TRANSACTION_BURST — core trigger ──────────────────────────")
# ---------------------------------------------------------------------------

engine = make_engine()
alerts = engine.process_transaction(
    make_tx(account=WHALE, rolling_tx_count_5m=BURST_TX_COUNT + 10, risk_score=0.4)
)
burst_alerts = [a for a in alerts if a.alert_type == "TRANSACTION_BURST"]

check("fires for whale + rolling_tx_count_5m >= threshold",
      len(burst_alerts) == 1, f"got {len(burst_alerts)}")
check("alert_type is TRANSACTION_BURST", burst_alerts[0].alert_type == "TRANSACTION_BURST")
check("message contains ⚡", "⚡" in burst_alerts[0].message)
check("details has rolling_tx_count_5m", "rolling_tx_count_5m" in burst_alerts[0].details)
check("details has burst_threshold",     "burst_threshold" in burst_alerts[0].details)
check("details has tx_per_minute",       "tx_per_minute" in burst_alerts[0].details)
check("burst_threshold in details equals constant",
      burst_alerts[0].details["burst_threshold"] == BURST_TX_COUNT)


# ---------------------------------------------------------------------------
print("\n── Test 2: TRANSACTION_BURST — gates ────────────────────────────────")
# ---------------------------------------------------------------------------

engine2 = make_engine()

# Non-whale should NOT fire
alerts_nw = engine2.process_transaction(
    make_tx(account=NON_WHALE, rolling_tx_count_5m=BURST_TX_COUNT + 50)
)
check("does NOT fire for non-whale",
      not any(a.alert_type == "TRANSACTION_BURST" for a in alerts_nw))

# Below threshold should NOT fire
alerts_low = engine2.process_transaction(
    make_tx(account=WHALE, rolling_tx_count_5m=BURST_TX_COUNT - 1)
)
check("does NOT fire below threshold",
      not any(a.alert_type == "TRANSACTION_BURST" for a in alerts_low))

# Exactly at threshold SHOULD fire
alerts_exact = engine2.process_transaction(
    make_tx(account=list(registry.whale_accounts)[1],
            rolling_tx_count_5m=BURST_TX_COUNT)
)
check("fires at exactly the threshold",
      any(a.alert_type == "TRANSACTION_BURST" for a in alerts_exact))


# ---------------------------------------------------------------------------
print("\n── Test 3: TRANSACTION_BURST — cooldown ─────────────────────────────")
# ---------------------------------------------------------------------------

engine3 = make_engine()
engine3.process_transaction(make_tx(account=WHALE, rolling_tx_count_5m=BURST_TX_COUNT + 5))
alerts_cd = engine3.process_transaction(make_tx(account=WHALE, rolling_tx_count_5m=BURST_TX_COUNT + 5))
check("cooldown blocks second TRANSACTION_BURST",
      not any(a.alert_type == "TRANSACTION_BURST" for a in alerts_cd))

# Different whale not blocked
second_whale = list(registry.whale_accounts)[2]
alerts_other = engine3.process_transaction(
    make_tx(account=second_whale, rolling_tx_count_5m=BURST_TX_COUNT + 5)
)
check("cooldown is per-account (different whale still fires)",
      any(a.alert_type == "TRANSACTION_BURST" for a in alerts_other))


# ---------------------------------------------------------------------------
print("\n── Test 4: ABNORMAL_TX_RATE — core trigger ───────────────────────────")
# ---------------------------------------------------------------------------

engine4 = make_engine()
alerts_z = engine4.process_transaction(
    make_tx(account=WHALE, tx_rate_z_score=BURST_Z_SCORE + 1.0,
            tx_per_minute=120.0, risk_score=0.6, is_anomaly=1)
)
z_alerts = [a for a in alerts_z if a.alert_type == "ABNORMAL_TX_RATE"]

check("fires for whale + tx_rate_z_score >= threshold",
      len(z_alerts) == 1, f"got {len(z_alerts)}")
check("alert_type is ABNORMAL_TX_RATE", z_alerts[0].alert_type == "ABNORMAL_TX_RATE")
check("message contains 🤖",            "🤖" in z_alerts[0].message)
check("message contains tx/min rate",   "tx/min" in z_alerts[0].message)
check("details has tx_rate_z_score",    "tx_rate_z_score" in z_alerts[0].details)
check("details has tx_per_minute",      "tx_per_minute" in z_alerts[0].details)
check("details has z_score_threshold",  "z_score_threshold" in z_alerts[0].details)
check("z_score in details is correct",
      abs(z_alerts[0].details["tx_rate_z_score"] - (BURST_Z_SCORE + 1.0)) < 0.01)


# ---------------------------------------------------------------------------
print("\n── Test 5: ABNORMAL_TX_RATE — gates ─────────────────────────────────")
# ---------------------------------------------------------------------------

engine5 = make_engine()

# Non-whale should NOT fire
alerts_nw5 = engine5.process_transaction(
    make_tx(account=NON_WHALE, tx_rate_z_score=BURST_Z_SCORE + 5.0)
)
check("does NOT fire for non-whale",
      not any(a.alert_type == "ABNORMAL_TX_RATE" for a in alerts_nw5))

# Below z-score threshold should NOT fire
alerts_low5 = engine5.process_transaction(
    make_tx(account=WHALE, tx_rate_z_score=BURST_Z_SCORE - 0.1)
)
check("does NOT fire below z-score threshold",
      not any(a.alert_type == "ABNORMAL_TX_RATE" for a in alerts_low5))

# Exactly at threshold SHOULD fire (use different whale to avoid cooldown)
whale3 = list(registry.whale_accounts)[3]
alerts_exact5 = engine5.process_transaction(
    make_tx(account=whale3, tx_rate_z_score=BURST_Z_SCORE)
)
check("fires at exactly the z-score threshold",
      any(a.alert_type == "ABNORMAL_TX_RATE" for a in alerts_exact5))


# ---------------------------------------------------------------------------
print("\n── Test 6: ABNORMAL_TX_RATE — cooldown & severity ───────────────────")
# ---------------------------------------------------------------------------

engine6 = make_engine()
engine6.process_transaction(
    make_tx(account=WHALE, tx_rate_z_score=BURST_Z_SCORE + 2.0)
)
alerts_cd6 = engine6.process_transaction(
    make_tx(account=WHALE, tx_rate_z_score=BURST_Z_SCORE + 2.0)
)
check("cooldown blocks second ABNORMAL_TX_RATE",
      not any(a.alert_type == "ABNORMAL_TX_RATE" for a in alerts_cd6))

# High ML score → high severity
engine7 = make_engine()
alerts_sev = engine7.process_transaction(
    make_tx(account=WHALE, tx_rate_z_score=BURST_Z_SCORE + 1.0,
            risk_score=0.85, is_anomaly=1)
)
z_sev = next((a for a in alerts_sev if a.alert_type == "ABNORMAL_TX_RATE"), None)
check("high risk_score → severity is high or critical",
      z_sev is not None and z_sev.severity in ("high", "critical"),
      f"got {z_sev.severity if z_sev else 'no alert'}")


# ---------------------------------------------------------------------------
print("\n── Test 7: Both Group 2 alerts can fire on same transaction ──────────")
# ---------------------------------------------------------------------------

engine8 = make_engine()
whale4 = list(registry.whale_accounts)[4]
alerts_both = engine8.process_transaction(
    make_tx(account=whale4,
            rolling_tx_count_5m=BURST_TX_COUNT + 20,
            tx_rate_z_score=BURST_Z_SCORE + 2.0,
            tx_per_minute=150.0)
)
types = {a.alert_type for a in alerts_both}
check("TRANSACTION_BURST fires",  "TRANSACTION_BURST"  in types)
check("ABNORMAL_TX_RATE fires",   "ABNORMAL_TX_RATE"   in types)
check("both fire simultaneously", len([a for a in alerts_both
      if a.alert_type in ("TRANSACTION_BURST", "ABNORMAL_TX_RATE")]) == 2)


# ---------------------------------------------------------------------------
print("\n── Summary ───────────────────────────────────────────────────────────")
# ---------------------------------------------------------------------------

total = passed + failed
print(f"\n  {passed}/{total} tests passed", end="")
print("  \033[92m— all good!\033[0m" if failed == 0 else f"  \033[91m— {failed} failed\033[0m")
print()
