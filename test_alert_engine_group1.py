"""
Tests for alert_engine.py — Group 1: Size & Whale Movements

Run with:
    python3 test_alert_engine_group1.py
"""

import sys
from datetime import datetime, timezone
from whale_registry import WhaleRegistry
from transaction_buffer import TransactionBuffer
from alert_engine import AlertEngine, Alert, LARGE_TX_PERCENTILE, VOLUME_SPIKE_RATIO, HIGH_RISK_THRESHOLD

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"  {PASS} {name}")
        passed += 1
    else:
        print(f"  {FAIL} {name}" + (f"  →  {detail}" if detail else ""))
        failed += 1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

print("\nBuilding whale registry (this takes ~10 seconds)...")
registry = WhaleRegistry.build(verbose=False)

# Grab a known whale and a known non-whale from the registry
WHALE   = next(iter(registry.whale_accounts))
NON_WHALE = next(
    acc for acc in registry._stats
    if not registry.is_whale(acc)
)
print(f"  whale     : {WHALE}")
print(f"  non-whale : {NON_WHALE}\n")


def make_engine():
    """Fresh engine + buffer for each test group."""
    return AlertEngine(registry, TransactionBuffer())


def make_tx(account=WHALE, tx_type="Payment", amount_xrp=10000.0,
            currency="XRP", destination="rDEST123",
            risk_score=0.5, is_anomaly=0,
            tx_size_percentile=0.995, is_large_tx=1,
            volume_spike_ratio=1.0, tx_rate_z_score=1.0,
            rolling_tx_count_5m=5):
    return {
        "timestamp": datetime.now(timezone.utc),
        "account": account,
        "tx_type": tx_type,
        "amount_xrp": amount_xrp,
        "currency": currency,
        "destination": destination,
        "risk_score": risk_score,
        "is_anomaly": is_anomaly,
        "tx_size_percentile": tx_size_percentile,
        "is_large_tx": is_large_tx,
        "volume_spike_ratio": volume_spike_ratio,
        "tx_rate_z_score": tx_rate_z_score,
        "rolling_tx_count_5m": rolling_tx_count_5m,
        "total_volume_5m": 500000.0,
    }


# ---------------------------------------------------------------------------
print("── Test 1: LARGE_XRP_TRANSFER — core trigger ─────────────────────────")
# ---------------------------------------------------------------------------

engine = make_engine()
tx = make_tx(account=WHALE, tx_type="Payment", is_large_tx=1,
             tx_size_percentile=0.997, risk_score=0.3)
alerts = engine.process_transaction(tx)

check("fires exactly 1 alert", len(alerts) == 1, f"got {len(alerts)}")
check("alert_type is LARGE_XRP_TRANSFER",
      alerts[0].alert_type == "LARGE_XRP_TRANSFER")
check("account is the whale", alerts[0].account == WHALE)
check("message contains 🐋", "🐋" in alerts[0].message)
check("returns Alert object", isinstance(alerts[0], Alert))
check("details has amount", "amount" in alerts[0].details)
check("details has currency", "currency" in alerts[0].details)
check("details has tx_size_percentile", "tx_size_percentile" in alerts[0].details)


# ---------------------------------------------------------------------------
print("\n── Test 2: LARGE_XRP_TRANSFER — gates ───────────────────────────────")
# ---------------------------------------------------------------------------

engine2 = make_engine()

# Non-whale should NOT fire
alerts_nw = engine2.process_transaction(
    make_tx(account=NON_WHALE, is_large_tx=1, tx_size_percentile=0.999)
)
check("does NOT fire for non-whale",
      not any(a.alert_type == "LARGE_XRP_TRANSFER" for a in alerts_nw))

# Small transaction should NOT fire
alerts_small = engine2.process_transaction(
    make_tx(account=WHALE, is_large_tx=0, tx_size_percentile=0.5)
)
check("does NOT fire for small tx",
      not any(a.alert_type == "LARGE_XRP_TRANSFER" for a in alerts_small))

# Wrong tx type should NOT fire
alerts_wrong_type = engine2.process_transaction(
    make_tx(account=WHALE, tx_type="AccountSet", is_large_tx=1)
)
check("does NOT fire for AccountSet tx_type",
      not any(a.alert_type == "LARGE_XRP_TRANSFER" for a in alerts_wrong_type))

# OfferCreate SHOULD fire (valid type)
alerts_offer = engine2.process_transaction(
    make_tx(account=WHALE, tx_type="OfferCreate", is_large_tx=1,
            tx_size_percentile=0.999)
)
check("fires for OfferCreate tx_type",
      any(a.alert_type == "LARGE_XRP_TRANSFER" for a in alerts_offer))


# ---------------------------------------------------------------------------
print("\n── Test 3: LARGE_XRP_TRANSFER — severity & cooldown ─────────────────")
# ---------------------------------------------------------------------------

engine3 = make_engine()

# Low ML score → medium or low severity
tx_low_risk = make_tx(account=WHALE, is_large_tx=1, risk_score=0.2, is_anomaly=0)
a_low = engine3.process_transaction(tx_low_risk)
check("low risk_score → severity is low or medium",
      a_low[0].severity in ("low", "medium"), f"got {a_low[0].severity}")

# High ML score (>= 0.6) → severity bumped
engine4 = make_engine()
tx_high_risk = make_tx(account=WHALE, is_large_tx=1, risk_score=0.75, is_anomaly=1)
a_high = engine4.process_transaction(tx_high_risk)
check("high risk_score bumps severity to high or critical",
      a_high[0].severity in ("high", "critical"), f"got {a_high[0].severity}")

# Cooldown — second call should NOT fire
engine5 = make_engine()
engine5.process_transaction(make_tx(account=WHALE, is_large_tx=1))
alerts_second = engine5.process_transaction(make_tx(account=WHALE, is_large_tx=1))
check("cooldown blocks second LARGE_XRP_TRANSFER within window",
      not any(a.alert_type == "LARGE_XRP_TRANSFER" for a in alerts_second))

# Different account not on cooldown
alerts_other = engine5.process_transaction(
    make_tx(account=list(registry.whale_accounts)[1], is_large_tx=1)
)
check("cooldown is per-account (other whale still fires)",
      any(a.alert_type == "LARGE_XRP_TRANSFER" for a in alerts_other))


# ---------------------------------------------------------------------------
print("\n── Test 4: VOLUME_SPIKE — core trigger ──────────────────────────────")
# ---------------------------------------------------------------------------

engine6 = make_engine()
tx_spike = make_tx(account=NON_WHALE, volume_spike_ratio=5.5,
                   is_large_tx=0, tx_size_percentile=0.3)
alerts_spike = engine6.process_transaction(tx_spike)

check("fires for volume_spike_ratio >= 3.0",
      any(a.alert_type == "VOLUME_SPIKE" for a in alerts_spike))
spike_alert = next(a for a in alerts_spike if a.alert_type == "VOLUME_SPIKE")
check("account is 'NETWORK'", spike_alert.account == "NETWORK")
check("message contains 📈", "📈" in spike_alert.message)
check("details has volume_spike_ratio", "volume_spike_ratio" in spike_alert.details)
check("details has trigger_account", "trigger_account" in spike_alert.details)
check("trigger_account matches sender", spike_alert.details["trigger_account"] == NON_WHALE)


# ---------------------------------------------------------------------------
print("\n── Test 5: VOLUME_SPIKE — gates & network cooldown ──────────────────")
# ---------------------------------------------------------------------------

engine7 = make_engine()

# Below threshold should NOT fire
alerts_no_spike = engine7.process_transaction(
    make_tx(volume_spike_ratio=2.9, is_large_tx=0, tx_size_percentile=0.1)
)
check("does NOT fire for volume_spike_ratio < 3.0",
      not any(a.alert_type == "VOLUME_SPIKE" for a in alerts_no_spike))

# Fire once, then cooldown applies for ALL accounts
engine8 = make_engine()
engine8.process_transaction(make_tx(account=WHALE, volume_spike_ratio=4.0,
                                    is_large_tx=0, tx_size_percentile=0.1))
alerts_second_spike = engine8.process_transaction(
    make_tx(account=NON_WHALE, volume_spike_ratio=6.0,
            is_large_tx=0, tx_size_percentile=0.1)
)
check("network cooldown blocks VOLUME_SPIKE from different account",
      not any(a.alert_type == "VOLUME_SPIKE" for a in alerts_second_spike))


# ---------------------------------------------------------------------------
print("\n── Test 6: HIGH_RISK_TRANSACTION — core trigger ─────────────────────")
# ---------------------------------------------------------------------------

engine9 = make_engine()
tx_hr = make_tx(account=NON_WHALE, risk_score=0.85, is_anomaly=1,
                is_large_tx=0, tx_size_percentile=0.2, volume_spike_ratio=1.0)
alerts_hr = engine9.process_transaction(tx_hr)

check("fires for risk_score >= 0.8 and is_anomaly=1",
      any(a.alert_type == "HIGH_RISK_TRANSACTION" for a in alerts_hr))
hr = next(a for a in alerts_hr if a.alert_type == "HIGH_RISK_TRANSACTION")
check("severity is high",           hr.severity == "high", f"got {hr.severity}")
check("message contains ⚠️",       "⚠️" in hr.message)
check("fires for NON-whale too",    hr.account == NON_WHALE)
check("details has tx_type",        "tx_type" in hr.details)
check("details has tx_rate_z_score","tx_rate_z_score" in hr.details)

# Critical threshold
engine10 = make_engine()
tx_crit = make_tx(account=NON_WHALE, risk_score=0.95, is_anomaly=1,
                  is_large_tx=0, tx_size_percentile=0.2, volume_spike_ratio=1.0)
alerts_crit = engine10.process_transaction(tx_crit)
crit = next(a for a in alerts_crit if a.alert_type == "HIGH_RISK_TRANSACTION")
check("severity is critical for risk_score >= 0.9",
      crit.severity == "critical", f"got {crit.severity}")


# ---------------------------------------------------------------------------
print("\n── Test 7: HIGH_RISK_TRANSACTION — gates ────────────────────────────")
# ---------------------------------------------------------------------------

engine11 = make_engine()

# Below risk threshold
alerts_low = engine11.process_transaction(
    make_tx(risk_score=0.79, is_anomaly=1, is_large_tx=0,
            tx_size_percentile=0.2, volume_spike_ratio=1.0)
)
check("does NOT fire for risk_score < 0.8",
      not any(a.alert_type == "HIGH_RISK_TRANSACTION" for a in alerts_low))

# High risk score but is_anomaly=0
alerts_no_flag = engine11.process_transaction(
    make_tx(risk_score=0.85, is_anomaly=0, is_large_tx=0,
            tx_size_percentile=0.2, volume_spike_ratio=1.0)
)
check("does NOT fire when is_anomaly=0 even with high risk_score",
      not any(a.alert_type == "HIGH_RISK_TRANSACTION" for a in alerts_no_flag))

# Cooldown
engine12 = make_engine()
engine12.process_transaction(make_tx(risk_score=0.85, is_anomaly=1, is_large_tx=0,
                                     tx_size_percentile=0.2, volume_spike_ratio=1.0))
alerts_cd = engine12.process_transaction(make_tx(risk_score=0.9, is_anomaly=1,
                                                  is_large_tx=0, tx_size_percentile=0.2,
                                                  volume_spike_ratio=1.0))
check("cooldown blocks second HIGH_RISK_TRANSACTION",
      not any(a.alert_type == "HIGH_RISK_TRANSACTION" for a in alerts_cd))


# ---------------------------------------------------------------------------
print("\n── Test 8: Alert object shape ───────────────────────────────────────")
# ---------------------------------------------------------------------------

engine13 = make_engine()
alerts_shape = engine13.process_transaction(
    make_tx(account=WHALE, is_large_tx=1, risk_score=0.85, is_anomaly=1,
            tx_size_percentile=0.999, volume_spike_ratio=4.0)
)

for alert in alerts_shape:
    check(f"{alert.alert_type} — has id",         bool(alert.id))
    check(f"{alert.alert_type} — has timestamp",  bool(alert.timestamp))
    check(f"{alert.alert_type} — has message",    bool(alert.message))
    check(f"{alert.alert_type} — severity valid",
          alert.severity in ("low", "medium", "high", "critical"),
          f"got {alert.severity}")
    check(f"{alert.alert_type} — risk_score in [0,1]",
          0.0 <= alert.risk_score <= 1.0, f"got {alert.risk_score}")
    check(f"{alert.alert_type} — to_dict works",
          isinstance(alert.to_dict(), dict))


# ---------------------------------------------------------------------------
print("\n── Summary ───────────────────────────────────────────────────────────")
# ---------------------------------------------------------------------------

total = passed + failed
print(f"\n  {passed}/{total} tests passed", end="")
if failed == 0:
    print("  \033[92m— all good!\033[0m")
else:
    print(f"  \033[91m— {failed} failed\033[0m")
print()
