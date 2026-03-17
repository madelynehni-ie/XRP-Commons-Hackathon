"""
Tests for alert_engine.py — Group 3: DEX & Trading Activity

Run with:
    python3 test_alert_engine_group3.py
"""

from datetime import datetime, timezone
from whale_registry import WhaleRegistry
from transaction_buffer import TransactionBuffer
from alert_engine import AlertEngine, TOKEN_OFFER_MIN, CANCEL_RATIO_THRESHOLD, MULTI_WHALE_MIN

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
WHALES    = list(registry.whale_accounts)
WHALE     = WHALES[0]
NON_WHALE = next(acc for acc in registry._stats if not registry.is_whale(acc))
print(f"  whale: {WHALE}  |  non-whale: {NON_WHALE}\n")


def make_engine():
    return AlertEngine(registry, TransactionBuffer())


def make_tx(account=None, tx_type="OfferCreate", currency="SOLO",
            amount_xrp=500.0, risk_score=0.4, is_anomaly=0,
            rolling_tx_count_5m=5, tx_rate_z_score=1.0,
            is_large_tx=0, tx_size_percentile=0.3, volume_spike_ratio=1.0):
    return {
        "timestamp": datetime.now(timezone.utc),
        "account": account or WHALE,
        "tx_type": tx_type,
        "amount_xrp": amount_xrp,
        "currency": currency,
        "destination": "",
        "risk_score": risk_score,
        "is_anomaly": is_anomaly,
        "rolling_tx_count_5m": rolling_tx_count_5m,
        "tx_rate_z_score": tx_rate_z_score,
        "tx_per_minute": 5.0,
        "is_large_tx": is_large_tx,
        "tx_size_percentile": tx_size_percentile,
        "volume_spike_ratio": volume_spike_ratio,
        "total_volume_5m": 50000.0,
    }


# ---------------------------------------------------------------------------
print("── Test 1: TOKEN_ACCUMULATION — core trigger ─────────────────────────")
# ---------------------------------------------------------------------------

engine = make_engine()
# Send TOKEN_OFFER_MIN OfferCreates for the same token from a whale
for _ in range(TOKEN_OFFER_MIN):
    engine.process_transaction(make_tx(account=WHALE, tx_type="OfferCreate", currency="SOLO"))

# The TOKEN_OFFER_MIN-th call should fire
all_alerts = []
for _ in range(TOKEN_OFFER_MIN):
    all_alerts += engine.process_transaction(
        make_tx(account=WHALES[1], tx_type="OfferCreate", currency="RLUSD")
    )

# Fresh engine to test cleanly
engine2 = make_engine()
fired = []
for i in range(TOKEN_OFFER_MIN + 1):
    alerts = engine2.process_transaction(
        make_tx(account=WHALE, tx_type="OfferCreate", currency="SOLO")
    )
    fired += [a for a in alerts if a.alert_type == "TOKEN_ACCUMULATION"]

check("fires after TOKEN_OFFER_MIN offers on same token",
      len(fired) >= 1, f"fired {len(fired)} times")
check("alert_type is TOKEN_ACCUMULATION", fired[0].alert_type == "TOKEN_ACCUMULATION")
check("message contains 🐋", "🐋" in fired[0].message)
check("details has token", "token" in fired[0].details)
check("details token = SOLO", fired[0].details["token"] == "SOLO")
check("details has offer_count", "offer_count" in fired[0].details)
check("offer_count >= threshold", fired[0].details["offer_count"] >= TOKEN_OFFER_MIN)


# ---------------------------------------------------------------------------
print("\n── Test 2: TOKEN_ACCUMULATION — gates ───────────────────────────────")
# ---------------------------------------------------------------------------

engine3 = make_engine()

# Non-whale should NOT fire
for _ in range(TOKEN_OFFER_MIN + 2):
    alerts_nw = engine3.process_transaction(
        make_tx(account=NON_WHALE, tx_type="OfferCreate", currency="SOLO")
    )
check("does NOT fire for non-whale",
      not any(a.alert_type == "TOKEN_ACCUMULATION" for a in alerts_nw))

# XRP currency should NOT fire
engine4 = make_engine()
xrp_fired = []
for _ in range(TOKEN_OFFER_MIN + 2):
    xrp_fired += [a for a in engine4.process_transaction(
        make_tx(account=WHALE, tx_type="OfferCreate", currency="XRP")
    ) if a.alert_type == "TOKEN_ACCUMULATION"]
check("does NOT fire for XRP currency (XRP is not a token)",
      len(xrp_fired) == 0)

# Below threshold should NOT fire
engine5 = make_engine()
below_fired = []
for _ in range(TOKEN_OFFER_MIN - 1):
    below_fired += [a for a in engine5.process_transaction(
        make_tx(account=WHALE, tx_type="OfferCreate", currency="ARK")
    ) if a.alert_type == "TOKEN_ACCUMULATION"]
check("does NOT fire before reaching threshold",
      len(below_fired) == 0)

# Non-OfferCreate should NOT fire
engine6 = make_engine()
pay_fired = []
for _ in range(TOKEN_OFFER_MIN + 2):
    pay_fired += [a for a in engine6.process_transaction(
        make_tx(account=WHALE, tx_type="Payment", currency="SOLO")
    ) if a.alert_type == "TOKEN_ACCUMULATION"]
check("does NOT fire for Payment tx_type",
      len(pay_fired) == 0)


# ---------------------------------------------------------------------------
print("\n── Test 3: OFFER_CANCEL_SPIKE — core trigger ─────────────────────────")
# ---------------------------------------------------------------------------

engine7 = make_engine()
# Build up offers then cancel most of them
for _ in range(3):
    engine7.process_transaction(make_tx(account=WHALE, tx_type="OfferCreate", currency="XRP"))
for _ in range(5):
    engine7.process_transaction(make_tx(account=WHALE, tx_type="OfferCancel", currency="XRP"))

# Find the cancel spike alert
cancel_alerts = []
for _ in range(3):
    alerts = engine7.process_transaction(
        make_tx(account=WHALES[2], tx_type="OfferCreate", currency="XRP")
    )

# Fresh targeted test
engine8 = make_engine()
# 2 creates, 8 cancels = 80% cancel ratio
for _ in range(2):
    engine8.process_transaction(make_tx(account=WHALE, tx_type="OfferCreate", currency="XRP"))
spike_alerts = []
for _ in range(8):
    spike_alerts += [a for a in engine8.process_transaction(
        make_tx(account=WHALE, tx_type="OfferCancel", currency="XRP")
    ) if a.alert_type == "OFFER_CANCEL_SPIKE"]

check("fires when cancel ratio >= threshold",
      len(spike_alerts) >= 1, f"fired {len(spike_alerts)} times")
check("alert_type is OFFER_CANCEL_SPIKE", spike_alerts[0].alert_type == "OFFER_CANCEL_SPIKE")
check("message contains 👻", "👻" in spike_alerts[0].message)
check("details has cancel_ratio", "cancel_ratio" in spike_alerts[0].details)
check("cancel_ratio >= threshold",
      spike_alerts[0].details["cancel_ratio"] >= CANCEL_RATIO_THRESHOLD,
      f"got {spike_alerts[0].details['cancel_ratio']}")
check("details has offer_creates", "offer_creates" in spike_alerts[0].details)
check("details has offer_cancels", "offer_cancels" in spike_alerts[0].details)


# ---------------------------------------------------------------------------
print("\n── Test 4: OFFER_CANCEL_SPIKE — gates ───────────────────────────────")
# ---------------------------------------------------------------------------

# Non-whale should NOT fire
engine9 = make_engine()
nw_cancel = []
for _ in range(2):
    engine9.process_transaction(make_tx(account=NON_WHALE, tx_type="OfferCreate"))
for _ in range(8):
    nw_cancel += [a for a in engine9.process_transaction(
        make_tx(account=NON_WHALE, tx_type="OfferCancel")
    ) if a.alert_type == "OFFER_CANCEL_SPIKE"]
check("does NOT fire for non-whale", len(nw_cancel) == 0)

# Too few total offers should NOT fire (below CANCEL_MIN_OFFERS)
engine10 = make_engine()
few_offers = []
engine10.process_transaction(make_tx(account=WHALES[3], tx_type="OfferCreate"))
for _ in range(3):
    few_offers += [a for a in engine10.process_transaction(
        make_tx(account=WHALES[3], tx_type="OfferCancel")
    ) if a.alert_type == "OFFER_CANCEL_SPIKE"]
check("does NOT fire when total offers below minimum",
      len(few_offers) == 0, f"fired {len(few_offers)} times")

# Low cancel ratio should NOT fire
engine11 = make_engine()
low_ratio = []
for _ in range(8):
    engine11.process_transaction(make_tx(account=WHALES[4], tx_type="OfferCreate"))
for _ in range(2):
    low_ratio += [a for a in engine11.process_transaction(
        make_tx(account=WHALES[4], tx_type="OfferCancel")
    ) if a.alert_type == "OFFER_CANCEL_SPIKE"]
check("does NOT fire when cancel ratio < threshold",
      len(low_ratio) == 0)


# ---------------------------------------------------------------------------
print("\n── Test 5: MULTI_WHALE_CONVERGENCE — core trigger ────────────────────")
# ---------------------------------------------------------------------------

engine12 = make_engine()
multi_alerts = []

# MULTI_WHALE_MIN different whales all offer the same token
for i in range(MULTI_WHALE_MIN):
    alerts = engine12.process_transaction(
        make_tx(account=WHALES[i], tx_type="OfferCreate", currency="SOLO")
    )
    multi_alerts += [a for a in alerts if a.alert_type == "MULTI_WHALE_CONVERGENCE"]

check(f"fires after {MULTI_WHALE_MIN} whales trade same token",
      len(multi_alerts) >= 1, f"fired {len(multi_alerts)} times")
check("alert_type is MULTI_WHALE_CONVERGENCE",
      multi_alerts[0].alert_type == "MULTI_WHALE_CONVERGENCE")
check("account is 'NETWORK'", multi_alerts[0].account == "NETWORK")
check("message contains 🔥", "🔥" in multi_alerts[0].message)
check("details has token", "token" in multi_alerts[0].details)
check("details token = SOLO", multi_alerts[0].details["token"] == "SOLO")
check("details has whale_count", "whale_count" in multi_alerts[0].details)
check("whale_count >= MULTI_WHALE_MIN",
      multi_alerts[0].details["whale_count"] >= MULTI_WHALE_MIN)
check("details has whale_accounts list",
      isinstance(multi_alerts[0].details.get("whale_accounts"), list))


# ---------------------------------------------------------------------------
print("\n── Test 6: MULTI_WHALE_CONVERGENCE — gates & token cooldown ──────────")
# ---------------------------------------------------------------------------

# Fewer than MULTI_WHALE_MIN whales should NOT fire
engine13 = make_engine()
too_few = []
for i in range(MULTI_WHALE_MIN - 1):
    too_few += [a for a in engine13.process_transaction(
        make_tx(account=WHALES[i], tx_type="OfferCreate", currency="ARK")
    ) if a.alert_type == "MULTI_WHALE_CONVERGENCE"]
check("does NOT fire with fewer than MULTI_WHALE_MIN whales",
      len(too_few) == 0)

# Non-XRP only — XRP offers should NOT fire
engine14 = make_engine()
xrp_multi = []
for i in range(MULTI_WHALE_MIN + 2):
    xrp_multi += [a for a in engine14.process_transaction(
        make_tx(account=WHALES[i], tx_type="OfferCreate", currency="XRP")
    ) if a.alert_type == "MULTI_WHALE_CONVERGENCE"]
check("does NOT fire for XRP currency", len(xrp_multi) == 0)

# Cooldown is per-token: different token still fires
engine15 = make_engine()
for i in range(MULTI_WHALE_MIN):
    engine15.process_transaction(
        make_tx(account=WHALES[i], tx_type="OfferCreate", currency="SOLO")
    )
# Now fire on a different token — should NOT be blocked
rlusd_alerts = []
for i in range(MULTI_WHALE_MIN):
    rlusd_alerts += [a for a in engine15.process_transaction(
        make_tx(account=WHALES[i], tx_type="OfferCreate", currency="RLUSD")
    ) if a.alert_type == "MULTI_WHALE_CONVERGENCE"]
check("cooldown is per-token (different token still fires)",
      len(rlusd_alerts) >= 1)

# Same token is on cooldown
solo_again = []
for i in range(MULTI_WHALE_MIN):
    solo_again += [a for a in engine15.process_transaction(
        make_tx(account=WHALES[i], tx_type="OfferCreate", currency="SOLO")
    ) if a.alert_type == "MULTI_WHALE_CONVERGENCE"]
check("same token is blocked by cooldown", len(solo_again) == 0)


# ---------------------------------------------------------------------------
print("\n── Summary ───────────────────────────────────────────────────────────")
# ---------------------------------------------------------------------------

total = passed + failed
print(f"\n  {passed}/{total} tests passed", end="")
print("  \033[92m— all good!\033[0m" if failed == 0 else f"  \033[91m— {failed} failed\033[0m")
print()
