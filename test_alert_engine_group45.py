"""
Tests for alert_engine.py — Groups 4 & 5: Memo/Spam + Behaviour Shifts

Run with:
    python3 test_alert_engine_group45.py
"""

from datetime import datetime, timezone
from whale_registry import WhaleRegistry
from transaction_buffer import TransactionBuffer
from alert_engine import (AlertEngine, MEMO_SPAM_MIN, MEMO_ENTROPY_HIGH,
                           URL_SPAM_TX_MIN, BEHAVIOUR_SHIFT_MIN_TXS,
                           BEHAVIOUR_SHIFT_DELTA)

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


def make_tx(account=None, tx_type="Payment", currency="XRP",
            risk_score=0.4, is_anomaly=0,
            duplicate_memo_count=0, memo_entropy=0.0, memo_length=0,
            contains_url=0, rolling_tx_count_5m=5,
            tx_rate_z_score=1.0, is_large_tx=0,
            tx_size_percentile=0.3, volume_spike_ratio=1.0,
            amount_xrp=100.0):
    return {
        "timestamp": datetime.now(timezone.utc),
        "account": account or WHALE,
        "tx_type": tx_type,
        "amount_xrp": amount_xrp,
        "currency": currency,
        "destination": "rDEST",
        "risk_score": risk_score,
        "is_anomaly": is_anomaly,
        "duplicate_memo_count": duplicate_memo_count,
        "memo_entropy": memo_entropy,
        "memo_length": memo_length,
        "contains_url": contains_url,
        "rolling_tx_count_5m": rolling_tx_count_5m,
        "tx_rate_z_score": tx_rate_z_score,
        "tx_per_minute": 5.0,
        "is_large_tx": is_large_tx,
        "tx_size_percentile": tx_size_percentile,
        "volume_spike_ratio": volume_spike_ratio,
        "total_volume_5m": 50000.0,
    }


# ---------------------------------------------------------------------------
print("── Test 1: MEMO_SPAM — core trigger ─────────────────────────────────")
# ---------------------------------------------------------------------------

engine = make_engine()
spam_alerts = []
for i in range(MEMO_SPAM_MIN + 2):
    alerts = engine.process_transaction(
        make_tx(account=NON_WHALE, duplicate_memo_count=MEMO_SPAM_MIN + i)
    )
    spam_alerts += [a for a in alerts if a.alert_type == "MEMO_SPAM"]

check("fires when duplicate_memo_count >= threshold",
      len(spam_alerts) >= 1, f"fired {len(spam_alerts)} times")
check("alert_type is MEMO_SPAM", spam_alerts[0].alert_type == "MEMO_SPAM")
check("message contains 📨", "📨" in spam_alerts[0].message)
check("fires for non-whale too", spam_alerts[0].account == NON_WHALE)
check("details has duplicate_memo_count", "duplicate_memo_count" in spam_alerts[0].details)
check("details has memo_spam_threshold", "memo_spam_threshold" in spam_alerts[0].details)


# ---------------------------------------------------------------------------
print("\n── Test 2: MEMO_SPAM — gates & cooldown ─────────────────────────────")
# ---------------------------------------------------------------------------

engine2 = make_engine()
below = [a for a in engine2.process_transaction(
    make_tx(duplicate_memo_count=MEMO_SPAM_MIN - 1)
) if a.alert_type == "MEMO_SPAM"]
check("does NOT fire below threshold", len(below) == 0)

# Cooldown
engine3 = make_engine()
engine3.process_transaction(make_tx(account=WHALE, duplicate_memo_count=MEMO_SPAM_MIN + 5))
cd = [a for a in engine3.process_transaction(
    make_tx(account=WHALE, duplicate_memo_count=MEMO_SPAM_MIN + 10)
) if a.alert_type == "MEMO_SPAM"]
check("cooldown blocks second MEMO_SPAM", len(cd) == 0)


# ---------------------------------------------------------------------------
print("\n── Test 3: URL_SPAM — core trigger ──────────────────────────────────")
# ---------------------------------------------------------------------------

engine4 = make_engine()
url_alerts = []
for _ in range(URL_SPAM_TX_MIN + 1):
    alerts = engine4.process_transaction(
        make_tx(account=NON_WHALE, contains_url=1, memo_length=50)
    )
    url_alerts += [a for a in alerts if a.alert_type == "URL_SPAM"]

check("fires after URL_SPAM_TX_MIN url-containing txs",
      len(url_alerts) >= 1, f"fired {len(url_alerts)} times")
check("alert_type is URL_SPAM", url_alerts[0].alert_type == "URL_SPAM")
check("message contains 🔗", "🔗" in url_alerts[0].message)
check("details has url_tx_count", "url_tx_count" in url_alerts[0].details)
check("url_tx_count >= threshold",
      url_alerts[0].details["url_tx_count"] >= URL_SPAM_TX_MIN)


# ---------------------------------------------------------------------------
print("\n── Test 4: URL_SPAM — gates ─────────────────────────────────────────")
# ---------------------------------------------------------------------------

engine5 = make_engine()
no_url = [a for a in engine5.process_transaction(
    make_tx(contains_url=0, memo_length=50)
) if a.alert_type == "URL_SPAM"]
check("does NOT fire when contains_url=0", len(no_url) == 0)

# Below count threshold should NOT fire
engine6 = make_engine()
few_url = []
for _ in range(URL_SPAM_TX_MIN - 1):
    few_url += [a for a in engine6.process_transaction(
        make_tx(account=WHALES[1], contains_url=1)
    ) if a.alert_type == "URL_SPAM"]
check("does NOT fire below url count threshold", len(few_url) == 0)


# ---------------------------------------------------------------------------
print("\n── Test 5: HIGH_ENTROPY_MEMO — core trigger ─────────────────────────")
# ---------------------------------------------------------------------------

engine7 = make_engine()
ent_alerts = [a for a in engine7.process_transaction(
    make_tx(memo_entropy=MEMO_ENTROPY_HIGH + 0.5, memo_length=80, risk_score=0.3)
) if a.alert_type == "HIGH_ENTROPY_MEMO"]

check("fires when memo_entropy >= threshold",
      len(ent_alerts) == 1, f"fired {len(ent_alerts)}")
check("alert_type is HIGH_ENTROPY_MEMO", ent_alerts[0].alert_type == "HIGH_ENTROPY_MEMO")
check("message contains 🔐", "🔐" in ent_alerts[0].message)
check("details has memo_entropy", "memo_entropy" in ent_alerts[0].details)
check("details has entropy_threshold", "entropy_threshold" in ent_alerts[0].details)


# ---------------------------------------------------------------------------
print("\n── Test 6: HIGH_ENTROPY_MEMO — gates ────────────────────────────────")
# ---------------------------------------------------------------------------

engine8 = make_engine()
low_ent = [a for a in engine8.process_transaction(
    make_tx(memo_entropy=MEMO_ENTROPY_HIGH - 0.1, memo_length=50)
) if a.alert_type == "HIGH_ENTROPY_MEMO"]
check("does NOT fire below entropy threshold", len(low_ent) == 0)

engine9 = make_engine()
no_memo = [a for a in engine9.process_transaction(
    make_tx(memo_entropy=MEMO_ENTROPY_HIGH + 1.0, memo_length=0)
) if a.alert_type == "HIGH_ENTROPY_MEMO"]
check("does NOT fire when memo_length=0 (no memo)", len(no_memo) == 0)


# ---------------------------------------------------------------------------
print("\n── Test 7: WALLET_DRAIN — core trigger ──────────────────────────────")
# ---------------------------------------------------------------------------

# Find a whale with known historical xrp_sent > 0
whale_with_xrp = next(
    (acc for acc in WHALES if registry.get_stats(acc).xrp_sent > 0),
    None
)

if whale_with_xrp:
    hist_sent = registry.get_stats(whale_with_xrp).xrp_sent
    drain_amount = hist_sent * 0.9  # 90% of historical = triggers drain

    engine10 = make_engine()
    drain_alerts = [a for a in engine10.process_transaction(
        make_tx(account=whale_with_xrp, tx_type="Payment",
                currency="XRP", amount_xrp=drain_amount)
    ) if a.alert_type == "WALLET_DRAIN"]

    check("fires when xrp_out >= 80% of historical sent",
          len(drain_alerts) >= 1, f"fired {len(drain_alerts)}, hist_sent={hist_sent:.2f}, drain={drain_amount:.2f}")
    if drain_alerts:
        check("message contains ⚠️", "⚠️" in drain_alerts[0].message)
        check("details has drain_ratio", "drain_ratio" in drain_alerts[0].details)
        check("drain_ratio >= 0.8", drain_alerts[0].details["drain_ratio"] >= 0.8,
              f"got {drain_alerts[0].details['drain_ratio']:.3f}")
else:
    print("  (skipped — no whale with XRP history in dataset)")

# Non-whale should NOT fire
engine11 = make_engine()
nw_drain = [a for a in engine11.process_transaction(
    make_tx(account=NON_WHALE, amount_xrp=99999.0, currency="XRP")
) if a.alert_type == "WALLET_DRAIN"]
check("does NOT fire for non-whale", len(nw_drain) == 0)


# ---------------------------------------------------------------------------
print("\n── Test 8: BEHAVIOUR_SHIFT — core trigger ────────────────────────────")
# ---------------------------------------------------------------------------

# Find a whale whose historical dominant type is Payment
payment_whale = next(
    (acc for acc in WHALES
     if registry.get_stats(acc).tx_types.get("Payment", 0) ==
        max(registry.get_stats(acc).tx_types.values(), default=0)
     and max(registry.get_stats(acc).tx_types.values(), default=0) > 0),
    None
)

if payment_whale:
    engine12 = make_engine()
    shift_alerts = []
    # Flood the window with OfferCreate (different from historical Payment dominance)
    for _ in range(BEHAVIOUR_SHIFT_MIN_TXS + 2):
        alerts = engine12.process_transaction(
            make_tx(account=payment_whale, tx_type="OfferCreate", currency="SOLO")
        )
        shift_alerts += [a for a in alerts if a.alert_type == "BEHAVIOUR_SHIFT"]

    check("fires when dominant tx_type changes from historical",
          len(shift_alerts) >= 1, f"fired {len(shift_alerts)}")
    if shift_alerts:
        check("message contains 🔄", "🔄" in shift_alerts[0].message)
        check("details has historical_dominant_type",
              "historical_dominant_type" in shift_alerts[0].details)
        check("details has window_dominant_type",
              "window_dominant_type" in shift_alerts[0].details)
        check("dominant types differ",
              shift_alerts[0].details["historical_dominant_type"] !=
              shift_alerts[0].details["window_dominant_type"])
else:
    print("  (skipped — no suitable whale found)")

# Too few txs in window should NOT fire
engine13 = make_engine()
too_few_shift = [a for a in engine13.process_transaction(
    make_tx(account=WHALE, tx_type="OfferCreate")
) if a.alert_type == "BEHAVIOUR_SHIFT"]
check("does NOT fire with fewer than BEHAVIOUR_SHIFT_MIN_TXS txs in window",
      len(too_few_shift) == 0)


# ---------------------------------------------------------------------------
print("\n── Test 9: NEW_WHALE_EMERGENCE — core trigger ───────────────────────")
# ---------------------------------------------------------------------------

threshold = registry.whale_tx_threshold
engine14 = make_engine()
emerge_alerts = []

for _ in range(threshold + 1):
    alerts = engine14.process_transaction(make_tx(account=NON_WHALE))
    emerge_alerts += [a for a in alerts if a.alert_type == "NEW_WHALE_EMERGENCE"]

check("fires when non-whale reaches whale tx threshold in session",
      len(emerge_alerts) >= 1, f"fired {len(emerge_alerts)}")
if emerge_alerts:
    check("message contains 🆕", "🆕" in emerge_alerts[0].message)
    check("account is the non-whale", emerge_alerts[0].account == NON_WHALE)
    check("details has session_tx_count", "session_tx_count" in emerge_alerts[0].details)
    check("details has whale_threshold", "whale_threshold" in emerge_alerts[0].details)
    check("session_tx_count >= threshold",
          emerge_alerts[0].details["session_tx_count"] >= threshold)

# Known whale should NOT fire
engine15 = make_engine()
whale_emerge = []
for _ in range(threshold + 1):
    whale_emerge += [a for a in engine15.process_transaction(
        make_tx(account=WHALE)
    ) if a.alert_type == "NEW_WHALE_EMERGENCE"]
check("does NOT fire for already-known whale", len(whale_emerge) == 0)

# Below threshold should NOT fire
engine16 = make_engine()
below_emerge = []
for _ in range(threshold - 1):
    below_emerge += [a for a in engine16.process_transaction(
        make_tx(account=NON_WHALE)
    ) if a.alert_type == "NEW_WHALE_EMERGENCE"]
check("does NOT fire below tx threshold", len(below_emerge) == 0)


# ---------------------------------------------------------------------------
print("\n── Summary ───────────────────────────────────────────────────────────")
# ---------------------------------------------------------------------------

total = passed + failed
print(f"\n  {passed}/{total} tests passed", end="")
print("  \033[92m— all good!\033[0m" if failed == 0 else f"  \033[91m— {failed} failed\033[0m")
print()
