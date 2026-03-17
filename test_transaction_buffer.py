"""
Tests for transaction_buffer.py

Run with:
    python3 test_transaction_buffer.py
"""

import time
from datetime import datetime, timezone, timedelta
from transaction_buffer import TransactionBuffer, AccountState, NetworkState, WINDOW_SECONDS, COOLDOWN_SECONDS

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


def make_tx(account, seconds_ago=0, tx_type="Payment", amount_xrp=100.0,
            currency="XRP", destination="rDEST", risk_score=0.1,
            is_anomaly=0, memo_text=""):
    """Build a minimal scored transaction dict."""
    ts = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    return {
        "timestamp": ts,
        "account": account,
        "destination": destination,
        "tx_type": tx_type,
        "amount_xrp": amount_xrp,
        "currency": currency,
        "risk_score": risk_score,
        "is_anomaly": is_anomaly,
        "memo_text": memo_text,
    }


# ---------------------------------------------------------------------------
print("\n── Test 1: Buffer basics ─────────────────────────────────────────────")
# ---------------------------------------------------------------------------

buf = TransactionBuffer()

check("starts empty", buf.size == 0)

buf.add(make_tx("rA"))
buf.add(make_tx("rB"))
buf.add(make_tx("rA"))

check("size = 3 after adding 3 txs", buf.size == 3, f"got {buf.size}")
check("active_accounts contains rA and rB",
      {"rA", "rB"} == buf.active_accounts)


# ---------------------------------------------------------------------------
print("\n── Test 2: Window pruning ────────────────────────────────────────────")
# ---------------------------------------------------------------------------

buf2 = TransactionBuffer(window_seconds=60)  # 1-minute window for testing

# Add 2 old transactions (2 minutes ago — outside window)
buf2.add(make_tx("rA", seconds_ago=130))
buf2.add(make_tx("rA", seconds_ago=120))
# Add 2 recent transactions (inside window)
buf2.add(make_tx("rA", seconds_ago=30))
buf2.add(make_tx("rA", seconds_ago=10))

check("old txs pruned, only 2 recent remain",
      buf2.size == 2, f"got {buf2.size}")


# ---------------------------------------------------------------------------
print("\n── Test 3: AccountState correctness ─────────────────────────────────")
# ---------------------------------------------------------------------------

buf3 = TransactionBuffer()

# rWHALE sends XRP, creates offers, cancels one
buf3.add(make_tx("rWHALE", tx_type="Payment",      amount_xrp=5000.0, currency="XRP",  risk_score=0.3))
buf3.add(make_tx("rWHALE", tx_type="OfferCreate",  amount_xrp=200.0,  currency="SOLO", risk_score=0.5))
buf3.add(make_tx("rWHALE", tx_type="OfferCreate",  amount_xrp=300.0,  currency="SOLO", risk_score=0.6))
buf3.add(make_tx("rWHALE", tx_type="OfferCancel",  amount_xrp=0.0,    currency="XRP",  risk_score=0.2))
buf3.add(make_tx("rWHALE", tx_type="Payment",      amount_xrp=1000.0, currency="XRP",  risk_score=0.8, memo_text="hello"))

# rOTHER receives XRP as destination
buf3.add(make_tx("rSENDER", tx_type="Payment", amount_xrp=999.0, currency="XRP",
                 destination="rWHALE", risk_score=0.1))

state = buf3.get_account_state("rWHALE")

check("returns AccountState", isinstance(state, AccountState))
check("tx_count = 5",         state.tx_count == 5,          f"got {state.tx_count}")
check("xrp_out = 6000.0",     state.xrp_out == 6000.0,      f"got {state.xrp_out}")
check("xrp_in = 999.0 (received as destination)",
                               state.xrp_in == 999.0,        f"got {state.xrp_in}")
check("offer_creates = 2",    state.offer_creates == 2,      f"got {state.offer_creates}")
check("offer_cancels = 1",    state.offer_cancels == 1,      f"got {state.offer_cancels}")
check("offer_cancel_ratio = 1/3",
      abs(state.offer_cancel_ratio - (1/3)) < 0.001,         f"got {state.offer_cancel_ratio:.4f}")
check("memo_texts has 1 entry",  len(state.memo_texts) == 1, f"got {state.memo_texts}")
check("risk_scores has 5 entries", len(state.risk_scores) == 5, f"got {len(state.risk_scores)}")
check("avg_risk_score ≈ 0.48", abs(state.avg_risk_score - 0.48) < 0.01, f"got {state.avg_risk_score:.4f}")
check("max_risk_score = 0.8",  state.max_risk_score == 0.8,  f"got {state.max_risk_score}")
check("net_xrp = 999 - 6000 = -5001",
      abs(state.net_xrp - (-5001.0)) < 0.01,                 f"got {state.net_xrp}")
check("Payment in tx_types",   "Payment" in state.tx_types)
check("OfferCreate count = 2", state.tx_types.get("OfferCreate") == 2)

# Unknown account returns empty but valid state
empty = buf3.get_account_state("rNOBODY")
check("unknown account → tx_count = 0",      empty.tx_count == 0)
check("unknown account → avg_risk_score = 0", empty.avg_risk_score == 0.0)


# ---------------------------------------------------------------------------
print("\n── Test 4: NetworkState correctness ─────────────────────────────────")
# ---------------------------------------------------------------------------

buf4 = TransactionBuffer()
buf4.add(make_tx("rA", amount_xrp=100.0, currency="XRP",  risk_score=0.2, is_anomaly=0))
buf4.add(make_tx("rB", amount_xrp=200.0, currency="XRP",  risk_score=0.9, is_anomaly=1))
buf4.add(make_tx("rC", amount_xrp=0.0,   currency="SOLO", risk_score=0.4, is_anomaly=0))

net = buf4.get_network_state()

check("returns NetworkState",         isinstance(net, NetworkState))
check("tx_count = 3",                 net.tx_count == 3,        f"got {net.tx_count}")
check("active_accounts = 3",          net.active_accounts == 3, f"got {net.active_accounts}")
check("total_xrp_volume = 300.0",     net.total_xrp_volume == 300.0, f"got {net.total_xrp_volume}")
check("anomaly_count = 1",            net.anomaly_count == 1,   f"got {net.anomaly_count}")
check("avg_risk_score ≈ 0.5",
      abs(net.avg_risk_score - 0.5) < 0.01,                    f"got {net.avg_risk_score:.4f}")

# Empty buffer returns zeros
empty_net = TransactionBuffer().get_network_state()
check("empty buffer → tx_count = 0", empty_net.tx_count == 0)


# ---------------------------------------------------------------------------
print("\n── Test 5: Cooldown management ──────────────────────────────────────")
# ---------------------------------------------------------------------------

buf5 = TransactionBuffer(cooldown_seconds=1)   # 1-second cooldown for testing

check("not on cooldown before any alert",
      not buf5.is_on_cooldown("rA", "TRANSACTION_BURST"))

buf5.set_cooldown("rA", "TRANSACTION_BURST")

check("on cooldown immediately after set",
      buf5.is_on_cooldown("rA", "TRANSACTION_BURST"))

check("different alert type not on cooldown",
      not buf5.is_on_cooldown("rA", "VOLUME_SPIKE"))

check("different account not on cooldown",
      not buf5.is_on_cooldown("rB", "TRANSACTION_BURST"))

# Wait for cooldown to expire
time.sleep(1.1)
check("cooldown expires after window",
      not buf5.is_on_cooldown("rA", "TRANSACTION_BURST"))

# Manual clear
buf5.set_cooldown("rA", "WALLET_DRAIN")
buf5.clear_cooldown("rA", "WALLET_DRAIN")
check("clear_cooldown removes it immediately",
      not buf5.is_on_cooldown("rA", "WALLET_DRAIN"))


# ---------------------------------------------------------------------------
print("\n── Test 6: Config constants ──────────────────────────────────────────")
# ---------------------------------------------------------------------------

check(f"WINDOW_SECONDS = 600 (10 min)",   WINDOW_SECONDS == 600,   f"got {WINDOW_SECONDS}")
check(f"COOLDOWN_SECONDS = 120 (2 min)",  COOLDOWN_SECONDS == 120, f"got {COOLDOWN_SECONDS}")

default_buf = TransactionBuffer()
check("default buffer uses WINDOW_SECONDS",
      default_buf.window_seconds == WINDOW_SECONDS)
check("default buffer uses COOLDOWN_SECONDS",
      default_buf.cooldown_seconds == COOLDOWN_SECONDS)


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
