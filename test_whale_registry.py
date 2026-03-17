"""
Tests for whale_registry.py

Run with:
    python3 test_whale_registry.py
"""

import sys
from whale_registry import WhaleRegistry, AccountStats, WHALE_PERCENTILE

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
print("\n── Building registry from real data ──────────────────────────────────")
# ---------------------------------------------------------------------------

try:
    registry = WhaleRegistry.build(verbose=True)
    build_ok = True
except Exception as e:
    print(f"  FATAL: could not build registry: {e}")
    sys.exit(1)

print()

# ---------------------------------------------------------------------------
print("── Test 1: Basic shape ───────────────────────────────────────────────")
# ---------------------------------------------------------------------------

summary = registry.summary()

check("total_accounts > 0",
      summary["total_accounts"] > 0,
      f"got {summary['total_accounts']}")

check("whale_count > 0",
      summary["whale_accounts"] > 0,
      f"got {summary['whale_accounts']}")

check(f"whale % is close to {WHALE_PERCENTILE*100:.0f}%  (within 1–10%)",
      1 <= summary["whale_pct"] <= 10,
      f"got {summary['whale_pct']}%")

check("whale_tx_threshold > 0",
      summary["whale_tx_threshold"] > 0,
      f"got {summary['whale_tx_threshold']}")

print(f"\n  summary: {summary}")

# ---------------------------------------------------------------------------
print("\n── Test 2: Known top accounts are whales ─────────────────────────────")
# ---------------------------------------------------------------------------

# From our exploration, these 3 accounts had 30K+ transactions — definitely whales
KNOWN_WHALES = [
    "rUg8ac5ikpTaWk5RPei8xuYkNEyUs53G1i",
    "rBTwLga3i2gz3doX6Gva3MgEV8ZCD8jjah",
    "rQJgPT6xhpT5Jr6GhcQQSWH3qYq3dyFSqY",
]

for acc in KNOWN_WHALES:
    check(f"is_whale({acc[:12]}...)",
          registry.is_whale(acc),
          f"tx_count={registry.get_stats(acc).tx_count if registry.get_stats(acc) else 'unknown'}")

# ---------------------------------------------------------------------------
print("\n── Test 3: get_stats returns correct types ────────────────────────────")
# ---------------------------------------------------------------------------

sample_whale = KNOWN_WHALES[0]
stats = registry.get_stats(sample_whale)

check("get_stats returns AccountStats",
      isinstance(stats, AccountStats))

check("stats.tx_count is int > 0",
      isinstance(stats.tx_count, int) and stats.tx_count > 0,
      f"got {stats.tx_count}")

check("stats.percentile is float in (0, 1]",
      isinstance(stats.percentile, float) and 0 < stats.percentile <= 1.0,
      f"got {stats.percentile}")

check("stats.is_whale is True for known whale",
      stats.is_whale is True)

check("stats.xrp_sent >= 0",
      stats.xrp_sent >= 0,
      f"got {stats.xrp_sent}")

check("stats.unique_destinations >= 0",
      stats.unique_destinations >= 0,
      f"got {stats.unique_destinations}")

check("stats.tx_types is non-empty dict",
      isinstance(stats.tx_types, dict) and len(stats.tx_types) > 0,
      f"got {stats.tx_types}")

print(f"\n  Sample whale stats for {sample_whale[:16]}...:")
print(f"    tx_count          : {stats.tx_count:,}")
print(f"    percentile        : {stats.percentile:.4f}")
print(f"    xrp_sent          : {stats.xrp_sent:,.4f}")
print(f"    xrp_received      : {stats.xrp_received:,.4f}")
print(f"    unique_destinations: {stats.unique_destinations}")
print(f"    tokens_traded     : {len(stats.tokens_traded)} tokens")
print(f"    tx_types          : {stats.tx_types}")

# ---------------------------------------------------------------------------
print("\n── Test 4: Unknown account returns safe defaults ─────────────────────")
# ---------------------------------------------------------------------------

FAKE = "rFAKEACCOUNTTHATDOESNOTEXIST123456"

check("is_whale(unknown) returns False",
      registry.is_whale(FAKE) is False)

check("get_stats(unknown) returns None",
      registry.get_stats(FAKE) is None)

check("percentile(unknown) returns 0.0",
      registry.percentile(FAKE) == 0.0)

# ---------------------------------------------------------------------------
print("\n── Test 5: percentile() is consistent with is_whale() ────────────────")
# ---------------------------------------------------------------------------

# Every whale should have percentile >= WHALE_PERCENTILE
whale_pcts = [
    registry.percentile(acc)
    for acc in registry.whale_accounts
]
# Allow a small tolerance: accounts exactly at the boundary score just below
# WHALE_PERCENTILE due to how searchsorted works, but their tx_count still
# meets the threshold. We accept anything within 1% of the cutoff.
TOLERANCE = 0.01
all_above_threshold = all(p >= WHALE_PERCENTILE - TOLERANCE for p in whale_pcts)
check(f"all whale accounts have percentile >= {WHALE_PERCENTILE - TOLERANCE:.2f} (threshold ± tolerance)",
      all_above_threshold,
      f"min percentile among whales: {min(whale_pcts):.4f}" if whale_pcts else "no whales")

# ---------------------------------------------------------------------------
print("\n── Test 6: whale_accounts property is consistent ─────────────────────")
# ---------------------------------------------------------------------------

whale_set = registry.whale_accounts
check("whale_accounts is a set", isinstance(whale_set, set))
check("whale_accounts count matches whale_count",
      len(whale_set) == registry.whale_count,
      f"{len(whale_set)} vs {registry.whale_count}")

for acc in KNOWN_WHALES:
    check(f"{acc[:12]}... in whale_accounts set",
          acc in whale_set)

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
