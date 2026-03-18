"""Quick exploration of the XRPL transactions dataset."""

import ast
import pandas as pd
import numpy as np

DATA_PATH = "/Users/artemabesadze/Desktop/XRPL Commons Work/XRPL_ML_WhaleProject/ie_dataset/transactions.csv"

COLS = [
    "account",
    "transaction_type",
    "destination",
    "amount",
    "fee",
    "ledger_index",
    "ledger_close_time_human",
    "taker_gets",
    "taker_pays",
]

print("Loading data...")
df = pd.read_csv(DATA_PATH, usecols=COLS, low_memory=False)
df["date"] = pd.to_datetime(df["ledger_close_time_human"], errors="coerce", utc=True)
print(f"Done. Shape: {df.shape}\n")


# ── 1. Basic stats ────────────────────────────────────────────────────────────
print("=" * 60)
print("1. BASIC STATS")
print("=" * 60)
print(f"Total rows      : {len(df):,}")
print(f"Unique accounts : {df['account'].nunique():,}")
print(f"Null amounts    : {df['amount'].isna().sum():,}")
print(f"Earliest        : {df['date'].min()}")
print(f"Latest          : {df['date'].max()}")
span_minutes = (df['date'].max() - df['date'].min()).total_seconds() / 60
print(f"Span            : {span_minutes:.0f} minutes ({span_minutes/60:.1f} hours)")


# ── 2. Transaction type breakdown ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. TRANSACTION TYPE BREAKDOWN")
print("=" * 60)
for tx_type, count in df["transaction_type"].value_counts().items():
    print(f"  {tx_type:<30} {count:>8,}  ({count/len(df)*100:.1f}%)")


# ── 3. Parse amount field ─────────────────────────────────────────────────────
def parse_amount(val):
    """Return (value_float, currency, issuer). XRP drops → convert to XRP."""
    if pd.isna(val):
        return np.nan, "XRP", ""
    try:
        # Plain drops integer → native XRP
        return float(val) / 1_000_000, "XRP", ""
    except (ValueError, TypeError):
        pass
    try:
        d = ast.literal_eval(str(val))
        if isinstance(d, dict):
            return float(d.get("value", 0)), d.get("currency", ""), d.get("issuer", "")
    except Exception:
        pass
    return np.nan, "", ""

print("\nParsing amount field (may take a moment)...")
parsed = df["amount"].apply(parse_amount)
df["amount_value"] = parsed.apply(lambda x: x[0])
df["currency"]     = parsed.apply(lambda x: x[1])
df["issuer"]       = parsed.apply(lambda x: x[2])

payments = df[df["transaction_type"] == "Payment"].copy()
xrp_pay  = payments[payments["currency"] == "XRP"]
tok_pay  = payments[payments["currency"] != "XRP"]

print(f"\nPayments total     : {len(payments):,}")
print(f"  Native XRP       : {len(xrp_pay):,}")
print(f"  Token payments   : {len(tok_pay):,}")


# ── 4. XRP payment distribution ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. XRP PAYMENT SIZE DISTRIBUTION")
print("=" * 60)
if len(xrp_pay):
    desc = xrp_pay["amount_value"].describe(percentiles=[.5, .75, .90, .95, .99])
    for stat, val in desc.items():
        print(f"  {stat:<8}: {val:>18,.2f} XRP")
else:
    print("  No native XRP payments found.")


# ── 5. Token payment breakdown ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. TOP TOKENS BY PAYMENT COUNT")
print("=" * 60)
if len(tok_pay):
    for cur, cnt in tok_pay["currency"].value_counts().head(10).items():
        print(f"  {str(cur)[:40]:<42} {cnt:>7,}")


# ── 6. Whale identification ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. WHALE IDENTIFICATION")
print("   (top 5% of accounts by total XRP sent OR total tx count)")
print("=" * 60)

# By XRP volume (if any XRP payments)
if len(xrp_pay):
    vol_by_account = xrp_pay.groupby("account")["amount_value"].sum().sort_values(ascending=False)
    threshold = vol_by_account.quantile(0.95)
    whales_vol = vol_by_account[vol_by_account >= threshold]
    print(f"\nBy XRP volume — threshold: {threshold:,.2f} XRP")
    print(f"Whale accounts: {len(whales_vol):,}")
    for acc, vol in whales_vol.head(5).items():
        print(f"  {acc}  →  {vol:>14,.2f} XRP")

# By transaction count (works regardless of currency)
tx_by_account = df.groupby("account").size().sort_values(ascending=False)
threshold_tx = tx_by_account.quantile(0.95)
whales_tx = tx_by_account[tx_by_account >= threshold_tx]
print(f"\nBy tx count — threshold: {threshold_tx:.0f} txs")
print(f"Whale accounts: {len(whales_tx):,}")
print("Top 10 most active accounts:")
for acc, cnt in whales_tx.head(10).items():
    print(f"  {acc}  →  {cnt:>6,} txs")


# ── 7. OfferCreate sample (DEX activity) ─────────────────────────────────────
print("\n" + "=" * 60)
print("6. DEX ACTIVITY SAMPLE (OfferCreate — taker_gets/taker_pays)")
print("=" * 60)
offers = df[df["transaction_type"] == "OfferCreate"].head(5)
print(offers[["account", "taker_gets", "taker_pays"]].to_string(index=False))


# ── 8. Activity per 15-minute bucket ─────────────────────────────────────────
print("\n" + "=" * 60)
print("7. TRANSACTION VOLUME PER 15-MINUTE WINDOW")
print("=" * 60)
df["bucket"] = df["date"].dt.floor("15min")
buckets = df.groupby("bucket").size()
max_count = buckets.max()
for bucket, count in buckets.items():
    bar = "█" * int(count / max_count * 30)
    print(f"  {bucket}  {count:>6,}  {bar}")


print("\nDone.")

