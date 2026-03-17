# XRPL Whale Activity Monitor

> **XRP Commons Hackathon — Challenge 2: Analytics**
> Real-time detection of anomalous whale activity on the XRP Ledger, delivering actionable market insights to retail traders.

---

## Overview

Retail traders on the XRPL have no visibility into what large accounts (whales) are doing. This system bridges that gap:

1. Trains an Isolation Forest model on historical XRPL transaction data
2. Scores every live transaction as it arrives from the XRPL websocket
3. Translates ML anomaly scores into named, plain-English alerts
4. Serves those alerts via a REST API to a live dashboard and Telegram bot

---

## Architecture

```
Historical CSV (272K transactions)
        │
        ▼
┌─────────────────────────────────┐
│  feature_engineering.py         │  Normalise + compute 12 ML features
│  model.py                       │  Train Isolation Forest
└──────────────┬──────────────────┘
               │  models/isolation_forest.pkl
               ▼
┌─────────────────────────────────┐
│  score_realtime.py              │  Stream live XRPL transactions
│                                 │  → normalise → features → risk score
│                                 │  → run alert engine on every tx
└──────────────┬──────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
 data/alerts.json   data/realtime_scored.csv
       │
┌──────┴──────────────────────────┐
│  api.py (Flask)                 │  REST API — serves alerts + whale data
└──────┬──────────────────────────┘
       │
┌──────┴──────────────────────────┐
│  Lovable Frontend               │  Analytics dashboard
│  Telegram Bot                   │  Push notifications
└─────────────────────────────────┘
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Internet connection (for the XRPL websocket)
- Dataset: get `ie_dataset.zip` from your team and unzip it in the repo root

### Install

```bash
pip install -r requirements.txt
pip install flask flask-cors   # for the API
```

### Train the model

```bash
python3 feature_engineering.py && python3 model.py
```

Takes ~30 seconds. Outputs `data/featured_transactions.csv` and `models/isolation_forest.pkl`.

### Stream live transactions with alerts

```bash
python3 score_realtime.py
```

Scores every live transaction, fires alerts to the terminal, and writes them to `data/alerts.json`.

### Start the REST API

```bash
python3 api.py
```

Serves on `http://localhost:5000`. The Lovable frontend and Telegram bot both read from this API.

---

## File Structure

```
├── normalize.py                  # Shared normalisation — raw tx → 10-column schema
├── feature_engineering.py        # Compute 12 ML features from historical CSV
├── model.py                      # Train Isolation Forest + score transactions
├── score_realtime.py             # Stream live XRPL txs → score → run alert engine
├── demo.py                       # Colour-coded terminal demo (no alerts)
├── whale_registry.py             # Identifies the top-5% whale accounts
├── transaction_buffer.py         # 10-min rolling window, per-account state, cooldowns
├── alert_engine.py               # 15 alert types — fires on every scored transaction
├── alerts_writer.py              # Writes/reads data/alerts.json (thread-safe)
├── api.py                        # Flask REST API — 6 endpoints, CORS enabled
├── telegram_bot.py               # ← TODO: Telegram push notifications
├── requirements.txt
├── models/
│   └── isolation_forest.pkl      # Trained model (generated, not in git)
├── data/                         # Generated outputs (not in git)
│   ├── featured_transactions.csv
│   ├── realtime_scored.csv
│   └── alerts.json               # Rolling log of fired alerts
└── ie_dataset/                   # Source data (not in git)
    ├── transactions.csv
    └── ledger.csv
```

---

## REST API Reference

Base URL: `http://localhost:5000`

All responses are JSON. CORS is enabled for all origins.

---

### `GET /api/health`

Simple health check.

**Response**
```json
{
  "status": "ok",
  "whales": 187
}
```

---

### `GET /api/alerts`

Returns the most recent alerts. Alerts are stored newest-first, capped at 500 entries.

**Query parameters**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | int | 50 | Max alerts to return (max 500) |
| `severity` | string | — | Filter by severity: `low`, `medium`, `high`, `critical` |

**Example**
```
GET /api/alerts?limit=20&severity=high
```

**Response**
```json
{
  "count": 3,
  "alerts": [
    {
      "id": "alert_1742123456_rWDwq",
      "timestamp": "2026-03-17T14:00:00Z",
      "account": "rWDwqFVGDFQFooat1TtPq1B1DYCU6ZJrh",
      "alert_type": "TOKEN_ACCUMULATION",
      "severity": "high",
      "risk_score": 0.87,
      "is_anomaly": 1,
      "message": "🐋 Whale active on SOLO: 12 offers in last 10 min",
      "details": {
        "token": "SOLO",
        "offer_count": 12,
        "token_offer_threshold": 5
      }
    }
  ]
}
```

---

### `GET /api/whales`

Returns whale accounts with per-account statistics.

**Query parameters**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | int | 100 | Max accounts to return (max 500) |
| `sort` | string | `tx_count` | Sort field: `tx_count`, `xrp_sent`, `percentile` |

**Example**
```
GET /api/whales?limit=10&sort=tx_count
```

**Response**
```json
{
  "total_whales": 187,
  "whale_tx_threshold": 144,
  "returned": 10,
  "whales": [
    {
      "account": "rWDwqFVGDFQFooat1TtPq1B1DYCU6ZJrh",
      "tx_count": 35201,
      "xrp_sent": 12400.50,
      "xrp_received": 8200.00,
      "unique_destinations": 43,
      "percentile": 0.9997,
      "tx_types": { "OfferCreate": 30000, "Payment": 5000, "OfferCancel": 201 },
      "tokens_traded": ["SOLO", "RLUSD", "XRPH"]
    }
  ]
}
```

---

### `GET /api/whales/<account>`

Returns stats for a single XRPL account (whale or not).

**Example**
```
GET /api/whales/rWDwqFVGDFQFooat1TtPq1B1DYCU6ZJrh
```

**Response (whale)**
```json
{
  "is_whale": true,
  "account": "rWDwqFVGDFQFooat1TtPq1B1DYCU6ZJrh",
  "tx_count": 35201,
  "xrp_sent": 12400.50,
  "xrp_received": 8200.00,
  "unique_destinations": 43,
  "percentile": 0.9997,
  "tx_types": { "OfferCreate": 30000, "Payment": 5000 },
  "tokens_traded": ["SOLO", "RLUSD"]
}
```

**Response (non-whale)**
```json
{
  "is_whale": false,
  "account": "r1AZgefio4GMb57YRQmyUXm9jTH24THkj",
  "tx_count": 12,
  "percentile": 0.21
}
```

---

### `GET /api/stats`

Returns a high-level system summary: registry stats and alert breakdown.

**Response**
```json
{
  "total_accounts": 3686,
  "whale_count": 187,
  "whale_threshold_tx_count": 144,
  "total_alerts_stored": 42,
  "alert_type_breakdown": {
    "LARGE_XRP_TRANSFER": 10,
    "TOKEN_ACCUMULATION": 18,
    "TRANSACTION_BURST": 7,
    "VOLUME_SPIKE": 4,
    "HIGH_RISK_TRANSACTION": 3
  }
}
```

---

### `GET /api/network`

Live rolling-window stats from the 10-minute transaction buffer. Use this to power the "right now" metrics panel — how busy the network is, how many whales are active, and whether the anomaly rate is elevated.

**Response**
```json
{
  "window_minutes": 10,
  "tx_count": 1240,
  "active_accounts": 87,
  "active_whale_count": 14,
  "active_whale_accounts": ["rWDwq...", "rHb9C...", "rPT1J..."],
  "total_xrp_volume": 842300.5,
  "avg_risk_score": 0.38,
  "anomaly_count": 62,
  "anomaly_rate_pct": 5.0,
  "alerts_last_10min": 7
}
```

---

### `GET /api/tokens`

Per-token whale activity from the rolling 10-min window. Only includes tokens where at least one whale has placed or cancelled an offer. Use this to power the token heatmap — which tokens are whales most active on right now.

**Response**
```json
{
  "window_minutes": 10,
  "active_token_count": 4,
  "tokens": [
    {
      "token": "SOLO",
      "whale_count": 5,
      "whale_accounts": ["rWDwq...", "rHb9C...", "rPT1J...", "rG1Qu...", "rN7n4..."],
      "offer_count": 48
    },
    {
      "token": "RLUSD",
      "whale_count": 3,
      "whale_accounts": ["rWDwq...", "rHb9C...", "rG1Qu..."],
      "offer_count": 21
    }
  ]
}
```

---

## Alert Reference

Every alert has the following shape:

```json
{
  "id": "alert_{unix_ts}_{account[:5]}",
  "timestamp": "2026-03-17T14:00:00Z",
  "account": "rXXX...",
  "alert_type": "LARGE_XRP_TRANSFER",
  "severity": "high",
  "risk_score": 0.87,
  "is_anomaly": 1,
  "message": "Plain-English one-liner for the trader",
  "details": { }
}
```

**Severity levels:** `low` → `medium` → `high` → `critical`

Severity is set by the ML risk score:
- `critical` — risk ≥ 0.9 and flagged as anomaly
- `high` — risk ≥ 0.8 or flagged as anomaly
- `medium` — risk ≥ 0.4
- `low` — everything else

---

### Group 1 — Transaction Size & Whale Movements

| Alert Type | Trigger | Example Message |
|------------|---------|-----------------|
| `LARGE_XRP_TRANSFER` | Whale account sends a Payment or OfferCreate above the 99th percentile by size | `🐋 Whale moved 2.4M SOLO → rDEST... (top 0.3% tx size)` |
| `VOLUME_SPIKE` | Network-wide 5-min volume exceeds 3× the historical average | `📈 Network volume spike: 8.2× above average` |
| `HIGH_RISK_TRANSACTION` | Any transaction with risk score ≥ 0.8 and `is_anomaly = 1` | `⚠️ High-risk transaction detected — risk score 0.91` |

---

### Group 2 — Burst & Bot Activity

| Alert Type | Trigger | Example Message |
|------------|---------|-----------------|
| `TRANSACTION_BURST` | Whale account sends ≥ 50 transactions in the last 5 minutes | `⚡ Whale burst: 340 transactions in 5 min` |
| `ABNORMAL_TX_RATE` | Whale account's tx rate is ≥ 3 standard deviations above the network mean | `🤖 Abnormal tx rate: 120.0 tx/min — 8.3× above network average` |

---

### Group 3 — DEX & Trading Activity

| Alert Type | Trigger | Example Message |
|------------|---------|-----------------|
| `TOKEN_ACCUMULATION` | Whale account places ≥ 5 OfferCreate transactions on the same non-XRP token within the 10-min window | `🐋 Whale active on SOLO: 12 offers in last 10 min` |
| `OFFER_CANCEL_SPIKE` | Whale account's cancel ratio (cancels / total offers) ≥ 70%, with at least 5 total offers | `👻 Possible spoofing: 82% of offers cancelled` |
| `MULTI_WHALE_CONVERGENCE` | ≥ 3 distinct whale accounts trade the same non-XRP token within the 10-min window | `🔥 3 whales trading SOLO simultaneously` |

---

### Group 4 — Memo & Spam Detection

| Alert Type | Trigger | Example Message |
|------------|---------|-----------------|
| `MEMO_SPAM` | Any account has `duplicate_memo_count ≥ 10` — the same memo text repeated across many transactions | `📨 Memo spam: 47 transactions with identical memo` |
| `URL_SPAM` | Account has `contains_url = 1` and has sent ≥ 5 URL-containing transactions in the window | `🔗 URL spam: 7 transaction memos with URLs` |
| `HIGH_ENTROPY_MEMO` | Transaction has `memo_entropy ≥ 4.5` and a non-empty memo — indicates encrypted or binary payload | `🔐 High-entropy memo — possible encrypted payload` |

---

### Group 5 — Behaviour Shifts

| Alert Type | Trigger | Example Message |
|------------|---------|-----------------|
| `WALLET_DRAIN` | Whale account's total XRP out in the window reaches ≥ 80% of its full historical XRP sent | `⚠️ Wallet drain: 9400.0 XRP (87.3% of historical) in last 10 min` |
| `BEHAVIOUR_SHIFT` | Whale account has ≥ 10 txs in the window and its dominant transaction type has changed from its historical baseline | `🔄 Behaviour shift: switched from Payment to OfferCreate` |
| `NEW_WHALE_EMERGENCE` | A non-whale account's session transaction count crosses the whale threshold (144 txs) | `🆕 New whale emerging: 145 txs in session` |

---

## Frontend Guide (Lovable)

This section is written for the Lovable AI frontend builder. It describes the exact layout, data sources, and behaviour we want for each part of the dashboard.

### Setup

The backend API runs locally at `http://localhost:5000`. All endpoints support CORS — no proxy needed. Poll every **5 seconds** for live data. Use `fetch` or `axios`.

```js
const API = "http://localhost:5000"
const refresh = () => {
  fetch(`${API}/api/alerts?limit=20`).then(r => r.json()).then(setAlerts)
  fetch(`${API}/api/network`).then(r => r.json()).then(setNetwork)
  fetch(`${API}/api/tokens`).then(r => r.json()).then(setTokens)
}
setInterval(refresh, 5000)
refresh()
```

---

### Visual Style

- **Theme:** Dark background (near-black, e.g. `#0d0f14`). This is a monitoring tool — dark mode only.
- **Accent colours:** Use the severity colour system consistently everywhere:
  - `critical` → red (`#ef4444`)
  - `high` → orange (`#f97316`)
  - `medium` → yellow (`#eab308`)
  - `low` → blue-grey (`#64748b`)
- **Font:** Monospace for addresses and numbers (e.g. `font-mono`). Sans-serif for labels and headings.
- **Layout:** Full-width dashboard. No sidebar. Cards arranged in a responsive grid.
- **Tone:** Data-dense but readable. Think Bloomberg terminal meets modern SaaS.

---

### Page Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  Header: "🐋 XRPL Whale Monitor"   [live indicator dot]  [time]      │
├────────────┬────────────┬────────────┬────────────┬──────────────────┤
│  Whales    │  Txs (10m) │  Anomaly % │  Alerts    │  Active Tokens   │
│  Active    │  in window │  rate      │  (10 min)  │  (whale trades)  │
├────────────┴────────────┴────────────┴────────────┴──────────────────┤
│                                                                        │
│   LEFT (60%)                        RIGHT (40%)                        │
│   ┌──────────────────────────┐      ┌──────────────────────────────┐  │
│   │  Live Alert Feed         │      │  Token Heatmap               │  │
│   │  (scrolling list)        │      │  (which tokens whales trade) │  │
│   └──────────────────────────┘      ├──────────────────────────────┤  │
│   ┌──────────────────────────┐      │  Whale Leaderboard           │  │
│   │  Alert Type Breakdown    │      │  (top 10 most active whales) │  │
│   │  (bar chart)             │      └──────────────────────────────┘  │
│   └──────────────────────────┘                                         │
└──────────────────────────────────────────────────────────────────────┘
```

---

### Component 1 — Header

A single top bar with:
- App name: **"🐋 XRPL Whale Monitor"**
- A pulsing green dot labelled **"LIVE"** when data is fresh (last fetch < 10s ago), grey **"OFFLINE"** otherwise
- Current UTC time (update every second)

---

### Component 2 — Stats Row (5 KPI cards)

Poll: `GET /api/network` every 5 seconds.

| Card | Field | Label | Format |
|------|-------|-------|--------|
| Whales Active | `active_whale_count` | Whales Active (10 min) | Number, large font |
| Transactions | `tx_count` | Txs in Last 10 Min | Number with comma separator |
| Anomaly Rate | `anomaly_rate_pct` | Anomaly Rate | `X.X%` — colour red if > 10% |
| Alerts Fired | `alerts_last_10min` | Alerts (10 min) | Number — colour orange if > 0 |
| Avg Risk Score | `avg_risk_score` | Avg Risk Score | `0.XX` — colour red if > 0.6 |

---

### Component 3 — Live Alert Feed

Poll: `GET /api/alerts?limit=20` every 5 seconds.

Display as a scrolling card list, newest at the top. Each alert card shows:

```
┌─────────────────────────────────────────────────────┐
│  [CRITICAL]  TOKEN_ACCUMULATION          2 min ago  │
│  🐋 Whale active on SOLO: 12 offers in last 10 min  │
│  Account: rWDwq...FooZ    Risk: 0.87  [anomaly]     │
└─────────────────────────────────────────────────────┘
```

- Left-border colour matches severity (red / orange / yellow / grey)
- Severity badge in the top-left (coloured pill)
- Alert type in monospace, timestamp relative (e.g. "2 min ago")
- Message in plain text — this is the main content, make it readable
- Account address truncated to `rXXXX...XXXX` (first 6, last 4 chars)
- Risk score shown as a number; if `is_anomaly = 1`, show a small red "anomaly" badge
- Clicking an alert card should open a detail drawer/modal showing the full `details` object as formatted key-value pairs

**Alert type → emoji mapping** (already in the message, but use for icons too):
- `LARGE_XRP_TRANSFER` → 🐋
- `VOLUME_SPIKE` → 📈
- `HIGH_RISK_TRANSACTION` → ⚠️
- `TRANSACTION_BURST` → ⚡
- `ABNORMAL_TX_RATE` → 🤖
- `TOKEN_ACCUMULATION` → 🐋
- `OFFER_CANCEL_SPIKE` → 👻
- `MULTI_WHALE_CONVERGENCE` → 🔥
- `MEMO_SPAM` → 📨
- `URL_SPAM` → 🔗
- `HIGH_ENTROPY_MEMO` → 🔐
- `WALLET_DRAIN` → ⚠️
- `BEHAVIOUR_SHIFT` → 🔄
- `NEW_WHALE_EMERGENCE` → 🆕

---

### Component 4 — Alert Type Breakdown

Poll: `GET /api/stats` every 10 seconds.

Use `alert_type_breakdown` to render a horizontal bar chart showing how many times each alert type has fired in total. Sort by count descending. Colour bars by the typical severity of that alert type:

- `HIGH_RISK_TRANSACTION`, `WALLET_DRAIN`, `MULTI_WHALE_CONVERGENCE` → red
- `TOKEN_ACCUMULATION`, `TRANSACTION_BURST`, `ABNORMAL_TX_RATE`, `BEHAVIOUR_SHIFT` → orange
- `OFFER_CANCEL_SPIKE`, `MEMO_SPAM`, `URL_SPAM`, `HIGH_ENTROPY_MEMO` → yellow
- `VOLUME_SPIKE`, `LARGE_XRP_TRANSFER`, `NEW_WHALE_EMERGENCE` → blue

---

### Component 5 — Token Heatmap

Poll: `GET /api/tokens` every 5 seconds.

Show each active token as a card/tile. Sort by `whale_count` descending.

```
┌─────────────────────────────────┐
│  SOLO                           │
│  ████████████  5 whales         │
│  48 offers in last 10 min       │
└─────────────────────────────────┘
```

- Token name large and prominent
- A mini bar or dot indicators for whale count (max 10 dots)
- Offer count as secondary info
- Highlight the top token with a gold border or glow effect
- If no tokens are active, show: "No whale token activity in the last 10 minutes"

---

### Component 6 — Whale Leaderboard

Poll: `GET /api/whales?limit=10&sort=tx_count` once on load (historical data, no need to poll frequently).

Show the top 10 whales as a ranked table:

| # | Account | Tx Count | Top Activity | Tokens Traded | Percentile |
|---|---------|----------|-------------|---------------|------------|
| 1 | rWDwq...FooZ | 35,201 | OfferCreate | SOLO, RLUSD | Top 0.03% |
| 2 | rHb9C...XYZa | 31,450 | OfferCreate | XRPH, SOLO | Top 0.05% |

- Rank number on the left (1, 2, 3...)
- Account address truncated and monospaced (`rXXXX...XXXX`)
- "Top Activity" = the tx_type with the highest count from `tx_types`
- "Tokens Traded" = first 3 tokens from `tokens_traded`, then "+ N more" if there are more
- "Percentile" = `(1 - percentile) * 100` formatted as "Top X.XX%"
- Clicking a row opens a detail card/modal using `GET /api/whales/<account>`

**Whale detail modal** (on row click):
- Full address (copyable)
- Historical tx count, XRP sent, XRP received
- Bar chart of tx type breakdown (from `tx_types`)
- All tokens ever traded
- Recent alerts for this account (filter `GET /api/alerts` by account client-side)

---

### Data Freshness & Error States

- If any fetch fails, show a subtle "⚠ Data may be stale" warning in the header — do not crash or show empty components
- If `alerts` array is empty, show: "No alerts yet — waiting for whale activity..."
- If `tokens` array is empty, show: "No whale token activity in the last 10 minutes"
- If `active_whale_count` is 0, the network stats row should still render (show 0s, not blank)

---

## Data Pipeline (Internal)

### Step 1 — Normalisation (`normalize.py`)

Converts both historical CSV rows and live websocket messages into a unified 10-column schema:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime (UTC) | Ledger close time |
| `ledger_index` | int | Ledger sequence number |
| `tx_hash` | str | Unique transaction identifier |
| `tx_type` | str | Payment, OfferCreate, NFTokenBurn, etc. |
| `account` | str | Sender address |
| `destination` | str | Receiver address (Payments only) |
| `fee` | int | Transaction fee in drops |
| `amount_xrp` | float | Transaction value in XRP equivalent |
| `currency` | str | "XRP" for native, token ticker otherwise |
| `issuer` | str | Token issuer address (empty for XRP) |

### Step 2 — Feature Engineering (`feature_engineering.py`)

Computes 12 features per transaction used by the ML model:

| Feature | Description |
|---------|-------------|
| `tx_size_percentile` | Where this transaction's amount ranks vs. all historical txs (0–1) |
| `is_large_tx` | 1 if amount exceeds the 99th percentile |
| `wallet_balance_change` | Net XRP flow for this account (negative = net sender) |
| `rolling_tx_count_5m` | Transactions from this account in the last 5 minutes |
| `tx_per_minute` | Average transaction rate for this account |
| `tx_rate_z_score` | How abnormal this account's rate is vs. the network mean |
| `total_volume_5m` | Total XRP volume across all accounts in the last 5 minutes |
| `volume_spike_ratio` | Current 5-min volume vs. historical average (> 1 = above average) |
| `memo_entropy` | Shannon entropy of memo text (higher = more random/encrypted) |
| `memo_length` | Character length of decoded memo |
| `duplicate_memo_count` | How many times this exact memo text appears in the dataset |
| `contains_url` | 1 if the memo contains http / https / www |

### Step 3 — Anomaly Detection (`model.py`)

Trains an **Isolation Forest** (unsupervised) on the 12 features:

- No labelled data required — learns "normal" from the data itself
- `contamination = 0.05` — expects ~5% of transactions to be anomalous
- `n_estimators = 200` — 200 trees for stable results
- Outputs `risk_score` (0–1, where 1 = most anomalous) and `is_anomaly` flag

### Step 4 — Realtime Scoring + Alerts (`score_realtime.py`)

For every live transaction from `wss://xrplcluster.com/`:

1. Normalise via `normalize.py`
2. Compute 12 features using historical stats as reference baseline
3. Score with the trained Isolation Forest
4. Run the alert engine — check all 15 alert types
5. Write any fired alerts to `data/alerts.json` via `alerts_writer.py`
6. Append scored transaction to `data/realtime_scored.csv`

---

## Whale Registry (`whale_registry.py`)

Identifies whale accounts from historical data before any alert logic runs.

**How it works:**
1. Loads transaction data (featured CSV if available, raw CSV otherwise)
2. Computes per-account statistics across all 271,990 historical transactions
3. Ranks every account by total transaction count
4. Classifies the top 5% (≥ 144 txs) as whales — **187 accounts** out of 3,686

**Why transaction count, not XRP volume?**
The dataset is almost entirely token-based — native XRP payments are rare, so XRP volume is near-zero for most accounts. Transaction count is the most reliable activity signal.

**Python API:**
```python
registry = WhaleRegistry.build()
registry.is_whale("rXXX...")       # True / False
registry.get_stats("rXXX...")      # AccountStats dataclass
registry.percentile("rXXX...")     # 0.0 – 1.0
registry.whale_accounts            # set of all whale addresses
registry.summary()                 # {"total_accounts": 3686, "whale_count": 187, ...}
```

---

## Transaction Buffer (`transaction_buffer.py`)

The alert engine's memory layer. A single transaction tells you very little — but 340 transactions from the same account in 10 minutes tells you a lot.

**Rolling window:** 10 minutes
**Cooldown:** 2 minutes per alert type per account (prevents duplicate alerts)

Every scored transaction is passed to `buffer.add()`. Alert detectors then call `buffer.get_account_state(account)` to query what that account has been doing in the current window.

**`AccountState` fields:**

| Field | Description |
|-------|-------------|
| `tx_count` | Transactions from this account in the window |
| `xrp_out` | Total XRP sent |
| `xrp_in` | Total XRP received |
| `net_xrp` | `xrp_in - xrp_out` (positive = net receiver) |
| `offer_creates` | OfferCreate count |
| `offer_cancels` | OfferCancel count |
| `offer_cancel_ratio` | `cancels / (creates + cancels)` |
| `tx_types` | Breakdown by type e.g. `{"OfferCreate": 12, "Payment": 5}` |
| `tokens` | Per-token activity e.g. `{"SOLO": {"buys": 3, "sells": 0}}` |
| `risk_scores` | ML risk score for each tx in the window |
| `avg_risk_score` | Mean ML risk score |
| `max_risk_score` | Highest ML risk score in the window |
| `memo_texts` | Decoded memo strings |
| `last_seen` | Timestamp of the most recent transaction |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data pipeline | Python 3.10+ · pandas · numpy |
| ML model | scikit-learn (Isolation Forest) · joblib |
| XRPL client | xrpl-py (websocket streaming) |
| Alert engine | Python — 15 alert types across 5 groups |
| Alert storage | JSON file (`data/alerts.json`) |
| REST API | Flask · flask-cors |
| Notifications | Telegram Bot API |
| Frontend | Lovable |

---

## Limitations

- **Anomaly ≠ fraud** — the model flags unusual patterns, not confirmed malicious activity. A legitimate whale transaction may score high.
- **Short training window** — the historical dataset covers ~2.5 hours of XRPL activity. More data would give the model a stronger baseline.
- **Unsupervised only** — no labelled examples of known scam transactions. A supervised model would be more precise given labelled data.
- **Realtime features are approximate** — rolling window features in realtime use an in-memory buffer, not the full historical dataset.

---

## Roadmap

- [x] Data normalisation pipeline
- [x] Feature engineering (12 features)
- [x] Isolation Forest model training
- [x] Realtime transaction scoring
- [x] Whale registry (top 5% accounts, 187 whales)
- [x] Transaction buffer (10-min rolling window, per-account state, cooldowns)
- [x] Alert engine (15 alert types across 5 groups)
- [x] Alert storage (`alerts_writer.py` — atomic JSON, 500-entry rolling cap)
- [x] Flask REST API (`api.py` — 6 endpoints, CORS enabled)
- [ ] Lovable analytics dashboard
- [ ] Telegram bot push notifications
- [ ] User-configurable alert thresholds
- [ ] Cloud deployment
