# XRPL Whale Activity Monitor

> **XRP Commons Hackathon — Challenge 2: Analytics**
> Real-time detection of anomalous whale activity on the XRP Ledger, delivering actionable market insights to retail traders.

---

## What it does

1. Loads ~272,000 historical XRPL transactions from CSV files
2. Normalises them into a clean 10-column schema
3. Engineers 12 features per transaction (size, velocity, memo patterns, volume spikes)
4. Trains an Isolation Forest (unsupervised anomaly detection) on the historical data
5. Connects to the XRPL websocket and scores every live transaction in real time
6. Runs scored transactions through an alert engine — generating 15 named, plain-English alerts
7. Serves alerts via a REST API to a live dashboard and Telegram bot

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
│                                 │  → upload to Supabase
└──────────────┬──────────────────┘
               │
       ┌───────┴───────────────┐
       ▼                       ▼
 data/alerts.json         data/realtime_scored.csv
       │
┌──────┴──────────────────────────┐
│  api.py (Flask)                 │  REST API — 6 endpoints, CORS enabled
└──────┬──────────────────────────┘
       │
┌──────┴──────────────────────────┐
│  Lovable Frontend               │  Analytics dashboard
│  telegram_bot.py                │  Push notifications (@XRPL_Whale_WatchBot)
└─────────────────────────────────┘
```

---

## Project Structure

```
XRP-Commons-Hackathon/
│
├── ie_dataset/                     # Historical data (not in git, see setup)
│   ├── ledger.csv                  #   2,373 ledger records
│   └── transactions.csv            #   271,990 transaction records
│
├── data/                           # Generated outputs (not in git)
│   ├── featured_transactions.csv   #   Historical data + 12 features
│   ├── scored_transactions.csv     #   Historical data + risk scores
│   ├── realtime_scored.csv         #   Realtime data + risk scores
│   ├── alerts.json                 #   Rolling log of fired alerts (cap 500)
│   └── subscribers.json            #   Telegram bot subscriber list
│
├── models/
│   └── isolation_forest.pkl        # Trained model (generated, not in git)
│
├── normalize.py                    # Shared normalisation — raw tx → 10-column schema
├── feature_engineering.py          # Compute 12 ML features from historical CSV
├── model.py                        # Train Isolation Forest + score transactions
├── score_realtime.py               # Stream live txs → score → alert engine → Supabase
├── whale_registry.py               # Identifies top-5% whale accounts (187 whales)
├── transaction_buffer.py           # 10-min rolling window, per-account state, cooldowns
├── alert_engine.py                 # 15 alert types across 5 groups
├── alerts_writer.py                # Atomic JSON alert store (thread-safe)
├── api.py                          # Flask REST API — 6 endpoints, CORS enabled
├── telegram_bot.py                 # Push notifications (@XRPL_Whale_WatchBot)
├── supabase_uploader.py            # Uploads scored transactions to Supabase
├── demo.py                         # Colour-coded terminal demo
├── xrpl_historical.py              # Loads historical CSVs → normalised CSV
├── xrpl_realtime.py                # Streams live transactions → raw CSV
├── .env                            # Secrets — never committed (see .env.example)
├── .env.example                    # Template for environment variables
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Internet connection (for the XRPL websocket)
- Dataset: get `ie_dataset.zip` from your team and unzip in the repo root

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install flask flask-cors python-telegram-bot python-dotenv
```

### 2. Add the dataset

```bash
unzip ie_dataset.zip
```

### 3. Train the model

```bash
python3 feature_engineering.py && python3 model.py
```

Takes ~30 seconds. Outputs `data/featured_transactions.csv` and `models/isolation_forest.pkl`.

### 4. Configure secrets

```bash
cp .env.example .env
# Edit .env and fill in your TELEGRAM_BOT_TOKEN
```

### 5. Run the full system (3 terminals)

```bash
# Terminal 1 — live scoring + alert engine
python3 score_realtime.py

# Terminal 2 — REST API for frontend
python3 api.py

# Terminal 3 — Telegram push notifications
python3 telegram_bot.py
```

Users subscribe to the Telegram bot by clicking the button on the dashboard:
`https://t.me/XRPL_Whale_WatchBot?start=1`

---

## How the Pipeline Works

### Step 1 — Normalisation (`normalize.py`)

Both historical CSV and live websocket data arrive in different formats. `normalize.py` converts both into the same flat 10-column schema:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime (UTC) | When the ledger closed |
| `ledger_index` | int | Ledger sequence number |
| `tx_hash` | str | Unique transaction identifier |
| `tx_type` | str | Payment, OfferCreate, NFTokenBurn, etc. |
| `account` | str | Sender/initiator address |
| `destination` | str | Receiver address (Payments only) |
| `fee` | int | Transaction fee in drops |
| `amount_xrp` | float | Transaction value in XRP |
| `currency` | str | "XRP" for native, token ticker otherwise |
| `issuer` | str | Token issuer address (empty for XRP) |

### Step 2 — Feature Engineering (`feature_engineering.py`)

Computes 12 features per transaction:

| Feature | Description |
|---------|-------------|
| `tx_size_percentile` | Where this amount ranks among all transactions (0–1) |
| `is_large_tx` | 1 if amount exceeds the 99th percentile |
| `wallet_balance_change` | Net XRP flow for this account (negative = net sender) |
| `rolling_tx_count_5m` | How many transactions this account made in the last 5 minutes |
| `tx_per_minute` | Average transaction rate for this account |
| `tx_rate_z_score` | How abnormal this account's rate is vs. all accounts |
| `total_volume_5m` | Total XRP volume across all accounts in a 5-min window |
| `volume_spike_ratio` | Current 5-min volume vs. historical average (> 1 = spike) |
| `memo_entropy` | Shannon entropy of memo text (higher = more random/encrypted) |
| `memo_length` | Character length of the decoded memo |
| `duplicate_memo_count` | How many times this exact memo appears in the dataset |
| `contains_url` | 1 if the memo contains http / https / www |

### Step 3 — Anomaly Detection (`model.py`)

Trains an **Isolation Forest** (unsupervised):

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
5. Write any fired alerts to `data/alerts.json`
6. Upload scored transaction to Supabase
7. Append to `data/realtime_scored.csv`

---

## What Gets Flagged as Risky

The model detects transactions that deviate from normal patterns. In testing, the top anomalies were:

- **Burst senders** — accounts firing hundreds of micro-transactions per minute (spam pattern)
- **Unusual volume** — transactions during network-wide volume spikes
- **Abnormal rates** — accounts with tx rates far above the network average
- **Memo anomalies** — high entropy content, duplicate spam, or embedded URLs

Example: account `rpr6g53...` was consistently the riskiest — thousands of tiny payments in rapid bursts, a classic spam/wash-trading pattern.

---

## Whale Registry (`whale_registry.py`)

Identifies whale accounts from historical data before any alert logic runs.

- Ranks every account by total transaction count
- Classifies the top 5% (≥ 144 txs) as whales — **187 accounts** out of 3,686
- Why tx count, not XRP volume? The dataset is almost entirely token-based — native XRP payments are rare

```python
registry = WhaleRegistry.build()
registry.is_whale("rXXX...")       # True / False
registry.get_stats("rXXX...")      # AccountStats dataclass
registry.whale_accounts            # set of all whale addresses
```

---

## Transaction Buffer (`transaction_buffer.py`)

The alert engine's memory layer — a 10-minute sliding window of scored transactions.

- **Rolling window:** 10 minutes
- **Cooldown:** 2 minutes per alert type per account (prevents duplicate alerts)
- Exposes `get_account_state(account)` and `get_network_state()` to alert detectors

---

## Alert Engine (`alert_engine.py`)

15 alert types across 5 groups. Every scored transaction is checked against all of them.

### Group 1 — Transaction Size & Whale Movements

| Alert | Trigger | Example Message |
|-------|---------|-----------------|
| `LARGE_XRP_TRANSFER` | Whale sends Payment/OfferCreate above the 99th percentile | `🐋 Whale moved 2.4M SOLO → rDEST... (top 0.3% tx size)` |
| `VOLUME_SPIKE` | Network 5-min volume exceeds 3× historical average | `📈 Network volume spike: 8.2× above average` |
| `HIGH_RISK_TRANSACTION` | risk ≥ 0.8 and `is_anomaly = 1` | `⚠️ High-risk transaction detected — risk score 0.91` |

### Group 2 — Burst & Bot Activity

| Alert | Trigger | Example Message |
|-------|---------|-----------------|
| `TRANSACTION_BURST` | Whale sends ≥ 50 txs in the last 5 minutes | `⚡ Whale burst: 340 transactions in 5 min` |
| `ABNORMAL_TX_RATE` | Whale tx rate ≥ 3 standard deviations above network mean | `🤖 Abnormal tx rate: 120.0 tx/min — 8.3× above average` |

### Group 3 — DEX & Trading Activity

| Alert | Trigger | Example Message |
|-------|---------|-----------------|
| `TOKEN_ACCUMULATION` | Whale places ≥ 5 OfferCreates on same token in 10-min window | `🐋 Whale active on SOLO: 12 offers in last 10 min` |
| `OFFER_CANCEL_SPIKE` | Whale cancel ratio ≥ 70% with ≥ 5 total offers | `👻 Possible spoofing: 82% of offers cancelled` |
| `MULTI_WHALE_CONVERGENCE` | ≥ 3 distinct whales trade the same token in 10-min window | `🔥 3 whales trading SOLO simultaneously` |

### Group 4 — Memo & Spam Detection

| Alert | Trigger | Example Message |
|-------|---------|-----------------|
| `MEMO_SPAM` | `duplicate_memo_count ≥ 10` | `📨 Memo spam: 47 transactions with identical memo` |
| `URL_SPAM` | `contains_url = 1` + ≥ 5 URL txs from account in window | `🔗 URL spam: 7 transaction memos with URLs` |
| `HIGH_ENTROPY_MEMO` | `memo_entropy ≥ 4.5` and non-empty memo | `🔐 High-entropy memo — possible encrypted payload` |

### Group 5 — Behaviour Shifts

| Alert | Trigger | Example Message |
|-------|---------|-----------------|
| `WALLET_DRAIN` | Whale XRP out in window ≥ 80% of historical XRP sent | `⚠️ Wallet drain: 9400.0 XRP (87.3% of historical) in last 10 min` |
| `BEHAVIOUR_SHIFT` | Whale dominant tx type changed from historical baseline | `🔄 Behaviour shift: switched from Payment to OfferCreate` |
| `NEW_WHALE_EMERGENCE` | Non-whale crosses whale tx threshold in session | `🆕 New whale emerging: 145 txs in session` |

### Alert Output Format

```json
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
```

**Severity levels:** `low` → `medium` → `high` → `critical`
- `critical` — risk ≥ 0.9 and anomaly flag
- `high` — risk ≥ 0.8 or anomaly flag
- `medium` — risk ≥ 0.4
- `low` — everything else

---

## REST API Reference (`api.py`)

Base URL: `http://localhost:5000` — CORS enabled for all origins.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/alerts` | Recent alerts (`?limit=50&severity=high`) |
| GET | `/api/whales` | Whale list with stats (`?limit=100&sort=tx_count`) |
| GET | `/api/whales/<account>` | Single account stats |
| GET | `/api/network` | Live 10-min rolling window stats |
| GET | `/api/stats` | Registry summary + alert breakdown |
| GET | `/api/tokens` | Per-token whale activity heatmap |

### `GET /api/alerts`

```json
{
  "count": 3,
  "alerts": [ { ...alert object... } ]
}
```

### `GET /api/network`

```json
{
  "window_minutes": 10,
  "tx_count": 1240,
  "active_accounts": 87,
  "active_whale_count": 14,
  "active_whale_accounts": ["rWDwq...", "rHb9C..."],
  "total_xrp_volume": 842300.5,
  "avg_risk_score": 0.38,
  "anomaly_count": 62,
  "anomaly_rate_pct": 5.0,
  "alerts_last_10min": 7
}
```

### `GET /api/tokens`

```json
{
  "window_minutes": 10,
  "active_token_count": 4,
  "tokens": [
    { "token": "SOLO", "whale_count": 5, "whale_accounts": [...], "offer_count": 48 }
  ]
}
```

---

## Frontend Guide (Lovable)

### Setup

API base URL: `http://localhost:5000`. Poll every **5 seconds** for live data.

**Telegram subscribe button** — place as a fixed CTA at the bottom of every page:

```jsx
<a href="https://t.me/XRPL_Whale_WatchBot?start=1" target="_blank" rel="noopener noreferrer">
  <button>📲 Get Alerts on Telegram</button>
</a>
```

Style with Telegram blue (`#2AABEE`). When clicked, opens the bot and auto-subscribes the user — no typing required.

### Page Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  Header: "🐋 XRPL Whale Monitor"   [● LIVE]              [UTC time]  │
├────────────┬────────────┬────────────┬────────────┬──────────────────┤
│  Whales    │  Txs (10m) │  Anomaly % │  Alerts    │  Avg Risk Score  │
│  Active    │  in window │  rate      │  (10 min)  │                  │
├────────────┴────────────┴────────────┴────────────┴──────────────────┤
│                                                                        │
│   LEFT (60%)                        RIGHT (40%)                        │
│   ┌──────────────────────────┐      ┌──────────────────────────────┐  │
│   │  Live Alert Feed         │      │  Token Heatmap               │  │
│   │  (scrolling list)        │      │  (whale token activity)      │  │
│   └──────────────────────────┘      ├──────────────────────────────┤  │
│   ┌──────────────────────────┐      │  Whale Leaderboard           │  │
│   │  Alert Type Breakdown    │      │  (top 10 most active)        │  │
│   │  (bar chart)             │      └──────────────────────────────┘  │
│   └──────────────────────────┘                                         │
├──────────────────────────────────────────────────────────────────────┤
│  📲 Get Alerts on Telegram                               [Subscribe] │
└──────────────────────────────────────────────────────────────────────┘
```

### Visual Style

- **Theme:** Dark background (`#0d0f14`) — dark mode only
- **Severity colours:** critical → red (`#ef4444`), high → orange (`#f97316`), medium → yellow (`#eab308`), low → grey (`#64748b`)
- **Font:** Monospace for addresses and numbers, sans-serif for labels
- **Tone:** Data-dense but readable — Bloomberg terminal meets modern SaaS

### Components

**Stats Row** — poll `GET /api/network` every 5s

| Card | Field | Format |
|------|-------|--------|
| Whales Active | `active_whale_count` | Large number |
| Txs in Last 10 Min | `tx_count` | Number with commas |
| Anomaly Rate | `anomaly_rate_pct` | `X.X%` — red if > 10% |
| Alerts (10 min) | `alerts_last_10min` | Orange if > 0 |
| Avg Risk Score | `avg_risk_score` | `0.XX` — red if > 0.6 |

**Live Alert Feed** — poll `GET /api/alerts?limit=20` every 5s. Each card:
- Left border coloured by severity
- Alert type + severity badge + relative timestamp
- Plain-English message as the main content
- Truncated address (`rXXXX...XXXX`) and risk score
- Click → detail drawer showing full `details` object

**Token Heatmap** — poll `GET /api/tokens` every 5s. Each active token shown as a tile with whale count and offer activity. Sort by `whale_count` descending. Top token gets a gold highlight.

**Whale Leaderboard** — poll `GET /api/whales?limit=10&sort=tx_count` on load. Ranked table with account, tx count, dominant activity type, tokens traded, and percentile rank. Click a row → detail modal using `GET /api/whales/<account>`.

**Alert Type Breakdown** — poll `GET /api/stats` every 10s. Horizontal bar chart of `alert_type_breakdown`, sorted by count, coloured by typical severity.

### Data Freshness

- If fetch fails: show subtle `⚠ Data may be stale` in header — do not crash
- Empty alerts: `"No alerts yet — waiting for whale activity..."`
- Empty tokens: `"No whale token activity in the last 10 minutes"`

---

## Telegram Bot (`telegram_bot.py`)

Bot: **@XRPL_Whale_WatchBot**

Users subscribe via the frontend button. The bot sends alerts directly to their DMs based on their chosen severity threshold. All controls are inline keyboard buttons — no typing required.

**Inline keyboard buttons:**
- 📊 **My Status** — shows subscription and current filter
- ⚙️ **Set Severity** — choose Low / Medium / High / Critical
- 🔕 **Unsubscribe** — with confirmation step

**Environment variables** (set in `.env`):

```
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_MIN_SEVERITY=high
```

---

## Limitations

- **Anomaly ≠ fraud** — the model flags unusual patterns, not confirmed malicious activity
- **2.5 hours of training data** — more data would give the model a stronger baseline
- **Unsupervised only** — no labelled scam examples; a supervised model would be more precise
- **New whales** — the whale registry is built from historical data; brand new accounts won't be classified as whales until they cross the threshold via `NEW_WHALE_EMERGENCE`
- **Realtime features are approximate** — rolling window features use an in-memory buffer, not the full historical dataset

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data pipeline | Python 3.10+ · pandas · numpy |
| ML model | scikit-learn (Isolation Forest) · joblib |
| XRPL client | xrpl-py (websocket streaming) |
| Database | Supabase (scored transaction storage) |
| Alert engine | Python — 15 alert types across 5 groups |
| Alert storage | JSON file (`data/alerts.json`) |
| REST API | Flask · flask-cors |
| Notifications | Telegram Bot API · python-telegram-bot |
| Frontend | Lovable |

---

## Roadmap

- [x] Data normalisation pipeline
- [x] Feature engineering (12 features)
- [x] Isolation Forest model training
- [x] Realtime transaction scoring
- [x] Supabase integration
- [x] Whale registry (187 whales, top 5%)
- [x] Transaction buffer (10-min rolling window, cooldowns)
- [x] Alert engine (15 alert types across 5 groups)
- [x] Alert storage (`alerts_writer.py`)
- [x] Flask REST API (6 endpoints, CORS enabled)
- [x] Telegram bot (`@XRPL_Whale_WatchBot`)
- [ ] Lovable analytics dashboard
- [ ] User-configurable alert thresholds
- [ ] Cloud deployment
