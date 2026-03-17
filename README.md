# XRPL Whale Activity Monitor

> **XRP Commons Hackathon — Challenge 2: Analytics**
> Real-time detection of anomalous whale activity on the XRP Ledger, delivering actionable market insights to retail traders.

---

## Overview

Retail traders on the XRPL have no visibility into what large accounts (whales) are doing. This system bridges that gap by:

1. Training an Isolation Forest model on historical XRPL transaction data
2. Scoring every live transaction as it arrives from the XRPL websocket
3. Translating ML anomaly scores into named, plain-English alerts
4. Delivering those alerts to retail traders via a dashboard and Telegram bot

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
└──────────────┬──────────────────┘
               │  risk_score (0–1) + is_anomaly flag per transaction
               ▼
┌─────────────────────────────────┐  ← IN PROGRESS
│  alert_engine.py                │  Named alerts + plain-English messages
└──────────────┬──────────────────┘
               │  alerts.json
        ┌──────┴──────────────┐
        ▼                     ▼
  Flask API              Telegram Bot
  /api/alerts            push notifications
        │
        ▼
  Lovable Frontend
  (analytics dashboard)
```

---

## Pipeline

### Step 1 — Normalisation (`normalize.py`)

Converts both historical CSV data and live websocket messages into a unified 10-column schema:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime (UTC) | Ledger close time |
| `ledger_index` | int | Ledger sequence number |
| `tx_hash` | str | Unique transaction identifier |
| `tx_type` | str | Payment, OfferCreate, NFTokenBurn, etc. |
| `account` | str | Sender / initiator address |
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
| `rolling_tx_count_5m` | Transactions from this account in the last 5 minutes |
| `tx_per_minute` | Average transaction rate for this account |
| `tx_rate_z_score` | How abnormal this account's rate is vs. all accounts |
| `total_volume_5m` | Total XRP volume across all transactions in a 5-min window |
| `volume_spike_ratio` | Current 5-min volume vs. historical average (>1 = spike) |
| `memo_entropy` | Shannon entropy of memo text (higher = more random) |
| `memo_length` | Character length of decoded memo |
| `duplicate_memo_count` | How many times this exact memo appears in the dataset |
| `contains_url` | 1 if the memo contains http / https / www |

### Step 3 — Anomaly Detection (`model.py`)

Trains an **Isolation Forest** (unsupervised) on the 12 features:

- No labelled data required — learns "normal" from the data itself
- `contamination=0.05` — expects ~5% of transactions to be anomalous
- `n_estimators=200` — 200 trees for stable results
- Outputs `risk_score` (0–1, where 1 = most anomalous) and `is_anomaly` flag per transaction

### Step 4 — Realtime Scoring (`score_realtime.py`)

Connects to `wss://xrplcluster.com/` and for every live transaction:
1. Normalises it via `normalize.py`
2. Computes the 12 features using historical statistics as reference
3. Scores it with the trained Isolation Forest
4. Appends to `data/realtime_scored.csv` immediately

---

## Alert Engine *(in progress)*

The alert engine sits on top of the scored stream and translates `risk_score` + raw transaction fields into named, human-readable alerts for retail traders. Each alert combines a rule-based trigger with the ML anomaly score as a confidence signal.

### Whale Definition

An account is classified as a **whale** if it falls in the **top 5% by total transaction volume** in the dataset. This threshold is computed at the start of each pipeline run.

### Alert Types

#### Group 1 — Transaction Size & Whale Movements
*Driven by `amount_xrp`, `tx_size_percentile`, `is_large_tx`, `risk_score`*

| Alert | Example Message | Trigger |
|-------|----------------|---------|
| `LARGE_XRP_TRANSFER` | `🐋 Whale moved 2.4M XRP to rXXX...` | Single payment above 99th percentile from a whale account |
| `VOLUME_SPIKE` | `📈 XRP volume spiked 8× above average in the last 5 min` | `volume_spike_ratio > 3.0` |
| `HIGH_RISK_TRANSACTION` | `⚠️ Anomalous transaction detected — risk score 0.91` | `risk_score > 0.8` and `is_anomaly = 1` |

#### Group 2 — Burst & Bot Activity
*Driven by `rolling_tx_count_5m`, `tx_per_minute`, `tx_rate_z_score`*

| Alert | Example Message | Trigger |
|-------|----------------|---------|
| `TRANSACTION_BURST` | `⚡ Account rXXX... sent 340 txs in the last 5 min — possible bot` | `rolling_tx_count_5m` exceeds threshold for a whale account |
| `ABNORMAL_TX_RATE` | `🤖 Account rXXX... sending at 120 tx/min — 8× above network average` | `tx_rate_z_score > 3.0` |

#### Group 3 — DEX & Trading Activity
*Driven by `tx_type = OfferCreate / OfferCancel`, `taker_gets`, `taker_pays`*

| Alert | Example Message | Trigger |
|-------|----------------|---------|
| `TOKEN_ACCUMULATION` | `🐋 Whale accumulating SOLO — bought 3.2M units in the last 10 min` | Whale account placing large repeated buy offers on the same token |
| `FLASH_DUMP` | `🚨 Whale dumping RLUSD — 4.1M units sold in 5 min` | Large repeated sell offers from a whale account |
| `OFFER_CANCEL_SPIKE` | `👻 rXXX... placed and cancelled 400 offers — possible spoofing` | High OfferCancel / OfferCreate ratio for an account |
| `MULTI_WHALE_CONVERGENCE` | `🔥 3 whales trading SOLO simultaneously` | ≥3 whale accounts active on the same token within a short window |

#### Group 4 — Memo & Spam Detection
*Driven by `memo_entropy`, `memo_length`, `duplicate_memo_count`, `contains_url`*

| Alert | Example Message | Trigger |
|-------|----------------|---------|
| `MEMO_SPAM` | `📨 rXXX... sent 200 transactions with identical memo content` | `duplicate_memo_count > 50` for a single account |
| `URL_SPAM` | `🔗 Suspicious: account embedding URLs in transaction memos` | `contains_url = 1` combined with high transaction volume |
| `HIGH_ENTROPY_MEMO` | `🔐 Encrypted or binary payload detected in memo field` | `memo_entropy > 4.5` |

#### Group 5 — Account Behaviour Shifts
*Driven by `wallet_balance_change`, `tx_type` distribution over time*

| Alert | Example Message | Trigger |
|-------|----------------|---------|
| `WALLET_DRAIN` | `⚠️ rXXX... net outflow of 800K XRP in last 5 min — wallet draining` | `wallet_balance_change` strongly negative for a whale account |
| `BEHAVIOUR_SHIFT` | `🔄 Whale switched from trading to mass payments — unusual pattern` | Significant change in an account's transaction type distribution |
| `NEW_WHALE_EMERGENCE` | `🆕 New account crossed whale threshold — 500K XRP moved` | Account volume crosses the top 5% threshold for the first time |

### Alert Output Format

```json
{
  "id": "alert_1742123456_rXXX",
  "timestamp": "2026-03-17T14:00:00Z",
  "account": "rXXX...",
  "alert_type": "TOKEN_ACCUMULATION",
  "severity": "high",
  "risk_score": 0.87,
  "is_anomaly": 1,
  "message": "🐋 Whale accumulating SOLO — bought 3.2M units in the last 10 min",
  "details": {
    "token": "SOLO",
    "volume_bought": 3200000,
    "offer_count": 47,
    "window_minutes": 10
  }
}
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data pipeline | Python 3.10+ · pandas · numpy |
| ML model | scikit-learn (Isolation Forest) · joblib |
| XRPL client | xrpl-py (websocket streaming) |
| Alert engine | Python *(in progress)* |
| API | Flask *(planned)* |
| Notifications | Telegram Bot API *(planned)* |
| Frontend | Lovable *(planned)* |
| Data storage | CSV files |

---

## File Structure

```
├── normalize.py                 # Shared normalisation module
├── feature_engineering.py       # Compute 12 ML features from raw CSV
├── model.py                     # Train Isolation Forest + score transactions
├── score_realtime.py            # Stream + score live XRPL transactions
├── demo.py                      # Colour-coded terminal demo
├── xrpl_historical.py           # Load historical CSV → normalised CSV
├── xrpl_realtime.py             # Stream live transactions → raw CSV
├── alert_engine.py              # ← IN PROGRESS: named alerts from scored stream
├── api.py                       # ← PLANNED: Flask REST API
├── telegram_bot.py              # ← PLANNED: Telegram alert bot
├── requirements.txt
├── models/
│   └── isolation_forest.pkl     # Trained model
├── data/                        # Generated outputs (not in git)
│   ├── featured_transactions.csv
│   ├── scored_transactions.csv
│   └── realtime_scored.csv
└── ie_dataset/                  # Source data (not in git)
    ├── transactions.csv
    └── ledger.csv
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Internet connection (for the XRPL websocket)

### Install

```bash
pip install -r requirements.txt
```

### Add the dataset

The `ie_dataset/` folder is not in git. Get `ie_dataset.zip` from your team and unzip it in the repo root:

```bash
unzip ie_dataset.zip
```

### Train the model

```bash
python3 feature_engineering.py && python3 model.py
```

Takes ~30 seconds. Outputs `data/featured_transactions.csv`, `data/scored_transactions.csv`, and `models/isolation_forest.pkl`.

### Run the live demo

```bash
python3 demo.py
```

Streams live XRPL transactions with colour-coded risk scores. Press `Ctrl+C` to stop.

---

## Environment Variables

```bash
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
FLASK_PORT=4000
```

---

## Limitations

- **Anomaly ≠ fraud** — the model flags unusual patterns, not confirmed malicious activity. A legitimate whale transaction may be flagged.
- **2.5 hours of training data** — the historical dataset covers ~2.5 hours of XRPL activity. More data would improve the model's baseline.
- **Unsupervised** — no labelled examples of known scam transactions. A supervised model would be more precise given labelled data.
- **Realtime features are approximate** — rolling window features in realtime are computed from a 5-minute in-memory buffer, not the full historical dataset.

---

## Roadmap

- [x] Data normalisation pipeline
- [x] Feature engineering (12 features)
- [x] Isolation Forest model training
- [x] Realtime transaction scoring
- [ ] Alert engine (15 alert types)
- [ ] Flask REST API
- [ ] Telegram bot notifications
- [ ] Lovable analytics dashboard
- [ ] Cloud deployment
- [ ] User-configurable alert thresholds
