# Flowlens Backend

> Real-time XRPL anomaly detection engine — three-layer detection pipeline with Isolation Forest, robust z-scores, and rule-based thresholds.

---

## Architecture

```
XRPL Mainnet (wss://xrplcluster.com)
        │  validated ledgers + transactions
        ▼
┌─────────────────────────────────────────────┐
│  Ingest  (src/ingest/ingest.js)             │
│  • Parse & normalise tx                     │
│  • Filter to: Payment, TrustSet,            │
│    OfferCreate, OfferCancel                 │
│  • Update per-account state snapshots       │
│  • Update network aggregates                │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Feature Engine  (src/features/)            │
│  Computes 10 features per account-window:   │
│                                             │
│  F1  txCount5       burst (5-ledger)        │
│  F2  txCount20      burst (20-ledger)       │
│  F3  dormancyGap    ledgers since last seen │
│  F4  offerRatio     cancel/(create+cancel)  │
│  F5  destHHI        destination HHI         │
│  F6  memoEntropy    Shannon entropy of memo │
│  F7  failedRatio    failed tx fraction      │
│  F8  reserveDelta   owner-count delta       │
│  F9  mixShift       tx-type cosine shift    │
│  F10 burstZ         MAD z-score of tx rate  │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Detection  (src/detection/)                │
│                                             │
│  Layer 1 — Rule thresholds (instant)        │
│  Layer 2 — Robust z-score (MAD baseline)    │
│  Layer 3 — Isolation Forest (multi-dim)     │
│                                             │
│  → weighted risk score + plain-English      │
│    explanation per finding                  │
└──────────────────┬──────────────────────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
    REST API           WebSocket
    /api/*             ws://…
    (Express)          (broadcast)
```

---

## Detected Anomaly Types

### 🐋 Whale Movements / Large Transactions

A whale event is flagged when a large transfer combines with behavioural indicators — size alone is not enough.

**How it works:**

The feature engine tracks every account's transfer history. When a Payment lands, the engine computes a MAD-based z-score (`burstZ`, F10) of the account's per-ledger transaction rate and cross-references it against the destination concentration score (`destHHI`, F5). A single large transfer from a normally quiet account scores higher than a routinely active one of the same size.

**Signals used:**

| Signal | Feature | Why it matters |
|--------|---------|---------------|
| Transfer size vs network baseline | `burstZ` (F10) | Large relative to recent history — not just an absolute threshold |
| Destination spread | `destHHI` (F5) | Whale dispersing to many addresses scores higher than a single peer transfer |
| Dormant reactivation | `dormancyGap` (F3) | A long-silent large account suddenly moving funds is higher risk than an active one |
| Behaviour shift | `mixShift` (F9) | A whale that was only doing OfferCreate suddenly sending Payments is suspicious |

**Severity thresholds** (configurable in `.env`):

```
BURST_20_LEDGERS=40      → sustained high volume
ZSCORE_WARN=3.0          → notable size deviation
ZSCORE_CRITICAL=5.0      → extreme size deviation
DORMANT_LEDGER_GAP=5000  → ~7 days of silence before reactivation flag
```

**Example anomaly output:**
```json
{
  "severity": "high",
  "findings": [
    { "feature": "burstZ", "zScore": "4.2", "explanation": "burstZ has robust z-score 4.2 — extreme deviation from network baseline." },
    { "feature": "dormancyGap", "value": 6100, "explanation": "Account dormant for 6,100 ledgers (~8.7 days), suddenly active." }
  ]
}
```

---

### 📨 Memo Spam Detection

Memo fields on XRPL transactions are optional but frequently abused — for spam campaigns, encoded data exfiltration, or bulk templated messages.

**How it works:**

For every transaction containing a `Memos` field, the engine decodes the hex `MemoData`, computes its **Shannon entropy** (F6), and stores it in a rolling sample for that account.

Shannon entropy measures information density:
- **Low entropy (< 0.5 bits/char)** — highly repetitive content. The same character or short pattern repeated. Classic spam: `AAAAAA`, `buy now buy now`.
- **High entropy (> 4.5 bits/char)** — near-random byte distribution. Consistent with encrypted payloads, binary data, or encoded exfiltration.
- **Normal range (0.5–4.5)** — ordinary human-readable text or structured data.

The z-score layer additionally compares an account's current memo entropy against the **network baseline** — an account consistently sending unusual memos will accumulate a high MAD z-score even if individual values look borderline.

**Signals used:**

| Signal | Feature | Threshold | Indicates |
|--------|---------|-----------|-----------|
| Shannon entropy | `memoEntropy` (F6) | < 0.5 | Repetitive spam content |
| Shannon entropy | `memoEntropy` (F6) | > 4.5 | Encrypted / binary payload |
| Network z-score | MAD z-score of F6 | > 3.0 | Outlier vs all accounts currently on-ledger |

**Configurable in `.env`:**
```
MEMO_ENTROPY_LOW=0.5    # below this → spam
MEMO_ENTROPY_HIGH=4.5   # above this → encoded payload
```

**Example anomaly output:**
```json
{
  "severity": "medium",
  "findings": [
    {
      "feature": "memoEntropy",
      "value": 0.21,
      "explanation": "Memo entropy 0.21 is very low — repetitive or templated memo content (possible spam)."
    }
  ]
}
```

---

### ⚡ Transaction Burst Detection

Burst detection identifies accounts sending an abnormally high number of transactions in a short ledger window. This covers spam bots, automated scripts misfiring, and deliberate network congestion attacks.

**How it works:**

Two complementary burst features are computed on every incoming transaction:

**F1 — `txCount5` (short window):** counts all transactions from this account in the last 5 ledgers (~20 seconds at normal cadence). Catches rapid-fire bursts that start and end quickly.

**F2 — `txCount20` (medium window):** counts across the last 20 ledgers (~80 seconds). Catches sustained automated activity that might stay just below the short-window threshold.

**F10 — `burstZ` (statistical):** computes a MAD-based z-score of the account's own per-ledger transaction rate over its full history window. This catches accounts that burst *relative to their own baseline* — a market maker sending 10 tx/ledger is normal; a dormant retail wallet doing the same is anomalous.

All three are fed into the Isolation Forest as part of the feature vector, so burst combined with unusual destination spread or a high failed-tx ratio scores significantly higher than burst alone.

**Signals used:**

| Signal | Feature | Threshold | Indicates |
|--------|---------|-----------|-----------|
| Short burst | `txCount5` (F1) | > 15 tx in 5 ledgers | Rapid-fire bot or spam |
| Sustained burst | `txCount20` (F2) | > 40 tx in 20 ledgers | Prolonged automated activity |
| Statistical burst | `burstZ` (F10) | MAD z > 3.0 | Unusual vs account's own history |
| Failed burst | `failedRatio` (F7) | > 25% failed | Probing or misconfigured script |

**Configurable in `.env`:**
```
BURST_5_LEDGERS=15     # short window threshold
BURST_20_LEDGERS=40    # medium window threshold
ZSCORE_WARN=3.0        # statistical burst sensitivity
FAILED_TX_RATIO=0.25   # failed tx fraction flag
```

**Example anomaly output:**
```json
{
  "severity": "critical",
  "findings": [
    { "feature": "txCount5",  "value": 23, "explanation": "23 transactions in last 5 ledgers (threshold: 15) — rapid burst." },
    { "feature": "txCount20", "value": 51, "explanation": "51 transactions in last 20 ledgers (threshold: 40) — sustained burst." },
    { "feature": "burstZ",    "zScore": "5.1", "explanation": "burstZ has robust z-score 5.1 — extreme deviation from network baseline." }
  ]
}
```

---

## All Features

| # | Feature | Signal | Detects |
|---|---------|--------|---------|
| F1 | `txCount5` | Burst in last 5 ledgers | Rapid-fire spam, DDoS-style bursts |
| F2 | `txCount20` | Sustained burst over 20 ledgers | Prolonged automated activity |
| F3 | `dormancyGap` | Ledgers silent before now | Dormant wallet reactivation, whale wakeups |
| F4 | `offerRatio` | Cancel/(create+cancel) | Spoofing, wash trading, order book manipulation |
| F5 | `destHHI` | Destination concentration (HHI) | Fan-out attacks, money mule patterns, circular flow |
| F6 | `memoEntropy` | Shannon entropy of memo field | Spam (low entropy), encrypted payload (high entropy) |
| F7 | `failedRatio` | Failed tx / total tx | Probing, brute force, misconfigured bots |
| F8 | `reserveDelta` | Owner count change over window | Account draining, object spam, reserve manipulation |
| F9 | `mixShift` | Cosine distance of tx-type distribution | Behavioural fingerprint change |
| F10 | `burstZ` | MAD z-score of per-ledger tx count | Statistical burst relative to account's own baseline |

---

## Detection Layers

### Layer 1 — Rule-Based Thresholds
Fast, interpretable, zero training required. Each rule maps directly to one feature and produces a plain-English explanation.

### Layer 2 — Robust Z-Score (MAD)
Uses **Median Absolute Deviation** instead of standard deviation — resistant to outliers poisoning the baseline. Compares each feature value against a rolling 500-sample network baseline.

### Layer 3 — Isolation Forest
Pure-JS implementation (no external ML deps). Trained on the last N account-ledger feature vectors. Anomalous accounts are isolated in fewer random splits.

- Retrained every `IF_RETRAIN_EVERY` new samples (default 200)
- Score > 0.62 = anomalous
- Catches multi-dimensional combinations that rules and z-scores miss individually

All three layers contribute to a **weighted risk score** (0–100) with severity labels: `low / medium / high / critical`.

---

## Setup

### Requirements
- Node.js 18+
- npm

### Install

```bash
cd flowlens-backend
npm install
```

### Configure

```bash
cp .env.example .env
# Edit .env — defaults work out of the box
```

### Run

```bash
# Live XRPL data
node src/server.js

# Mock mode (no internet needed)
USE_MOCK=true node src/server.js
```

---

## API Reference

### REST

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/status` | KPIs + XRPL connection status |
| GET | `/api/ledger` | Latest validated ledger info |
| GET | `/api/transactions` | Recent transactions (filter: `type`, `account`, `success`, `limit`) |
| GET | `/api/anomalies` | Anomaly log (filter: `severity`, `account`, `layer`, `limit`) |
| GET | `/api/anomalies/summary` | Counts by type, severity, detection layer |
| GET | `/api/anomalies/:id` | Single anomaly with full feature vector |
| GET | `/api/accounts` | Top accounts by tx count |
| GET | `/api/accounts/:address` | Account profile + recent anomalies |
| GET | `/api/accounts/:address/features` | Latest computed feature vector |
| GET | `/api/volume` | XRP volume per ledger (last 100) |
| GET | `/api/tx-types` | Transaction type breakdown |

### WebSocket

Connect to `ws://localhost:4000`. On connect you receive a `snapshot` message with current state.

**Event types broadcast:**

| Event | Payload |
|-------|---------|
| `snapshot` | `{ transactions, anomalies, kpis, latestLedger }` |
| `tx` | Normalised transaction |
| `anomaly` | Full anomaly object with findings array |
| `ledger` | `{ index, hash, closeTime, txCount }` |
| `status` | `{ connected, node }` |

### Anomaly Object Schema

```json
{
  "id": "anom_1234_abc",
  "account": "rXXX...",
  "ledger": 80012345,
  "ts": "2026-03-17T13:00:00.000Z",
  "severity": "high",
  "score": 74,
  "confidence": 86,
  "detectionLayers": {
    "rules": true,
    "zscore": true,
    "iforest": false
  },
  "findings": [
    {
      "feature": "offerRatio",
      "value": "0.912",
      "threshold": 0.8,
      "explanation": "Offer cancel ratio 91.2% (threshold: 80%) — possible spoofing or wash activity.",
      "weight": 0.28
    }
  ],
  "summary": "Offer cancel ratio 91.2% ...",
  "features": { "txCount5": 3, "offerRatio": 0.912 }
}
```

---

## Project Structure

```
flowlens-backend/
├── src/
│   ├── server.js              # Entry point
│   ├── ingest/
│   │   ├── ingest.js          # Parse + normalise XRPL messages
│   │   ├── xrplStream.js      # XRPL WebSocket connector
│   │   └── mockStream.js      # Synthetic traffic generator
│   ├── features/
│   │   └── featureEngine.js   # Compute all 10 features
│   ├── detection/
│   │   ├── detector.js        # Rule + z-score + IF pipeline
│   │   └── isolationForest.js # Pure-JS Isolation Forest
│   ├── store/
│   │   └── store.js           # In-memory state
│   └── api/
│       ├── app.js             # Express setup
│       ├── ws.js              # WebSocket broadcast
│       └── routes.js          # All REST endpoints
├── .env.example
├── package.json
└── README.md
```

---

## Environment Variables

See `.env.example` for all variables and their defaults. Key ones:

```bash
USE_MOCK=true               # offline dev — no XRPL needed
BURST_5_LEDGERS=15          # rule threshold
BURST_20_LEDGERS=40         # rule threshold
OFFER_CANCEL_RATIO=0.80     # spoofing signal
MEMO_ENTROPY_LOW=0.5        # spam detection lower bound
MEMO_ENTROPY_HIGH=4.5       # encoded payload upper bound
DORMANT_LEDGER_GAP=5000     # ~7 days silence = dormant whale
ZSCORE_WARN=3.0             # MAD z-score warning
ZSCORE_CRITICAL=5.0         # MAD z-score critical
IF_RETRAIN_EVERY=200        # how often to retrain Isolation Forest
```

---

## Connecting to the Frontend

The frontend connects to:
- **REST** `http://localhost:4000/api/...`
- **WebSocket** `ws://localhost:4000`

CORS is open by default. Set `CORS_ORIGIN=http://localhost:3000` in `.env` to restrict it.
