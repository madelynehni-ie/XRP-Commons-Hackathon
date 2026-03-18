# XRPL Risk Monitor

A real-time anomaly detection system for the XRP Ledger. It trains an Isolation Forest model on historical transaction data and scores live transactions as they arrive from the XRPL websocket, flagging suspicious activity.

Built for the XRP Commons Hackathon.

---

## What it does

1. Loads ~272,000 historical XRPL transactions from CSV files
2. Normalises them into a clean 10-column schema
3. Engineers 12 features per transaction (size, velocity, memo patterns, volume spikes)
4. Trains an Isolation Forest (unsupervised anomaly detection) on the historical data
5. Connects to the XRPL websocket and scores every live transaction in real time
6. Flags anomalous transactions (top ~5%) and displays a colour-coded risk feed

---

## Project structure

```
XRP-Commons-Hackathon/
|
|-- ie_dataset/                     # Historical data (not in git, see setup)
|   |-- ledger.csv                  #   2,373 ledger records
|   |-- transactions.csv            #   271,990 transaction records
|
|-- data/                           # Generated outputs (not in git)
|   |-- historical_transactions.csv #   Normalised historical data
|   |-- featured_transactions.csv   #   Historical data + 12 features
|   |-- scored_transactions.csv     #   Historical data + risk scores
|   |-- realtime_transactions.csv   #   Raw normalised realtime stream
|   |-- realtime_scored.csv         #   Realtime data + risk scores
|
|-- models/                         # Saved models (not in git)
|   |-- isolation_forest.pkl        #   Trained Isolation Forest model
|
|-- normalize.py                    # Shared normalisation module
|-- feature_engineering.py          # Computes 12 ML features per transaction
|-- model.py                        # Trains and scores with Isolation Forest
|-- xrpl_historical.py              # Loads historical CSVs -> normalised CSV
|-- xrpl_realtime.py                # Streams live transactions -> CSV
|-- score_realtime.py               # Streams + scores live transactions -> CSV
|-- demo.py                         # Colour-coded terminal demo (temporary)
|-- requirements.txt                # Python dependencies
```

---

## Setup

### Prerequisites

- Python 3.10+
- Internet connection (for the XRPL websocket)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add the historical dataset

The `ie_dataset/` folder is not in git (too large). Get `ie_dataset.zip` from your team, place it in the repo root, and unzip:

```bash
unzip ie_dataset.zip
```

You should now have `ie_dataset/ledger.csv` and `ie_dataset/transactions.csv`.

### 3. Train the model

```bash
python3 feature_engineering.py && python3 model.py
```

This takes about 30 seconds on a laptop. It:
- Reads `ie_dataset/transactions.csv` (raw historical data)
- Normalises it and computes 12 features per transaction
- Saves `data/featured_transactions.csv`
- Trains an Isolation Forest (unsupervised, no labels needed)
- Saves the model to `models/isolation_forest.pkl`
- Saves scored historical data to `data/scored_transactions.csv`

### 4. Run the live demo

```bash
python3 demo.py
```

Leave it running. Transactions appear in real time with colour-coded risk scores. Press Ctrl+C to stop.

---

## How the pipeline works

### Step 1: Normalisation (`normalize.py`)

Both historical (CSV) and realtime (websocket) data arrive in different formats:
- CSV uses snake_case keys, JSON-encoded amounts, hex-encoded memos
- Websocket uses PascalCase keys, nested dicts, different field names

`normalize.py` converts both into the same flat 10-column schema:

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime (UTC) | When the ledger closed |
| ledger_index | int | Ledger sequence number |
| tx_hash | str | Unique transaction identifier |
| tx_type | str | Payment, OfferCreate, NFTokenBurn, etc. |
| account | str | Sender/initiator address |
| destination | str or NaN | Receiver address (Payments only) |
| fee | int | Transaction fee in drops (1 XRP = 1,000,000 drops) |
| amount_xrp | float or NaN | Transaction value in XRP |
| currency | str | "XRP" for native, token code otherwise |
| issuer | str or NaN | Token issuer address (NaN for XRP) |

### Step 2: Feature engineering (`feature_engineering.py`)

Computes 12 features per transaction from the normalised data:

**Transaction size features:**
- `tx_size_percentile` - Where this amount ranks among all transactions (0-1)
- `is_large_tx` - Binary flag: 1 if amount exceeds 99th percentile

**Account behaviour features:**
- `wallet_balance_change` - Net XRP flow for this account (negative = net sender)
- `rolling_tx_count_5m` - How many transactions this account made in the last 5 minutes
- `tx_per_minute` - Average transaction rate for this account
- `tx_rate_z_score` - How abnormal this account's rate is vs. all accounts

**Volume features:**
- `total_volume_5m` - Total XRP volume across all transactions in a 5-minute window
- `volume_spike_ratio` - Current 5-min volume vs. historical average (>1 = spike)

**Memo features:**
- `memo_entropy` - Shannon entropy of the memo text (higher = more random)
- `memo_length` - Character length of the decoded memo
- `duplicate_memo_count` - How many times this exact memo appears in the dataset
- `contains_url` - Binary flag: 1 if the memo contains http/https/www

### Step 3: Model training (`model.py`)

Uses an **Isolation Forest** — an unsupervised anomaly detection algorithm. It works by:
1. Building random decision trees that isolate data points
2. Anomalies are easier to isolate (fewer splits needed)
3. Points that are isolated quickly get higher anomaly scores

Configuration:
- `contamination=0.05` — expects ~5% of transactions to be anomalous
- `n_estimators=200` — number of trees (more = more stable)
- No labels needed — it learns "normal" from the data itself

Output per transaction:
- `risk_score` — float from 0 to 1 (1 = highest risk)
- `is_anomaly` — binary flag (1 = flagged as anomalous)

### Step 4: Realtime scoring (`score_realtime.py` / `demo.py`)

Connects to `wss://xrplcluster.com/` via websocket and for each live transaction:
1. Normalises it using `normalize.py`
2. Computes the 12 features using historical statistics as reference (e.g. the 99th percentile from training data) and a rolling 5-minute buffer for time-window features
3. Runs it through the trained Isolation Forest
4. Outputs the risk score and anomaly flag
5. Appends to `data/realtime_scored.csv` immediately (so other systems can read it live)

---

## What gets flagged as risky

The model detects transactions that deviate from normal patterns. In testing, the top anomalies were:

- **Burst senders** — accounts firing hundreds of micro-transactions per minute (spam pattern)
- **Unusual volume** — transactions during volume spikes or lulls
- **Abnormal rates** — accounts with tx rates far above the network average
- **Memo anomalies** — unusual memo content, high entropy, or duplicate memo spam

Example: account `rpr6g53...` was consistently flagged as the riskiest — it sent thousands of tiny payments in rapid bursts, a classic spam/wash-trading pattern.

---

## File-by-file reference

| File | What it does | When to run it |
|------|-------------|----------------|
| `normalize.py` | Shared module — converts raw data to unified schema | Imported by other files, never run directly |
| `xrpl_historical.py` | Loads CSVs and saves normalised data | `python3 xrpl_historical.py` — only if you need the intermediate normalised CSV |
| `xrpl_realtime.py` | Streams live transactions to CSV (no scoring) | `python3 xrpl_realtime.py` — only if you want raw realtime data without scoring |
| `feature_engineering.py` | Computes 12 features from raw historical data | `python3 feature_engineering.py` — run before training the model |
| `model.py` | Trains Isolation Forest, scores historical data | `python3 model.py` — run after feature engineering |
| `score_realtime.py` | Streams + scores live transactions | `python3 score_realtime.py` — run for plain-text realtime scoring |
| `demo.py` | Colour-coded terminal demo | `python3 demo.py` — run for the demo presentation |

---

## Quick start (copy-paste)

```bash
# Install
pip install -r requirements.txt

# Unzip data (if not already done)
unzip ie_dataset.zip

# Train the model (~30 seconds)
python3 feature_engineering.py && python3 model.py

# Run the demo
python3 demo.py
```

---

## Limitations

- **Anomaly is not fraud** — the model flags unusual patterns, not confirmed malicious activity. A legitimate whale transaction may be flagged. This is a starting point for investigation, not a verdict.
- **2.5 hours of training data** — the historical dataset covers ~2.5 hours of XRPL activity. More data would improve the model's sense of "normal".
- **No labelled data** — this is fully unsupervised. With labelled examples of known scam transactions, a supervised model would be more precise.
- **Realtime features are approximate** — rolling window features in realtime are computed from a 5-minute buffer of transactions seen during the session, not the full historical context.

---

## Tech stack

- **Python 3.10+**
- **xrpl-py** — XRPL websocket client
- **pandas / numpy** — data processing
- **scikit-learn** — Isolation Forest model
- **joblib** — model serialisation
