-- Run this in your Supabase SQL Editor to create the scored_transactions table.
-- Supabase dashboard > SQL Editor > New Query > paste & run.

create table if not exists scored_transactions (
    id          bigint generated always as identity primary key,
    created_at  timestamptz default now(),

    -- Normalised transaction fields
    timestamp         text,
    ledger_index      bigint,
    tx_hash           text,
    tx_type           text,
    account           text,
    destination       text,
    fee               float8,
    amount_xrp        float8,
    currency          text,
    issuer            text,

    -- Engineered features
    tx_size_percentile    float8,
    is_large_tx           int,
    wallet_balance_change float8,
    total_volume_5m       float8,
    memo_entropy          float8,
    memo_length           int,
    duplicate_memo_count  int,
    contains_url          int,
    rolling_tx_count_5m   int,
    tx_per_minute         float8,
    tx_rate_z_score       float8,
    volume_spike_ratio    float8,

    -- Model output
    risk_score        float8,
    is_anomaly        int
);

-- Enable realtime so Lovable frontend can subscribe to new rows
alter publication supabase_realtime add table scored_transactions;

-- Allow anonymous reads (Lovable frontend uses the anon key)
alter table scored_transactions enable row level security;

create policy "Allow public read access"
    on scored_transactions for select
    using (true);

create policy "Allow insert from anon key"
    on scored_transactions for insert
    with check (true);
