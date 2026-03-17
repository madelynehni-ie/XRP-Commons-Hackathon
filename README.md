# XRPL Risk Monitor (XRP-Commons-Hackathon)

A beginner-friendly XRPL dApp for monitoring token risk using live and historical ledger data.

## Main fraud signals
- Whale movements / large transactions
- Memo spam detection
- Transaction burst detection

## Simple pipeline
XRPL WebSocket (real-time data) + historical backfill (1 year) -> ML feature selection & engineering, predictions  -> risk scoring -> dashboard

## Tech stack
- Frontend: React + TypeScript
- Backend: Node.js + TypeScript
- Database: Postgres
- ML: Python or JavaScript

## Getting started
1. Copy `.env.example` to `.env`
2. Start Postgres
3. Run backend
4. Run frontend

## Notes
