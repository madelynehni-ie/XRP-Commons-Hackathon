"""
api.py — Flask REST API for the XRPL Whale Alert System

Endpoints:
    GET /api/alerts          — recent alerts (supports ?limit=N&severity=high)
    GET /api/whales          — whale account list with stats
    GET /api/stats           — live network summary from the transaction buffer
    GET /api/health          — simple health check

Usage:
    python3 api.py

The API reads from data/alerts.json (written by alerts_writer.py / score_realtime.py).
It does NOT require score_realtime.py to be running — it serves whatever is in the file.

Dependencies:
    pip install flask flask-cors
"""

import os
from flask import Flask, jsonify, request
from flask_cors import CORS

from alerts_writer import read_alerts
from whale_registry import WhaleRegistry
from transaction_buffer import TransactionBuffer

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)   # allow all origins so Lovable frontend can call freely

# Build whale registry once at startup (read-only, ~5 seconds)
print("Building whale registry...")
_registry = WhaleRegistry.build(verbose=False)
_buffer = TransactionBuffer()   # shared buffer (empty at startup; score_realtime populates it)
print(f"  {len(_registry.whale_accounts)} whales loaded\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _whale_stats(account: str) -> dict:
    stats = _registry.get_stats(account)
    return {
        "account": account,
        "tx_count": stats.tx_count,
        "xrp_sent": round(stats.xrp_sent, 2),
        "xrp_received": round(stats.xrp_received, 2),
        "unique_destinations": stats.unique_destinations,
        "percentile": round(_registry.percentile(account), 4),
        "tx_types": dict(stats.tx_types),
        "tokens_traded": list(stats.tokens_traded),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "whales": len(_registry.whale_accounts)})


@app.get("/api/alerts")
def get_alerts():
    """Return recent alerts.

    Query params:
        limit    int   max number of alerts to return (default 50, max 500)
        severity str   filter by severity: low | medium | high | critical
    """
    limit = min(int(request.args.get("limit", 50)), 500)
    severity = request.args.get("severity", None)

    alerts = read_alerts(limit=limit, severity=severity)
    return jsonify({
        "count": len(alerts),
        "alerts": alerts,
    })


@app.get("/api/whales")
def get_whales():
    """Return the full whale account list with per-account stats.

    Query params:
        limit    int   max number of whales to return (default 100)
        sort     str   sort field: tx_count | xrp_sent | percentile (default: tx_count)
    """
    limit = min(int(request.args.get("limit", 100)), 500)
    sort_by = request.args.get("sort", "tx_count")
    valid_sorts = {"tx_count", "xrp_sent", "percentile"}
    if sort_by not in valid_sorts:
        sort_by = "tx_count"

    whales = [_whale_stats(acc) for acc in _registry.whale_accounts]
    whales.sort(key=lambda w: w.get(sort_by, 0), reverse=True)
    whales = whales[:limit]

    return jsonify({
        "total_whales": len(_registry.whale_accounts),
        "whale_tx_threshold": _registry.whale_tx_threshold,
        "returned": len(whales),
        "whales": whales,
    })


@app.get("/api/whales/<account>")
def get_whale(account: str):
    """Return stats for a single account."""
    if not _registry.is_whale(account):
        stats = _registry.get_stats(account)
        return jsonify({
            "account": account,
            "is_whale": False,
            "tx_count": stats.tx_count,
            "percentile": round(_registry.percentile(account), 4),
        }), 200

    return jsonify({"is_whale": True, **_whale_stats(account)})


@app.get("/api/stats")
def get_stats():
    """Return registry-level network summary."""
    summary = _registry.summary()
    recent_alerts = read_alerts(limit=500)
    alert_counts = {}
    for a in recent_alerts:
        t = a.get("alert_type", "UNKNOWN")
        alert_counts[t] = alert_counts.get(t, 0) + 1

    return jsonify({
        "total_accounts": summary["total_accounts"],
        "whale_count": summary["whale_count"],
        "whale_threshold_tx_count": summary["whale_tx_threshold"],
        "total_alerts_stored": len(recent_alerts),
        "alert_type_breakdown": alert_counts,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting XRPL Whale Alert API on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
