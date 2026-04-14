#!/usr/bin/env python3
"""
nexus_mock_server.py — NeuroForge Nexus Mock API Server
========================================================
Simulates the Nexus BCI backend for local integration testing.
Runs Flask on port 8080.

Usage:
    python nexus_mock_server.py
    python nexus_mock_server.py --port 8080 --host 0.0.0.0
"""

import argparse
import json
import time
import uuid
import threading
from datetime import datetime

try:
    from flask import Flask, request, jsonify
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    print("Flask not installed. Run: pip install flask")
    import sys; sys.exit(1)

app = Flask("neuroforge-nexus-mock")

# ── In-memory state ───────────────────────────────────────────────────────────
state = {
    "sessions": {},
    "pi_payloads": [],
    "triggers": [],
    "requests_total": 0,
    "started_at": time.time(),
}


def log(msg: str):
    ts = datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]
    print(f"  [{ts}] {msg}")


# ── Health ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "server": "neuroforge-nexus-mock", "version": "1.0.0"})


# ── Pi receive ────────────────────────────────────────────────────────────────
@app.route("/nexus/receive", methods=["POST"])
def nexus_receive():
    state["requests_total"] += 1
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id", str(uuid.uuid4()))

    log(f"📡 Pi payload received | session={session_id[:16]}... | n_digits={data.get('n_digits', '?')}")

    record = {
        "received_at": time.time(),
        "session_id": session_id,
        "pi_digest": data.get("pi_digest", ""),
        "n_digits": data.get("n_digits", 0),
        "neural_load": data.get("neural_load", 0.0),
        "checksum": data.get("checksum", ""),
    }
    state["pi_payloads"].append(record)
    state["sessions"][session_id] = record

    return jsonify({
        "status": "ok",
        "message": "Pi payload received and integrated into BCI matrix",
        "session_id": session_id,
        "nexus_node": "nexus-mock-primary",
        "timestamp": time.time(),
        "neural_throughput_gbps": 1.21,
    }), 200


# ── Nexus test endpoint ───────────────────────────────────────────────────────
@app.route("/nexus/test", methods=["POST"])
def nexus_test():
    state["requests_total"] += 1
    data = request.get_json(force=True) or {}
    pi_digits = data.get("pi_digits", "")
    neural_load = data.get("neural_load", 0.0)

    log(f"🧪 Test endpoint | pi_digits={str(pi_digits)[:20]} | neural_load={neural_load}")

    return jsonify({
        "status": "ok",
        "message": "BCI bridge validated",
        "pi_received": str(pi_digits)[:10] + "..." if len(str(pi_digits)) > 10 else str(pi_digits),
        "neural_load": neural_load,
        "bci_status": "ACTIVE",
        "latency_ms": 0.42,
    }), 200


# ── Trigger ───────────────────────────────────────────────────────────────────
@app.route("/nexus/trigger", methods=["POST"])
def nexus_trigger():
    state["requests_total"] += 1
    data = request.get_json(force=True) or {}
    trigger_id = str(uuid.uuid4())[:8]

    log(f"⚡ Nexus workflow triggered | trigger_id={trigger_id}")
    state["triggers"].append({"trigger_id": trigger_id, "ts": time.time(), "data": data})

    return jsonify({
        "status": "ok",
        "trigger_id": trigger_id,
        "workflow": "pi-nexus-sync",
        "message": "Nexus workflow initiated — Pi-BCI fusion in progress",
        "actions_queued": ["validate_pi", "sync_bci_matrix", "update_nexus_topology"],
    }), 200


# ── Stats ─────────────────────────────────────────────────────────────────────
@app.route("/nexus/stats", methods=["GET"])
def nexus_stats():
    uptime = time.time() - state["started_at"]
    return jsonify({
        "uptime_s": round(uptime, 1),
        "requests_total": state["requests_total"],
        "pi_payloads_received": len(state["pi_payloads"]),
        "triggers_fired": len(state["triggers"]),
        "active_sessions": len(state["sessions"]),
    })


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NeuroForge Nexus Mock Server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"\n  🧠 NeuroForge Nexus Mock Server")
    print(f"  ═══════════════════════════════")
    print(f"  Listening on http://{args.host}:{args.port}")
    print(f"  Endpoints:")
    print(f"    GET  /health")
    print(f"    POST /nexus/receive   ← Pi BCI payload")
    print(f"    POST /nexus/test      ← Mock BCI test")
    print(f"    POST /nexus/trigger   ← Workflow trigger")
    print(f"    GET  /nexus/stats")
    print(f"  Press Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
