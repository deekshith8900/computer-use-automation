"""
operator/server.py — Human-in-the-loop operator dashboard.

A minimal Flask server that:
  - Lists pending intervention requests
  - Shows context: current step, screenshot, reason, URL
  - Lets the operator resolve (resume automation) with optional notes
  - Connects to the same intervention files the automation reads/writes

The browser session remains LIVE and PAUSED on the automation side.
The operator takes manual control in that same window, then signals resume here.
"""

import json
import os
import sys
from pathlib import Path

# Add parent to path so we can import agent modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, redirect, url_for, jsonify
from agent.escalation import (
    list_pending_interventions,
    InterventionRequest,
    INTERVENTIONS_DIR,
    EscalationManager,
)

app = Flask(__name__)


@app.route("/")
def index():
    pending = list_pending_interventions()
    # Also load recently resolved
    resolved = []
    for path in sorted(INTERVENTIONS_DIR.glob("*.json"), reverse=True)[:10]:
        try:
            req = InterventionRequest.load(path)
            if req.status == "resolved":
                resolved.append(req)
        except Exception:
            pass
    return render_template("dashboard.html", pending=pending, resolved=resolved)


@app.route("/interventions/<intervention_id>")
def intervention_detail(intervention_id: str):
    try:
        req = InterventionRequest.load_by_id(intervention_id)
    except FileNotFoundError:
        return "Intervention not found", 404
    return render_template("intervention.html", req=req)


@app.route("/interventions/<intervention_id>/resolve", methods=["POST"])
def resolve_intervention(intervention_id: str):
    notes = request.form.get("notes", "")
    manager = EscalationManager(run_id="operator", artifact_goal="")
    try:
        manager.resolve(intervention_id, notes)
        return redirect(url_for("index"))
    except ValueError as e:
        return str(e), 400


@app.route("/api/interventions")
def api_interventions():
    pending = list_pending_interventions()
    return jsonify([req.to_dict() for req in pending])


@app.route("/api/interventions/<intervention_id>/resolve", methods=["POST"])
def api_resolve(intervention_id: str):
    data = request.get_json(force=True) or {}
    notes = data.get("notes", "")
    manager = EscalationManager(run_id="operator", artifact_goal="")
    try:
        manager.resolve(intervention_id, notes)
        return jsonify({"status": "resolved"})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


if __name__ == "__main__":
    port = int(os.environ.get("OPERATOR_PORT", 6000))
    INTERVENTIONS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Operator UI running at http://localhost:{port}")
    app.run(debug=True, port=port)
