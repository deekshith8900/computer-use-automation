"""
BankDemo — A minimal banking demo app for testing computer-use automation.

Simulates a real multi-step financial UI with:
  - Member search → account detail
  - Fund transfer (with confirmation and reversibility)
  - Loan status lookup
  - Intentional error states (not-found, insufficient funds, session timeout)

No real data. No auth. Seeded in-memory store.
"""

import os
import time
import uuid
from flask import Flask, render_template, request, session, redirect, url_for, jsonify

app = Flask(__name__)
app.secret_key = "bankdemo-not-real-secret"

# ─── Seeded in-memory data ───────────────────────────────────────────────────

MEMBERS = {
    "M001": {
        "id": "M001",
        "name": "Jane Doe",
        "email": "jane.doe@example.com",
        "phone": "555-0101",
        "accounts": ["ACC-1001", "ACC-1002"],
    },
    "M002": {
        "id": "M002",
        "name": "John Smith",
        "email": "john.smith@example.com",
        "phone": "555-0102",
        "accounts": ["ACC-2001"],
    },
    "M003": {
        "id": "M003",
        "name": "Alice Johnson",
        "email": "alice.j@example.com",
        "phone": "555-0103",
        "accounts": ["ACC-3001", "ACC-3002"],
    },
    "M004": {
        "id": "M004",
        "name": "Bob Williams",
        "email": "bob.w@example.com",
        "phone": "555-0104",
        "accounts": ["ACC-4001"],
    },
}

ACCOUNTS = {
    "ACC-1001": {"id": "ACC-1001", "owner": "M001", "type": "Checking", "balance": 4823.50, "currency": "USD"},
    "ACC-1002": {"id": "ACC-1002", "owner": "M001", "type": "Savings",  "balance": 12340.00, "currency": "USD"},
    "ACC-2001": {"id": "ACC-2001", "owner": "M002", "type": "Checking", "balance": 892.75, "currency": "USD"},
    "ACC-3001": {"id": "ACC-3001", "owner": "M003", "type": "Checking", "balance": 15000.00, "currency": "USD"},
    "ACC-3002": {"id": "ACC-3002", "owner": "M003", "type": "Savings",  "balance": 50000.00, "currency": "USD"},
    "ACC-4001": {"id": "ACC-4001", "owner": "M004", "type": "Checking", "balance": 125.00, "currency": "USD"},
}

LOANS = {
    "LN-2024-001": {"id": "LN-2024-001", "owner": "M001", "type": "Mortgage",   "amount": 250000, "status": "Active",   "monthly_payment": 1450.00, "next_due": "2026-09-01"},
    "LN-2024-002": {"id": "LN-2024-002", "owner": "M002", "type": "Auto",        "amount": 18000,  "status": "Active",   "monthly_payment": 340.00,  "next_due": "2026-09-15"},
    "LN-2024-003": {"id": "LN-2024-003", "owner": "M003", "type": "Personal",    "amount": 5000,   "status": "Paid Off", "monthly_payment": 0.00,    "next_due": None},
}

# In-memory transaction log (mutated by transfers)
TRANSACTIONS: list[dict] = []


# ─── Helpers ────────────────────────────────────────────────────────────────

def search_members(query: str) -> list[dict]:
    q = query.strip().lower()
    if not q:
        return []
    return [m for m in MEMBERS.values() if q in m["name"].lower() or q in m["email"].lower()]


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/members", methods=["GET", "POST"])
def members():
    query = ""
    results = []
    searched = False
    if request.method == "POST":
        query = request.form.get("query", "")
        results = search_members(query)
        searched = True
    return render_template("members.html", query=query, results=results, searched=searched)


@app.route("/members/<member_id>")
def member_detail(member_id: str):
    member = MEMBERS.get(member_id)
    if not member:
        return render_template("error.html", code=404, message=f"Member '{member_id}' not found."), 404
    member_accounts = [ACCOUNTS[a] for a in member["accounts"] if a in ACCOUNTS]
    member_loans = [l for l in LOANS.values() if l["owner"] == member_id]
    return render_template("member_detail.html", member=member, accounts=member_accounts, loans=member_loans)


@app.route("/accounts/<account_id>")
def account_detail(account_id: str):
    account = ACCOUNTS.get(account_id)
    if not account:
        return render_template("error.html", code=404, message=f"Account '{account_id}' not found."), 404
    owner = MEMBERS.get(account["owner"])
    txns = [t for t in TRANSACTIONS if t["from_account"] == account_id or t["to_account"] == account_id]
    return render_template("account_detail.html", account=account, owner=owner, transactions=txns)


@app.route("/transfer", methods=["GET", "POST"])
def transfer():
    """Multi-step fund transfer: form → confirm → execute."""
    error = None
    success = None

    if request.method == "POST":
        step = request.form.get("step", "form")

        if step == "confirm":
            # Show confirmation page
            from_account = request.form.get("from_account", "").strip()
            to_account = request.form.get("to_account", "").strip()
            amount_str = request.form.get("amount", "0").strip()

            # Validate
            if from_account not in ACCOUNTS:
                error = f"Source account '{from_account}' not found."
            elif to_account not in ACCOUNTS:
                error = f"Destination account '{to_account}' not found."
            elif from_account == to_account:
                error = "Cannot transfer to the same account."
            else:
                try:
                    amount = float(amount_str)
                except ValueError:
                    error = "Invalid amount."
                else:
                    if amount <= 0:
                        error = "Amount must be positive."
                    elif ACCOUNTS[from_account]["balance"] < amount:
                        error = f"Insufficient funds. Available: ${ACCOUNTS[from_account]['balance']:.2f}"

            if not error:
                return render_template(
                    "transfer_confirm.html",
                    from_account=ACCOUNTS[from_account],
                    to_account=ACCOUNTS[to_account],
                    amount=amount,
                )

        elif step == "execute":
            from_id = request.form.get("from_account_id", "")
            to_id = request.form.get("to_account_id", "")
            amount = float(request.form.get("amount", 0))

            if from_id in ACCOUNTS and to_id in ACCOUNTS and from_id != to_id:
                if ACCOUNTS[from_id]["balance"] >= amount:
                    ACCOUNTS[from_id]["balance"] -= amount
                    ACCOUNTS[to_id]["balance"] += amount
                    txn = {
                        "id": f"TXN-{uuid.uuid4().hex[:8].upper()}",
                        "from_account": from_id,
                        "to_account": to_id,
                        "amount": amount,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    TRANSACTIONS.append(txn)
                    success = f"Transfer of ${amount:.2f} completed. Transaction ID: {txn['id']}"
                else:
                    error = "Insufficient funds at execution time."
            else:
                error = "Invalid accounts."

    all_accounts = list(ACCOUNTS.values())
    return render_template("transfer.html", accounts=all_accounts, error=error, success=success)


@app.route("/loans", methods=["GET", "POST"])
def loans():
    query = ""
    result = None
    error = None
    if request.method == "POST":
        query = request.form.get("loan_id", "").strip()
        result = LOANS.get(query)
        if not result:
            error = f"Loan application '{query}' not found."
    return render_template("loans.html", query=query, result=result, error=error)


# ─── API endpoints (for programmatic access / tests) ─────────────────────────

@app.route("/api/members")
def api_members():
    q = request.args.get("q", "")
    return jsonify(search_members(q) if q else list(MEMBERS.values()))


@app.route("/api/members/<member_id>")
def api_member(member_id):
    m = MEMBERS.get(member_id)
    if not m:
        return jsonify({"error": "not_found", "member_id": member_id}), 404
    return jsonify(m)


@app.route("/api/accounts/<account_id>")
def api_account(account_id):
    a = ACCOUNTS.get(account_id)
    if not a:
        return jsonify({"error": "not_found", "account_id": account_id}), 404
    return jsonify(a)


@app.route("/api/status")
def api_status():
    return jsonify({"status": "ok", "members": len(MEMBERS), "accounts": len(ACCOUNTS)})


if __name__ == "__main__":
    port = int(os.environ.get("BANKDEMO_PORT", 5001))
    print(f"BankDemo running at http://localhost:{port}")
    app.run(debug=True, port=port)
