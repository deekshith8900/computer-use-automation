# Computer-Use Automation System

A working **Computer-Use Automation System** built for Interface.ai Assignment A.

The system lets an LLM-driven agent **discover** UI flows on a live web app, record them as portable **artifacts**, and **replay** those flows deterministically — with full error handling, safety guardrails, and human-in-the-loop escalation.

---

## Quick Start

### 1. Prerequisites

```bash
python3.11+  pip  git
```

### 2. Install Dependencies

```bash
cd computer-use-automation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### 3. Set Your LLM API Key

```bash
cp .env.example .env
# Edit .env — choose one provider:
#   OpenRouter (recommended — supports many models):
#     LLM_PROVIDER=openrouter
#     OPENROUTER_API_KEY=sk-or-v1-...
#     LLM_MODEL=openai/gpt-4o
#
#   OpenAI directly:
#     LLM_PROVIDER=openai
#     OPENAI_API_KEY=sk-...
#
#   Anthropic:
#     LLM_PROVIDER=anthropic
#     ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Start the BankDemo App

```bash
python bankdemo/app.py
# → http://localhost:5001
```

### 5. Quick Demo (No LLM Required)

Generate a hand-crafted artifact and run all replay scenarios instantly:

```bash
python scripts/generate_demo_artifact.py --headless --url http://localhost:5001
```

This produces:
- `artifacts/find_member_*.json` — validated artifact with typed params/outputs
- `evidence/replay_*_success.log` — successful Jane Doe balance extraction
- `evidence/replay_*_business_outcome.log` — "no results" for unknown member
- `evidence/replay_*_hitl.log` — escalation → human resume → continuation

### 6. Discovery Run (LLM-driven)

```bash
python -m agent discover \
  --goal "Find member Jane Doe and return her account balance" \
  --url http://localhost:5001
```

This will:
- Start a Chromium browser (headed by default)
- Run a real LLM agent loop with safety guardrails enforced
- Record every action as a validated, typed artifact
- Save to `artifacts/<uuid>.json`
- Save logs + screenshots to `evidence/`

### 7. Replay Run (deterministic, no LLM)

```bash
python -m agent replay \
  --artifact artifacts/<uuid>.json \
  --params '{"member_name": "Jane Doe"}'
```

### 8. Error Replay (demonstrates business outcome handling)

```bash
python -m agent replay \
  --artifact artifacts/<uuid>.json \
  --params '{"member_name": "NONEXISTENT_PERSON_XYZ"}'
```

Returns a structured `business_outcome` result — never a crash.

---

## Project Structure

```
computer-use-automation/
├── bankdemo/               # Target banking demo app (Flask, port 5001)
│   ├── app.py             # Members, accounts, transfers, loans, loans
│   ├── templates/         # HTML with data-testid + ARIA labels
│   └── static/css/
├── agent/                 # Core automation system
│   ├── cli.py             # CLI entry point (discover / replay / list)
│   ├── discovery.py       # LLM-driven discovery engine + safety enforcement
│   ├── replay.py          # Deterministic replay + HITL escalation
│   ├── artifact.py        # Typed artifact schema + validation
│   ├── browser.py         # Playwright wrapper + multi-strategy locators
│   ├── safety.py          # Domain allowlist + reversibility guardrails
│   ├── escalation.py      # HITL pause/resume on same Playwright session
│   ├── logger.py          # Structured JSONL run logging
│   └── tests/             # 80 unit tests
├── scripts/
│   └── generate_demo_artifact.py  # Creates artifact + runs all scenarios
├── artifacts/             # Saved flow artifacts (.json) — committed
├── evidence/              # Run logs + screenshots — committed
├── operator/              # Human-in-the-loop operator UI (port 6000)
│   ├── server.py
│   └── templates/
├── REPORT.md              # 7-section design write-up
└── README.md
```

---

## Operator UI (Human-in-the-Loop)

When retries are exhausted the agent pauses and escalates. Start the operator dashboard:

```bash
python operator/server.py
# → http://localhost:6000
```

The operator sees: current step, URL, screenshot, reason for escalation.
They take manual action then click **Resume Automation** — the agent continues
on the **same** Playwright session with the human's page state intact.

---

## Running Tests

```bash
pytest agent/tests/ -v
# 80 tests — artifact schema, validation, replay logic, safety, param substitution
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | `openrouter` (default), `openai`, or `anthropic` |
| `OPENROUTER_API_KEY` | OpenRouter API key — supports GPT-4o, Claude, Gemini, Llama |
| `LLM_MODEL` | Model name for OpenRouter (e.g. `openai/gpt-4o`, `anthropic/claude-3-5-sonnet`) |
| `OPENAI_API_KEY` | OpenAI API key (when `LLM_PROVIDER=openai`) |
| `ANTHROPIC_API_KEY` | Anthropic API key (when `LLM_PROVIDER=anthropic`) |
| `ALLOWED_DOMAINS` | Comma-separated domain allowlist (default: `localhost:5001`) |
| `HEADLESS` | `true` to run browser headless (default: `false`) |
| `OPERATOR_PORT` | Port for operator UI (default: `6000`) |

---

## Evidence

The `evidence/` and `artifacts/` directories contain real outputs committed to this repo:

| File | What it shows |
|---|---|
| `artifacts/find_member_*.json` | Validated typed artifact — 6 steps, param + output declarations |
| `evidence/replay_*_success.log` | Jane Doe found, `account_balance` extracted |
| `evidence/replay_*_business_outcome.log` | Unknown member → `business_outcome` result |
| `evidence/replay_*_hitl.log` | Retry exhaustion → escalation → human resume → continuation |
| `operator/interventions/*.json` | Intervention record from HITL demo run |
