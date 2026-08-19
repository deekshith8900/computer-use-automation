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
cd Interface.ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### 3. Set Your LLM API Key

```bash
cp .env.example .env
# Edit .env and set your key:
#   OPENAI_API_KEY=sk-...
#   or ANTHROPIC_API_KEY=sk-ant-...
#   or GOOGLE_API_KEY=AIza...
```

### 4. Start the BankDemo App

```bash
python bankdemo/app.py
# → http://localhost:5000
```

### 5. Demo Path

#### Discovery Run (LLM-driven)
```bash
python -m agent discover \
  --goal "Find member Jane Doe and return her account balance" \
  --url http://localhost:5000
```
This will:
- Start a Chromium browser
- Run a real GPT-4o/Claude agent loop
- Record every action as a structured artifact
- Save to `artifacts/<uuid>.json`
- Save logs + screenshots to `evidence/`

#### Replay Run (deterministic, no LLM)
```bash
python -m agent replay \
  --artifact artifacts/<uuid>.json \
  --params '{"member_name": "Jane Doe"}'
```

#### Error Replay (demonstrates error handling)
```bash
python -m agent replay \
  --artifact artifacts/<uuid>.json \
  --params '{"member_name": "NONEXISTENT_PERSON_XYZ"}'
```
Returns a structured `business_outcome` result, not a crash.

---

## Project Structure

```
Interface.ai/
├── bankdemo/               # Target banking demo app
│   ├── app.py             # Flask app
│   ├── templates/         # HTML with ARIA labels
│   └── static/css/
├── agent/                 # Core automation system
│   ├── cli.py             # Entry point
│   ├── discovery.py       # LLM-driven discovery engine
│   ├── replay.py          # Deterministic replay engine
│   ├── artifact.py        # Artifact schema + store
│   ├── browser.py         # Playwright wrapper
│   ├── safety.py          # Policy guardrails
│   ├── escalation.py      # HITL escalation
│   ├── logger.py          # Structured logging
│   └── tests/             # Unit tests
├── artifacts/             # Saved flow artifacts (.json)
├── evidence/              # Run logs + screenshots
├── operator/              # Human-in-the-loop operator UI
│   ├── server.py
│   └── templates/
├── REPORT.md              # Design write-up
└── README.md
```

---

## Operator UI (Human-in-the-Loop)

When the agent gets stuck, it escalates to a human. Start the operator dashboard:

```bash
python operator/server.py
# → http://localhost:6000
```

The operator sees: current step, screenshot, reason for escalation.  
They can take manual action and click **Resume Automation**.

---

## Running Tests

```bash
pytest agent/tests/ -v
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (for GPT-4o) |
| `ANTHROPIC_API_KEY` | Anthropic API key (for Claude) |
| `GOOGLE_API_KEY` | Google API key (for Gemini) |
| `LLM_PROVIDER` | `openai` (default), `anthropic`, or `google` |
| `LLM_MODEL` | Model name override (e.g. `gpt-4o`, `claude-3-5-sonnet-20241022`) |
| `ALLOWED_DOMAINS` | Comma-separated domain allowlist (default: `localhost:5000`) |
| `HEADLESS` | `true` to run browser headless (default: `false`) |
| `OPERATOR_PORT` | Port for operator UI (default: `6000`) |
