# REPORT — Computer-Use Automation System

## 1. Architecture

### Overview

The system is a single-process Python application with clean internal boundaries between its three principal concerns:

```
CLI (discover / replay / resolve)
    │
    ├── DiscoveryEngine     — LLM-driven agent loop (uses BrowserManager + LLM API)
    │       │
    │       └── records → Artifact (JSON)
    │
    ├── ReplayEngine        — Deterministic executor (uses BrowserManager, NO LLM)
    │       │
    │       └── reads  → Artifact (JSON)
    │
    ├── Guardrail           — Policy enforcement (checked on every step, before execution)
    ├── EscalationManager   — HITL pause/resume (asyncio.Event + shared JSON files)
    └── RunLogger           — JSONL evidence + Rich console output
```

**Key architectural decision: single-process, simple.** I deliberately avoided microservices, queues, or distributed state. The interesting problems here are design problems (artifact schema, replay determinism, error classification, HITL seam), not scaling problems. A simple, correct, debuggable system beats a complex one for this scope.

**The LLM is used exactly once per flow: during discovery.** The replay engine is pure Python — no model calls, no non-determinism. This means replay is cheap, auditable, and consistent.

**Separation of perception and action.** The discovery engine sends a screenshot + accessibility tree text to the LLM. The LLM returns a structured tool call. The engine executes it, records the step, and loops. The LLM never sees internal system state — only what a human user would see on-screen.

### Trade-offs

| Decision | Alternative | Why I chose this |
|---|---|---|
| Single process | Separate services (orchestrator + browser worker) | Simpler to run, easier to debug, handoff is in-process (same event loop) |
| JSON artifact file | Database / artifact service | Human-readable, Git-friendly, no infra dependency, portable |
| asyncio-based HITL pause | REST polling / WebSocket push | Simpler; browser session lives in the same event loop so pausing is natural |
| Playwright over screenshot-only CUA | Pure screenshot + mouse coordinates | DOM/a11y tree gives semantic, stable locators; screenshots are supplementary for LLM vision |
| Fail-closed safety | Warn and continue | Any ambiguity should surface as an error, not silently proceed |

---

## 2. Artifact Schema

### Schema Design

```json
{
  "schema_version": "1.0",
  "id": "uuid",
  "goal": "Find member Jane Doe and return her account balance",
  "surface": "web",
  "base_url": "http://localhost:5000",
  "discovery_model": "openai/gpt-4o",
  "params": { "member_name": "string" },
  "outputs": ["account_balance"],
  "steps": [
    {
      "seq": 1, "action": "navigate", "url": "http://localhost:5000/members"
    },
    {
      "seq": 2, "action": "fill",
      "locator": { "strategy": "aria-label", "value": "Search members" },
      "value": "{{member_name}}"
    },
    {
      "seq": 3, "action": "click",
      "locator": { "strategy": "data-testid", "value": "search-button" },
      "checkpoint": { "type": "element_visible", "locator": { "strategy": "data-testid", "value": "search-results" } },
      "on_not_found": "business_outcome",
      "business_outcome_signal": "No members found"
    },
    {
      "seq": 4, "action": "extract",
      "locator": { "strategy": "data-testid", "value": "current-balance" },
      "output_key": "account_balance"
    }
  ],
  "safety": {
    "allowed_domains": ["localhost:5000"],
    "reversibility": "read-only",
    "requires_confirmation": false
  }
}
```

### Why this shape

**Parameterized, not hardcoded.** The `params` map and `{{placeholder}}` syntax mean a single recorded artifact can be replayed against any member name, loan ID, or account — without re-running the LLM.

**Declared outputs.** The caller (a human or an upstream AI agent) knows exactly what values they'll get back before running the artifact. This makes the artifact behave like a typed function: `find_member_balance(member_name: str) → {account_balance: str}`.

**Checkpoints are mandatory on mutating steps.** After every click that causes navigation or state change, the artifact asserts a condition. This is what makes replay verifiable rather than hopeful.

**Business outcomes in the schema, not the error handler.** If searching for an unknown member is a *valid* result the caller needs to act on, that's encoded directly in the step (`on_not_found: business_outcome`, `business_outcome_signal: "No members found"`). The replay engine doesn't guess — it checks.

**Safety embedded in the artifact.** The artifact carries its own `allowed_domains` and `reversibility` classification. This means the policy travels with the flow, and callers can inspect it before running.

**`schema_version`** allows forward compatibility without breaking existing artifacts.

---

## 3. Determinism & Error Handling

### Making Replay Deterministic

1. **No LLM on replay.** Every decision that was made during discovery is now encoded as a step. Replay reads and executes, period.

2. **Locator strategy priority.** Locators are stored using the most stable strategy first: `aria-label` > `data-testid` > `role+name` > `text` > `css`. Each locator also carries a `fallbacks` list. The browser module tries each in order, so if the primary selector changes, the fallback catches it.

3. **Explicit waits via checkpoints.** After every navigation or click, the replay engine evaluates the checkpoint (element visible, URL contains, text contains) before proceeding. It never assumes the click worked.

4. **Idempotent navigation.** Navigate steps include the full URL, not relative paths, so they're not sensitive to starting page state.

### Error Classification (the core design)

The replay engine distinguishes three error classes — never conflating them:

| Class | Definition | Example | Response |
|---|---|---|---|
| `business_outcome` | A known, valid non-success result the caller must handle | "No members found", "Insufficient funds" | Return structured result, not exception |
| `recoverable_error` | A transient condition the system can retry | Locator timeout on slow load, known interstitial | Dismiss/wait/retry up to `MAX_RETRIES_PER_STEP` times |
| `hard_failure` | Unexpected state; cannot safely continue | Unknown dialog, policy violation, checkpoint failure | Stop immediately, return error with step, detail, screenshot |

**The most common design mistake avoided here:** treating "no such member" as a crash. It's not — it's a legitimate answer the caller needs to route. The artifact schema encodes these signals explicitly, and the replay result type has a dedicated `business_outcome` status.

### UI Drift

Because we target a controlled, stable app (BankDemo), layout drift is not the primary concern. The locator priority (aria-label > testid > role) makes the replay robust to cosmetic changes without needing re-recording. For legacy surfaces with no test IDs, the fallback chain covers CSS and XPath as a last resort.

---

## 4. Heterogeneity & Multi-Tenant

### Surface Abstraction

The replay engine talks to the surface through the `BrowserManager` abstraction. The artifact schema is **surface-agnostic at the step level** — actions like `navigate`, `fill`, `click`, `extract` describe *intent*, not browser calls. The `surface` field in the artifact (`"web"` or `"desktop"`) routes to the correct backend implementation.

To add a **legacy web surface** (iframes, table-based layout, no test IDs):
- Same artifact schema
- `BrowserManager` extended with frame traversal and accessibility-tree-based locator resolution (already available in Playwright)
- Locator fallback chain already handles CSS and XPath for when ARIA attributes are absent

To add a **desktop surface** (Windows/Mac native apps):
- Same artifact schema
- `BrowserManager` replaced with a `DesktopManager` backed by `pywinauto` (Windows) or `pyautogui` + macOS Accessibility APIs
- The seam is clean: only the `BrowserManager` interface changes, not the artifact schema or replay engine

### Multi-Tenant Reuse

In production, hundreds of tenants run the same vendor product (e.g., the same loan management system). Recording one artifact per tenant would be wasteful and brittle.

**Proposed design:**
1. Artifacts are recorded against a **canonical** tenant (e.g., the vendor's reference deployment)
2. Tenant-specific **overrides** are stored separately: `{ "base_url": "https://tenant-A.vendor.com", "params": { "login_url": "..." } }`
3. The replay engine merges the canonical artifact with tenant overrides at runtime
4. **Drift detection**: a background job replays each artifact against each tenant on a schedule. If a checkpoint fails consistently, it surfaces a drift alert for that tenant — prompting a partial re-record

**What prevents per-tenant re-records?** Parameterized steps + ARIA/testid locators. If the vendor app uses consistent ARIA labels across tenants (which well-built vendor apps do), the same artifact replays on all tenants. Per-tenant overrides handle only the configuration differences (base URL, custom fields, locale).

---

## 5. Escalation & Handoff

### Detecting "Stuck"

The system escalates when:
1. **LLM explicitly calls `escalate`** — the model determines it cannot proceed safely
2. **Max retries exhausted** on a recoverable error — the system tried `MAX_RETRIES_PER_STEP` times and still failed
3. **Policy violation on a required step** — e.g., an irreversible action is required but blocked

### The Handoff Mechanism

```
Automation (async) ──pause──►  asyncio.Event (waiting)
                                     │
InterventionRequest.json ◄──────────┘
                                     │
Operator Dashboard (Flask) ◄─────────┘ reads JSON, shows context + screenshot
                                     │
Operator takes manual action in live browser window
                                     │
POST /interventions/<id>/resolve ────►  writes "resolved" to JSON file
                                     │
asyncio poller detects status change
                                     │
Automation ◄──resume──  continues on same Playwright session, same page state
```

**Critical property: same session, not a new one.** The Playwright browser and page objects remain alive throughout the pause. The human operator is directed to the same browser window (by URL). Their manual actions persist in the page state. When automation resumes, it continues from the real current state — not a re-navigated fresh one.

**Evidence across the handoff:** The intervention request records the step, URL, reason, and screenshot at escalation time. After resume, the log records what the human noted. Nothing is lost.

**The "real seam" vs. full operator console:** A production-grade real-time co-browsing console (like a shared tab + WebSocket video feed) is out of scope and explicitly documented as a cut. The mechanism here — pause + intervention file + operator UI + resume — is real, well-reasoned, and correct. The seam for a richer operator surface is clean and documented.

---

## 6. Safety

### Guardrail Model

Safety is checked **before execution**, not logged after. Every step passes through the `Guardrail` before the browser acts. Fail-closed: if unsure, block.

**Domain allowlist:** The agent may only navigate to domains in the configured allowlist (from environment variable `ALLOWED_DOMAINS` and the artifact's own `safety.allowed_domains`). Any navigation outside the allowlist raises a `HardFailureError` immediately.

**Reversibility classification:** Every action is classified:
- `read-only`: navigate, assert, extract, wait
- `write-reversible`: fill, select (user can navigate away)
- `write-irreversible`: click on a confirm/transfer/delete URL

By default, `write-irreversible` actions are **blocked**. They must be explicitly permitted by the caller (`--allow-irreversible` flag, or `irreversible_policy: "allow"` in policy config). This is the conservative choice: it prevents accidental money movement.

**PII and secret redaction:** The `redact()` function strips emails, card numbers, SSNs, API keys, and Bearer tokens from any value before it's written to an artifact or log. Field names in `_REDACT_FIELD_NAMES` (password, token, etc.) are always fully redacted to `[REDACTED]`.

### Limits

- The redaction is pattern-based, not semantic. A creative attacker could encode PII in a way that avoids patterns.
- The domain allowlist protects against navigation but not against a compromised target app redirecting to an off-list domain via JavaScript.
- Reversibility classification is heuristic (URL patterns + button labels). A well-named irreversible button on an innocuous-looking URL could slip through. Mitigation: always require explicit confirmation for real deployment targets.

---

## 7. Cuts

### Deliberately Omitted

| Feature | Reason | What I'd build next |
|---|---|---|
| Real-time co-browsing operator console | Out of scope per assignment; the seam is clean | Shared browser tab via CDP + WebSocket video feed |
| Multi-tenant artifact store with versioning | Premature infra | Artifact catalog API with tenant override storage |
| Desktop surface (pywinauto) | Same abstraction, different backend | Add `DesktopManager` behind the same `BrowserManager` interface |
| Authentication flows | No real credentials permitted | Credential injection via environment-only; never in artifacts |
| Artifact confidence scoring | Stretch goal; no time | Replay N times, track checkpoint failure rate, gate on `approved` state |
| LLM-assisted fallback on replay failure | Stretch goal | Bounded, policy-checked single-step recovery with evidence recording |
| Cross-tenant canonicalization | Stretch goal | URL parameterization (`/item/12345` → `/item/:id`) + override registry |

### Architecture Assumptions That Would Need Revisiting at Scale

- **Single-process**: works for one concurrent run. At scale, browser sessions would move to separate workers (Playwright in separate processes or containers).
- **File-based intervention store**: works for one agent. At scale, replace with a lightweight database and push notifications to the operator console.
- **Accessibility tree + screenshot perception**: accurate on modern, accessible apps. Legacy apps with no ARIA labels would need additional perception strategies (OCR, screenshot diff, coordinate-based fallback).
