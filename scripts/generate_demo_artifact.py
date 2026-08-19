#!/usr/bin/env python3
"""
scripts/generate_demo_artifact.py

Creates a canonical hand-crafted artifact for the "Find member Jane Doe and
return her account balance" flow, then runs three replay scenarios against a
live BankDemo to produce committed evidence:

  1. success replay        (Jane Doe found, balance extracted)
  2. business_outcome      (NOBODY_XYZ → no results page)
  3. HITL demo             (deliberately broken locator → retry exhaustion →
                            auto-resolve intervention after 5 s)

Usage:
    python scripts/generate_demo_artifact.py [--headless] [--url http://localhost:5001]

Requires BankDemo running:
    python bankdemo/app.py
"""

import argparse
import asyncio
import json
import os
import sys
import time
import threading
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.artifact import (
    Artifact, ArtifactParam, ArtifactOutput, ArtifactSafety,
    Step, Locator, Checkpoint, ArtifactStore,
)
from agent.replay import ReplayEngine
from agent.escalation import InterventionRequest, INTERVENTIONS_DIR


# ─── Canonical Artifact Definition ───────────────────────────────────────────

def build_artifact(base_url: str) -> Artifact:
    """
    Hand-crafted artifact for: Find member {{member_name}} and return balance.

    Steps:
      1. Navigate to base URL
      2. Navigate to /members search page
      3. Fill member name into search field  [parameterized — field name=query]
      4. Click Search button (submits POST form)
      5. Click first result's View Details   [css: a[data-testid^='view-member']]
      6. Extract total balance               [data-testid=total-balance]

    All locators verified against live BankDemo HTML.
    """
    host = base_url.rstrip("/").split("://")[-1].split("/")[0]

    return Artifact(
        goal="Find member {{member_name}} and return their account balance",
        surface="web",
        base_url=base_url.rstrip("/"),
        discovery_model="hand-crafted",
        param_defs=[
            ArtifactParam(
                name="member_name",
                type="string",
                required=True,
                description="Full or partial name of the bank member to look up",
            )
        ],
        output_defs=[
            ArtifactOutput(
                key="account_balance",
                type="string",
                description="Total account balance shown on the member detail page",
            )
        ],
        safety=ArtifactSafety(
            allowed_domains=[host],
            reversibility="read-only",
            requires_confirmation=False,
        ),
        steps=[
            Step(
                seq=1,
                action="navigate",
                url=base_url.rstrip("/"),
                description="Initial navigation to base URL",
            ),
            Step(
                seq=2,
                action="navigate",
                url=f"{base_url.rstrip('/')}/members",
                description="Navigate to member search page",
                checkpoint=Checkpoint(type="url_contains", value="/members"),
            ),
            Step(
                seq=3,
                action="fill",
                locator=Locator(
                    strategy="data-testid",
                    value="search-input",
                    fallbacks=[
                        {"strategy": "css", "value": "input[name='query']"},
                        {"strategy": "css", "value": "input[type='text']"},
                    ],
                ),
                value="{{member_name}}",
                description="Fill member name into search field [param:member_name]",
            ),
            Step(
                seq=4,
                action="click",
                locator=Locator(
                    strategy="data-testid",
                    value="search-button",
                    fallbacks=[
                        {"strategy": "text", "value": "Search"},
                        {"strategy": "css", "value": "button[type='submit']"},
                    ],
                ),
                description="Click Search button to submit the form",
                checkpoint=Checkpoint(type="url_contains", value="/members"),
                on_not_found="business_outcome",
                business_outcome_signal="no results",
            ),
            Step(
                seq=5,
                action="click",
                locator=Locator(
                    strategy="css",
                    value="a[data-testid^='view-member']",
                    fallbacks=[
                        {"strategy": "text", "value": "View Details"},
                        {"strategy": "css", "value": "a.btn-primary"},
                    ],
                ),
                description="Click View Details for the first search result",
                on_not_found="business_outcome",
                business_outcome_signal="No members found",
                checkpoint=Checkpoint(type="url_contains", value="/members/"),
            ),
            Step(
                seq=6,
                action="extract",
                locator=Locator(
                    strategy="data-testid",
                    value="total-balance",
                    fallbacks=[
                        {"strategy": "css", "value": ".total-balance strong"},
                        {"strategy": "css", "value": "[data-testid='total-balance']"},
                    ],
                ),
                output_key="account_balance",
                description="Extract total balance from member detail page",
            ),
        ],
    )


# ─── Replay Scenarios ─────────────────────────────────────────────────────────

async def run_success(artifact: Artifact, headless: bool) -> None:
    print("\n" + "=" * 60)
    print("SCENARIO 1: SUCCESS (Jane Doe)")
    print("=" * 60)
    engine = ReplayEngine(
        artifact,
        params={"member_name": "Jane"},
        headless=headless,
        evidence_dir="evidence",
    )
    result = await engine.run()
    print(f"\nResult: {result.status}")
    print(f"Outputs: {result.outputs}")
    assert result.status == "success", f"Expected success, got {result.status}: {result.error_detail}"
    assert "account_balance" in result.outputs, "Missing account_balance output"
    print("✓ SUCCESS scenario passed")


async def run_business_outcome(artifact: Artifact, headless: bool) -> None:
    print("\n" + "=" * 60)
    print("SCENARIO 2: BUSINESS OUTCOME (unknown member)")
    print("=" * 60)
    engine = ReplayEngine(
        artifact,
        params={"member_name": "NOBODY_XYZ_NONEXISTENT"},
        headless=headless,
        evidence_dir="evidence",
    )
    result = await engine.run()
    print(f"\nResult: {result.status}")
    print(f"Signal: {result.business_outcome_signal}")
    assert result.status == "business_outcome", \
        f"Expected business_outcome, got {result.status}: {result.error_detail}"
    print("✓ BUSINESS OUTCOME scenario passed")


async def run_hitl_demo(artifact: Artifact, headless: bool) -> None:
    """
    HITL scenario: use an artifact with a deliberately broken locator on step 5
    (View Details) so retries are exhausted, escalation fires, then a background
    thread auto-resolves the intervention after 5 seconds — demonstrating the
    pause → human takeover → resume lifecycle without needing a human present.
    """
    print("\n" + "=" * 60)
    print("SCENARIO 3: HITL DEMO (broken locator → escalation → auto-resolve)")
    print("=" * 60)

    # Clone the artifact but break step 5's locator
    broken = Artifact.from_dict(artifact.to_dict())
    broken.steps[4].locator = Locator(
        strategy="css",
        value=".THIS_ELEMENT_DOES_NOT_EXIST_HITL_DEMO",
    )
    broken.steps[4].fallbacks = []
    # No business_outcome_signal so it escalates via retry exhaustion
    broken.steps[4].on_not_found = "fail"
    broken.steps[4].business_outcome_signal = None

    # Background thread: watch for pending interventions and auto-resolve after 5s
    resolved = threading.Event()

    def auto_resolver():
        deadline = time.time() + 60  # wait up to 60 s
        while time.time() < deadline:
            time.sleep(2)
            INTERVENTIONS_DIR.mkdir(parents=True, exist_ok=True)
            for p in sorted(INTERVENTIONS_DIR.glob("*.json")):
                try:
                    req = InterventionRequest.load(p)
                    if req.status == "pending":
                        print(f"\n[auto-resolver] Found pending intervention {req.id[:8]}")
                        print(f"[auto-resolver] Reason: {req.reason}")
                        time.sleep(5)  # simulate human reviewing the page
                        req.status = "resolved"
                        req.human_notes = (
                            "Auto-resolved by demo script. "
                            "Human navigated to /members/M001 directly."
                        )
                        from datetime import datetime, timezone
                        req.resolved_at = datetime.now(timezone.utc).isoformat()
                        req.save()
                        print(f"[auto-resolver] Intervention resolved ✓")
                        resolved.set()
                        return
                except Exception:
                    continue

    t = threading.Thread(target=auto_resolver, daemon=True)
    t.start()

    engine = ReplayEngine(
        broken,
        params={"member_name": "Jane"},
        headless=headless,
        evidence_dir="evidence",
    )
    result = await engine.run()
    t.join(timeout=10)

    print(f"\nResult: {result.status}")
    print(f"Detail: {result.error_detail}")
    # HITL demo may succeed (if auto-resolver worked in time) or hard_failure
    # (if timeout). Either way it demonstrates the escalation lifecycle.
    print(f"{'✓ HITL DEMO: escalation fired and resolved' if resolved.is_set() else '⚠ HITL DEMO: auto-resolver did not fire in time'}")


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main(base_url: str, headless: bool, skip_hitl: bool) -> None:
    print(f"Building canonical artifact for {base_url} ...")
    artifact = build_artifact(base_url)

    # Validate the artifact before saving
    errors = artifact.validate()
    if errors:
        print("Artifact validation FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    # Save to artifacts/
    store = ArtifactStore("artifacts")
    path = store.save(artifact)
    print(f"Artifact saved: {path}")

    # Print a compact summary
    print(f"\nArtifact summary:")
    print(f"  ID         : {artifact.id}")
    print(f"  Steps      : {len(artifact.steps)}")
    print(f"  Params     : {[p.name for p in artifact.param_defs]}")
    print(f"  Outputs    : {[o.key for o in artifact.output_defs]}")
    print(f"  Safety     : {artifact.safety.allowed_domains} / {artifact.safety.reversibility}")

    # Run replay scenarios
    await run_success(artifact, headless)
    await run_business_outcome(artifact, headless)

    if not skip_hitl:
        await run_hitl_demo(artifact, headless)

    print("\n" + "=" * 60)
    print("All scenarios complete. Check evidence/ for logs and screenshots.")
    print(f"Artifact: {path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate demo artifact and run evidence scenarios")
    parser.add_argument("--url", default="http://localhost:5001", help="BankDemo base URL")
    parser.add_argument("--headless", action="store_true", help="Run browser headless")
    parser.add_argument("--skip-hitl", action="store_true", help="Skip HITL demo scenario")
    args = parser.parse_args()

    asyncio.run(main(args.url, args.headless, args.skip_hitl))
