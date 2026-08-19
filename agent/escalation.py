"""
agent/escalation.py — Human-in-the-loop escalation and handoff.

Design:
  - Automation detects a "stuck" state (max retries exceeded, explicit LLM escalation request)
  - Creates an InterventionRequest with full context (step, screenshot, reason, URL)
  - Writes request to a shared JSON file the operator server reads
  - Pauses automation on an asyncio.Event
  - Operator server signals "resume" by writing a response file
  - Automation wakes up, logs what the human did, and continues on the same session

The key design constraint: automation must pause and resume on the SAME browser session —
not a fresh one — so the human's manual actions persist in the same page state.

The operator UI (operator/server.py) reads and writes the same files.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


INTERVENTIONS_DIR = Path("operator/interventions")
INTERVENTION_TIMEOUT_S = 300  # 5 minutes before hard timeout


@dataclass
class InterventionRequest:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    artifact_goal: str = ""
    current_step_seq: int = 0
    current_url: str = ""
    reason: str = ""
    screenshot_path: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "pending"  # "pending" | "resolved" | "abandoned"
    human_notes: str = ""
    resolved_at: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, directory: Path = INTERVENTIONS_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "InterventionRequest":
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(**d)

    @classmethod
    def load_by_id(cls, intervention_id: str, directory: Path = INTERVENTIONS_DIR) -> "InterventionRequest":
        return cls.load(directory / f"{intervention_id}.json")


class EscalationManager:
    """
    Manages the pause-handoff-resume lifecycle.

    Usage in discovery/replay:
        escalator = EscalationManager(run_id=..., artifact_goal=...)
        try:
            await some_action()
        except StuckError:
            await escalator.escalate(reason="...", step_seq=3, current_url=..., screenshot_path=...)
            # ↑ blocks until human resumes or times out
    """

    def __init__(self, run_id: str, artifact_goal: str):
        self.run_id = run_id
        self.artifact_goal = artifact_goal
        self._resume_event: Optional[asyncio.Event] = None
        self._current_intervention: Optional[InterventionRequest] = None

    async def escalate(
        self,
        reason: str,
        step_seq: int,
        current_url: str,
        screenshot_path: str = "",
    ) -> InterventionRequest:
        """
        Pause automation and wait for human to resume.

        1. Creates and persists an InterventionRequest
        2. Waits on asyncio.Event (operator signals resume by writing to the file)
        3. Returns the resolved InterventionRequest when resumed
        4. Times out after INTERVENTION_TIMEOUT_S seconds
        """
        req = InterventionRequest(
            run_id=self.run_id,
            artifact_goal=self.artifact_goal,
            current_step_seq=step_seq,
            current_url=current_url,
            reason=reason,
            screenshot_path=screenshot_path,
        )
        path = req.save()
        self._current_intervention = req
        self._resume_event = asyncio.Event()

        print(f"\n⚑ ESCALATION — Human intervention required!")
        print(f"  Reason: {reason}")
        print(f"  Intervention ID: {req.id}")
        print(f"  Open operator UI at http://localhost:6000")
        print(f"  Or resolve manually: touch operator/interventions/{req.id}.resolved")
        print(f"  Waiting up to {INTERVENTION_TIMEOUT_S}s for resume signal...")

        # Poll for resume signal (operator server writes 'resolved' status to file)
        try:
            await asyncio.wait_for(self._poll_for_resume(path, req.id), timeout=INTERVENTION_TIMEOUT_S)
        except asyncio.TimeoutError:
            req.status = "abandoned"
            req.save()
            raise TimeoutError(f"Intervention {req.id} timed out after {INTERVENTION_TIMEOUT_S}s.")

        # Reload to get human notes
        resolved = InterventionRequest.load(path)
        print(f"\n↩ RESUMED by human. Notes: {resolved.human_notes or '(none)'}")
        return resolved

    async def _poll_for_resume(self, path: Path, intervention_id: str) -> None:
        """Poll the intervention file every 2 seconds until status = 'resolved'."""
        while True:
            try:
                req = InterventionRequest.load(path)
                if req.status == "resolved":
                    return
            except Exception:
                pass
            # Also check for a simple .resolved sentinel file
            if (path.parent / f"{intervention_id}.resolved").exists():
                req = InterventionRequest.load(path)
                req.status = "resolved"
                req.resolved_at = datetime.now(timezone.utc).isoformat()
                req.save()
                return
            await asyncio.sleep(2)

    def resolve(self, intervention_id: str, human_notes: str = "") -> None:
        """
        Called by the operator server or CLI to signal resume.
        Writes the resolved status to the intervention file.
        """
        try:
            req = InterventionRequest.load_by_id(intervention_id)
            req.status = "resolved"
            req.resolved_at = datetime.now(timezone.utc).isoformat()
            req.human_notes = human_notes
            req.save()
        except FileNotFoundError:
            raise ValueError(f"Intervention {intervention_id} not found.")


def list_pending_interventions(directory: Path = INTERVENTIONS_DIR) -> list[InterventionRequest]:
    """List all pending intervention requests (for operator UI)."""
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    for path in sorted(directory.glob("*.json")):
        try:
            req = InterventionRequest.load(path)
            if req.status == "pending":
                results.append(req)
        except Exception:
            continue
    return results
