"""
agent/replay.py — Deterministic replay engine.

Replays a saved artifact with NO LLM decisions.
Every step is executed exactly as recorded, with:
  - Parameter substitution ({{member_name}} → actual value)
  - Pre-execution safety checks
  - Checkpoint evaluation after each step
  - Structured error classification:
      success           — completed, outputs extracted
      business_outcome  — known non-success result (e.g. "no such member")
      recoverable_error — transient condition handled (dismissed interstitial, retried)
      hard_failure      — unexpected state, stop and return debuggable error

Design note: "no such member" is a legitimate result the caller needs to act on —
not a crash. We never conflate business outcomes with failures.
"""

from __future__ import annotations

import asyncio
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from agent.artifact import Artifact, Step, Locator, Checkpoint
from agent.browser import BrowserManager, evaluate_checkpoint
from agent.safety import Guardrail, Policy, PolicyViolation
from agent.escalation import EscalationManager
from agent.logger import RunLogger


# ─── Result Types ─────────────────────────────────────────────────────────────

@dataclass
class ReplayResult:
    """Structured result returned by replay. Never raises on business outcomes."""
    status: str  # "success" | "business_outcome" | "recoverable_error" | "hard_failure"
    outputs: dict[str, Any] = field(default_factory=dict)
    business_outcome_signal: str = ""
    error_step: Optional[int] = None
    error_detail: str = ""
    screenshot_path: str = ""
    log_path: str = ""

    def is_success(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "outputs": self.outputs,
            "business_outcome_signal": self.business_outcome_signal,
            "error_step": self.error_step,
            "error_detail": self.error_detail,
            "screenshot_path": self.screenshot_path,
            "log_path": self.log_path,
        }


# ─── Parameter Substitution ───────────────────────────────────────────────────

_PARAM_PATTERN = re.compile(r"\{\{(\w+)\}\}")


def substitute_params(value: str, params: dict[str, str]) -> str:
    """Replace {{param_name}} placeholders with actual values."""
    def replacer(match):
        key = match.group(1)
        if key not in params:
            raise ValueError(f"Missing required parameter: '{key}'")
        return params[key]
    return _PARAM_PATTERN.sub(replacer, value)


# ─── Replay Engine ────────────────────────────────────────────────────────────

MAX_RETRIES_PER_STEP = 2
WAIT_BETWEEN_RETRIES_MS = 1500


class ReplayEngine:
    """
    Executes an artifact deterministically.

    Usage:
        engine = ReplayEngine(artifact, params={"member_name": "Jane Doe"})
        result = await engine.run()
    """

    def __init__(
        self,
        artifact: Artifact,
        params: dict[str, str] | None = None,
        evidence_dir: str = "evidence",
        headless: bool = False,
        allow_irreversible: bool = False,
    ):
        self.artifact = artifact
        self.params = params or {}
        self.evidence_dir = evidence_dir
        self.headless = headless
        self.allow_irreversible = allow_irreversible

        self.run_id = str(uuid.uuid4())
        self.logger = RunLogger(self.run_id, "replay", evidence_dir)

        policy = Policy.from_env()
        if allow_irreversible:
            policy.irreversible_policy = "allow"
        else:
            policy.irreversible_policy = "block"
        self.guardrail = Guardrail(policy)
        self.escalator = EscalationManager(self.run_id, artifact.goal)

        self._outputs: dict[str, Any] = {}
        self._browser: Optional[BrowserManager] = None

    async def run(self) -> ReplayResult:
        self.logger.info(f"Replaying artifact: {self.artifact.id}")
        self.logger.info(f"Goal: {self.artifact.goal}")
        self.logger.info(f"Params: {self.params}")
        self.logger.info(f"Steps: {len(self.artifact.steps)}")

        self._browser = BrowserManager(
            headless=self.headless,
            evidence_dir=self.evidence_dir,
        )

        try:
            await self._browser.start()
            result = await self._execute_steps()
        except Exception as e:
            screenshot = ""
            try:
                screenshot = await self._browser.screenshot(f"hard_failure_{self.run_id[:8]}")
            except Exception:
                pass
            result = ReplayResult(
                status="hard_failure",
                error_detail=str(e),
                screenshot_path=screenshot,
                log_path=str(self.logger.log_path),
            )
        finally:
            await self._browser.stop()

        self.logger.run_end(result.status, result.to_dict())
        return result

    async def _execute_steps(self) -> ReplayResult:
        for step in self.artifact.steps:
            result = await self._execute_step_with_retry(step)
            if result is not None:
                return result  # terminal result (business outcome or hard failure)

        # All steps completed — gather outputs
        return ReplayResult(
            status="success",
            outputs=self._outputs,
            log_path=str(self.logger.log_path),
        )

    async def _execute_step_with_retry(self, step: Step) -> Optional[ReplayResult]:
        """
        Execute one step with up to MAX_RETRIES_PER_STEP retries for recoverable errors.
        Returns None if step succeeded, or a terminal ReplayResult on hard failure / business outcome.
        """
        for attempt in range(MAX_RETRIES_PER_STEP + 1):
            try:
                terminal = await self._execute_one_step(step)
                return terminal  # None = success, non-None = terminal
            except RecoverableError as e:
                self.logger.info(f"Recoverable error on step {step.seq} (attempt {attempt+1}): {e}")
                if attempt < MAX_RETRIES_PER_STEP:
                    await self._browser.wait(WAIT_BETWEEN_RETRIES_MS)
                    continue
                # Exhausted retries
                screenshot = await self._browser.screenshot(f"exhausted_retry_step{step.seq}")
                return ReplayResult(
                    status="hard_failure",
                    error_step=step.seq,
                    error_detail=f"Exhausted {MAX_RETRIES_PER_STEP} retries: {e}",
                    screenshot_path=screenshot,
                    log_path=str(self.logger.log_path),
                )
            except BusinessOutcomeError as e:
                self.logger.business_outcome(str(e), step.seq)
                return ReplayResult(
                    status="business_outcome",
                    business_outcome_signal=str(e),
                    outputs=self._outputs,
                    log_path=str(self.logger.log_path),
                )
            except HardFailureError as e:
                screenshot = await self._browser.screenshot(f"failure_step{step.seq}")
                self.logger.step_fail(step.seq, step.action, str(e), screenshot)
                return ReplayResult(
                    status="hard_failure",
                    error_step=step.seq,
                    error_detail=str(e),
                    screenshot_path=screenshot,
                    log_path=str(self.logger.log_path),
                )

    async def _execute_one_step(self, step: Step) -> Optional[ReplayResult]:
        """Execute a single step. Raises typed exceptions on error."""
        # Resolve param placeholders
        value = substitute_params(step.value, self.params) if step.value else step.value
        url = substitute_params(step.url, self.params) if step.url else step.url

        current_url = self._browser.page.url if self._browser._page else ""
        self.logger.step_start(step.seq, step.action, step.description or "")

        # ── Safety check ─────────────────────────────────────────────────────
        if step.action == "navigate" and url:
            violation = self.guardrail.check_navigate(url)
            if violation:
                raise HardFailureError(f"Policy violation: {violation.detail}")
        else:
            violation, reversibility = self.guardrail.check_action(
                step.action, current_url, value or ""
            )
            if violation:
                if reversibility == "write-irreversible" and not self.allow_irreversible:
                    raise HardFailureError(f"Policy violation: {violation.detail}")

        # ── Execute ───────────────────────────────────────────────────────────
        try:
            if step.action == "navigate":
                await self._browser.navigate(url)

            elif step.action == "fill":
                await self._browser.fill(step.locator, value)

            elif step.action == "click":
                await self._browser.click(step.locator)

            elif step.action == "select":
                await self._browser.select_option(step.locator, value)

            elif step.action == "wait":
                await self._browser.wait(step.wait_ms or 1000)

            elif step.action == "assert":
                # Pure assertion step — evaluate checkpoint inline
                if step.checkpoint:
                    ok, detail = await evaluate_checkpoint(self._browser.page, step.checkpoint)
                    if not ok:
                        raise HardFailureError(f"Assert failed at step {step.seq}: {detail}")

            elif step.action == "extract":
                # Extract text from element into outputs
                text = await self._browser.extract_text(step.locator)
                if step.output_key:
                    self._outputs[step.output_key] = text

        except Exception as e:
            err = str(e)
            # Check if this is a known business outcome signal
            if step.on_not_found == "business_outcome" and step.business_outcome_signal:
                body_has_signal = await self._browser.check_business_outcome(step.business_outcome_signal)
                if body_has_signal:
                    raise BusinessOutcomeError(step.business_outcome_signal)
            # Check if page shows a business outcome even without explicit signal
            page_text = await self._browser.get_page_text()
            for signal in ["not found", "no such member", "no results", "record not found"]:
                if signal in page_text.lower():
                    raise BusinessOutcomeError(signal)
            # Otherwise it's a locator/timeout failure
            if "timeout" in err.lower() or "locator" in err.lower():
                raise RecoverableError(err)
            raise HardFailureError(err)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if step.checkpoint:
            ok, detail = await evaluate_checkpoint(self._browser.page, step.checkpoint)
            if ok:
                self.logger.checkpoint_ok(step.seq, step.checkpoint.type)
            else:
                self.logger.checkpoint_fail(step.seq, step.checkpoint.type, detail)
                # Checkpoint failure after a write step is a hard failure
                if step.action in {"click", "fill", "select"}:
                    raise HardFailureError(f"Checkpoint failed after step {step.seq}: {detail}")
                else:
                    raise RecoverableError(f"Checkpoint failed: {detail}")

        self.logger.step_ok(step.seq, step.action)
        return None  # success, continue


# ─── Error Types ──────────────────────────────────────────────────────────────

class RecoverableError(Exception):
    """Transient error — dismiss interstitial, wait, retry."""

class BusinessOutcomeError(Exception):
    """Known non-success result — caller should handle, not crash."""

class HardFailureError(Exception):
    """Unexpected state — stop and surface a clear error."""
