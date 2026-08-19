"""
agent/logger.py — Structured run logger.

Produces a JSONL log file alongside a human-readable console output.
Each log entry records: step sequence, action, outcome, timing, screenshots.

The goal is to produce enough evidence to understand and debug any run.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.text import Text

console = Console(stderr=True)


class RunLogger:
    """
    Writes structured JSONL logs to evidence/ for both discovery and replay runs.
    Also streams a human-readable summary to stderr via Rich.
    """

    def __init__(self, run_id: str, run_type: str, evidence_dir: str = "evidence"):
        self.run_id = run_id
        self.run_type = run_type  # "discovery" | "replay"
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.log_path = self.evidence_dir / f"{run_type}_{run_id[:8]}_{timestamp}.log"
        self._entries: list[dict] = []
        self._start_time = time.monotonic()

        self._write({
            "event": "run_start",
            "run_id": run_id,
            "run_type": run_type,
            "timestamp": self._now(),
        })
        color = "cyan" if run_type == "discovery" else "green"
        console.print(f"\n[bold {color}]▶ {run_type.upper()} RUN[/bold {color}] — {run_id[:8]}")

    # ── Public API ────────────────────────────────────────────────────────────

    def step_start(self, seq: int, action: str, description: str = "") -> None:
        elapsed = self._elapsed()
        self._write({
            "event": "step_start",
            "seq": seq,
            "action": action,
            "description": description,
            "elapsed_s": elapsed,
            "timestamp": self._now(),
        })
        console.print(f"  [dim]step {seq:02d}[/dim] [bold white]{action}[/bold white]"
                      + (f" — {description}" if description else ""))

    def step_ok(self, seq: int, action: str, details: dict | None = None) -> None:
        self._write({
            "event": "step_ok",
            "seq": seq,
            "action": action,
            "details": details or {},
            "elapsed_s": self._elapsed(),
            "timestamp": self._now(),
        })
        console.print(f"       [green]✓[/green] step {seq:02d} completed")

    def step_fail(self, seq: int, action: str, error: str, screenshot_path: str = "") -> None:
        self._write({
            "event": "step_fail",
            "seq": seq,
            "action": action,
            "error": error,
            "screenshot": screenshot_path,
            "elapsed_s": self._elapsed(),
            "timestamp": self._now(),
        })
        console.print(f"       [red]✗[/red] step {seq:02d} failed: {error}")

    def checkpoint_ok(self, seq: int, checkpoint_type: str) -> None:
        self._write({
            "event": "checkpoint_ok",
            "seq": seq,
            "checkpoint_type": checkpoint_type,
            "elapsed_s": self._elapsed(),
            "timestamp": self._now(),
        })
        console.print(f"       [green]✓[/green] checkpoint [{checkpoint_type}] passed")

    def checkpoint_fail(self, seq: int, checkpoint_type: str, detail: str) -> None:
        self._write({
            "event": "checkpoint_fail",
            "seq": seq,
            "checkpoint_type": checkpoint_type,
            "detail": detail,
            "elapsed_s": self._elapsed(),
            "timestamp": self._now(),
        })
        console.print(f"       [yellow]⚠[/yellow] checkpoint [{checkpoint_type}] failed: {detail}")

    def llm_call(self, model: str, prompt_tokens: int, completion_tokens: int, tool_calls: list[str]) -> None:
        self._write({
            "event": "llm_call",
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tool_calls": tool_calls,
            "elapsed_s": self._elapsed(),
            "timestamp": self._now(),
        })
        console.print(f"  [dim cyan]LLM[/dim cyan] {model} → tools: {tool_calls}")

    def escalation(self, step_seq: int, reason: str, screenshot_path: str = "") -> None:
        self._write({
            "event": "escalation",
            "step_seq": step_seq,
            "reason": reason,
            "screenshot": screenshot_path,
            "elapsed_s": self._elapsed(),
            "timestamp": self._now(),
        })
        console.print(f"  [bold yellow]⚑ ESCALATION[/bold yellow] step {step_seq} — {reason}")

    def human_resumed(self, step_seq: int, notes: str = "") -> None:
        self._write({
            "event": "human_resumed",
            "step_seq": step_seq,
            "notes": notes,
            "elapsed_s": self._elapsed(),
            "timestamp": self._now(),
        })
        console.print(f"  [bold green]↩ RESUMED[/bold green] by human at step {step_seq}" +
                      (f" — {notes}" if notes else ""))

    def business_outcome(self, signal: str, step_seq: int) -> None:
        self._write({
            "event": "business_outcome",
            "signal": signal,
            "step_seq": step_seq,
            "elapsed_s": self._elapsed(),
            "timestamp": self._now(),
        })
        console.print(f"  [bold blue]ℹ BUSINESS OUTCOME[/bold blue] — {signal}")

    def run_end(self, status: str, result: dict) -> None:
        elapsed = self._elapsed()
        self._write({
            "event": "run_end",
            "status": status,
            "result": result,
            "elapsed_s": elapsed,
            "timestamp": self._now(),
        })
        color_map = {"success": "green", "business_outcome": "blue",
                     "recoverable_error": "yellow", "hard_failure": "red"}
        color = color_map.get(status, "white")
        console.print(f"\n[bold {color}]■ {status.upper()}[/bold {color}] in {elapsed:.1f}s")
        console.print(f"  log → {self.log_path}")

    def info(self, message: str, **kwargs) -> None:
        self._write({"event": "info", "message": message, **kwargs, "timestamp": self._now()})
        console.print(f"  [dim]{message}[/dim]")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _elapsed(self) -> float:
        return round(time.monotonic() - self._start_time, 3)

    def _write(self, entry: dict) -> None:
        self._entries.append(entry)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def get_entries(self) -> list[dict]:
        return list(self._entries)
