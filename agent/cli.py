"""
agent/cli.py — CLI entry points.

  python -m agent discover --goal "..." --url http://localhost:5000
  python -m agent replay --artifact artifacts/xxx.json [--params '{"member_name": "Jane"}']
  python -m agent list
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.json import JSON

load_dotenv()
console = Console()


@click.group()
def cli():
    """Computer-Use Automation Agent CLI."""
    pass


# ─── discover ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--goal", "-g", required=True, help="Natural language goal for the agent.")
@click.option("--url", "-u", default="http://localhost:5000", show_default=True, help="Base URL of the target app.")
@click.option("--evidence-dir", default="evidence", show_default=True)
@click.option("--artifact-dir", default="artifacts", show_default=True)
@click.option("--headless/--no-headless", default=False, show_default=True)
def discover(goal: str, url: str, evidence_dir: str, artifact_dir: str, headless: bool):
    """Run LLM-driven discovery on a live web app and save an artifact."""
    from agent.discovery import DiscoveryEngine

    console.print(f"\n[bold cyan]DISCOVER[/bold cyan] — {goal}")
    console.print(f"  URL: {url}")
    console.print(f"  LLM: {os.environ.get('LLM_PROVIDER', 'openai')}/{os.environ.get('LLM_MODEL', 'gpt-4o')}\n")

    async def _run():
        engine = DiscoveryEngine(
            goal=goal,
            base_url=url,
            evidence_dir=evidence_dir,
            artifact_dir=artifact_dir,
            headless=headless,
        )
        return await engine.run()

    artifact = asyncio.run(_run())

    console.print(f"\n[bold green]✓ Discovery complete![/bold green]")
    console.print(f"  Artifact ID: {artifact.id}")
    console.print(f"  Steps recorded: {len(artifact.steps)}")
    console.print(f"  Params: {list(artifact.params.keys())}")
    console.print(f"  Outputs: {artifact.outputs}")
    console.print()

    # Print the saved path
    from agent.artifact import ArtifactStore
    store = ArtifactStore(artifact_dir)
    paths = store.list()
    latest = max(paths, key=lambda p: p.stat().st_mtime) if paths else None
    if latest:
        console.print(f"  Saved: [bold]{latest}[/bold]")
        console.print(f"\n  To replay:")
        console.print(f"  [dim]python -m agent replay --artifact {latest}[/dim]")


# ─── replay ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--artifact", "-a", required=True, help="Path to artifact JSON file.")
@click.option("--params", "-p", default="{}", help='JSON string of parameters, e.g. \'{"member_name": "Jane"}\'')
@click.option("--evidence-dir", default="evidence", show_default=True)
@click.option("--headless/--no-headless", default=False, show_default=True)
@click.option("--allow-irreversible", is_flag=True, default=False,
              help="Allow irreversible actions (transfers, confirmations). Disabled by default for safety.")
def replay(artifact: str, params: str, evidence_dir: str, headless: bool, allow_irreversible: bool):
    """Replay a recorded artifact deterministically."""
    from agent.artifact import ArtifactStore
    from agent.replay import ReplayEngine

    # Load artifact
    try:
        store = ArtifactStore()
        art = store.load(artifact)
    except Exception as e:
        console.print(f"[red]Failed to load artifact: {e}[/red]")
        sys.exit(1)

    # Parse params
    try:
        param_dict = json.loads(params)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid params JSON: {e}[/red]")
        sys.exit(1)

    console.print(f"\n[bold green]REPLAY[/bold green] — {art.goal}")
    console.print(f"  Artifact: {artifact}")
    console.print(f"  Steps: {len(art.steps)}")
    console.print(f"  Params: {param_dict}\n")

    async def _run():
        engine = ReplayEngine(
            artifact=art,
            params=param_dict,
            evidence_dir=evidence_dir,
            headless=headless,
            allow_irreversible=allow_irreversible,
        )
        return await engine.run()

    result = asyncio.run(_run())

    # Display result
    status_color = {
        "success": "green",
        "business_outcome": "blue",
        "recoverable_error": "yellow",
        "hard_failure": "red",
    }.get(result.status, "white")

    console.print(f"\n[bold {status_color}]■ {result.status.upper()}[/bold {status_color}]")

    if result.status == "success":
        console.print(f"  Outputs:")
        for k, v in result.outputs.items():
            console.print(f"    {k}: [bold]{v}[/bold]")
    elif result.status == "business_outcome":
        console.print(f"  Signal: {result.business_outcome_signal}")
        console.print(f"  (This is a known outcome, not a crash. The caller should handle it.)")
    elif result.status in ("recoverable_error", "hard_failure"):
        console.print(f"  Step: {result.error_step}")
        console.print(f"  Error: {result.error_detail}")
        if result.screenshot_path:
            console.print(f"  Screenshot: {result.screenshot_path}")

    console.print(f"\n  Log: {result.log_path}")


# ─── list ─────────────────────────────────────────────────────────────────────

@cli.command("list")
@click.option("--artifact-dir", default="artifacts", show_default=True)
def list_artifacts(artifact_dir: str):
    """List all saved artifacts."""
    from agent.artifact import ArtifactStore

    store = ArtifactStore(artifact_dir)
    paths = store.list()

    if not paths:
        console.print("[yellow]No artifacts found.[/yellow]")
        return

    table = Table(title="Saved Artifacts", show_header=True, header_style="bold cyan")
    table.add_column("File", style="dim")
    table.add_column("Goal")
    table.add_column("Steps")
    table.add_column("Params")
    table.add_column("Created")

    for path in paths:
        try:
            art = store.load(str(path))
            table.add_row(
                path.name,
                art.goal[:50] + ("..." if len(art.goal) > 50 else ""),
                str(len(art.steps)),
                ", ".join(art.params.keys()) or "—",
                art.created_at[:19],
            )
        except Exception as e:
            table.add_row(path.name, f"[red]Error: {e}[/red]", "?", "?", "?")

    console.print(table)


# ─── resolve intervention ──────────────────────────────────────────────────────

@cli.command()
@click.argument("intervention_id")
@click.option("--notes", default="", help="Human notes about what was done.")
def resolve(intervention_id: str, notes: str):
    """Signal that a human intervention has been completed (resume automation)."""
    from agent.escalation import EscalationManager

    manager = EscalationManager(run_id="cli", artifact_goal="")
    try:
        manager.resolve(intervention_id, notes)
        console.print(f"[green]✓ Intervention {intervention_id[:8]} resolved.[/green]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    cli()


if __name__ == "__main__":
    main()
