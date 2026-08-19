"""
agent/artifact.py — Artifact schema, validation, and store.

An artifact is a structured, portable record of a UI flow:
  - Parameterized inputs (so it can replay with different values)
  - Typed steps (navigate, fill, click, select, wait, assert)
  - Checkpoints after mutating steps (assert expected state was reached)
  - Declared outputs (what values should be extracted and returned)
  - Safety metadata (allowed domains, reversibility)

Schema version is tracked for forward compatibility.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

# ─── Step Types ──────────────────────────────────────────────────────────────

ActionType = Literal["navigate", "fill", "click", "select", "wait", "assert", "extract"]

CheckpointType = Literal[
    "element_visible",   # an element matching locator exists and is visible
    "element_not_found", # an element should NOT exist (e.g., after close)
    "url_contains",      # page URL contains a substring
    "text_contains",     # some element contains text
    "element_count",     # count of matching elements equals N
]


@dataclass
class Locator:
    """How to find a UI element. Ordered from most stable to least stable."""
    strategy: Literal["aria-label", "data-testid", "role", "text", "css", "xpath", "url"]
    value: str
    # Optional refinements for role strategy
    name: Optional[str] = None
    # Fallback locators tried in order if primary fails
    fallbacks: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {"strategy": self.strategy, "value": self.value}
        if self.name:
            d["name"] = self.name
        if self.fallbacks:
            d["fallbacks"] = self.fallbacks
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Locator":
        return cls(
            strategy=d["strategy"],
            value=d["value"],
            name=d.get("name"),
            fallbacks=d.get("fallbacks", []),
        )


@dataclass
class Checkpoint:
    """Assertion evaluated after a step to verify expected state was reached."""
    type: CheckpointType
    locator: Optional[Locator] = None
    value: Optional[str] = None   # substring / count / url fragment
    count: Optional[int] = None

    def to_dict(self) -> dict:
        d: dict = {"type": self.type}
        if self.locator:
            d["locator"] = self.locator.to_dict()
        if self.value is not None:
            d["value"] = self.value
        if self.count is not None:
            d["count"] = self.count
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Checkpoint":
        locator = Locator.from_dict(d["locator"]) if "locator" in d else None
        return cls(
            type=d["type"],
            locator=locator,
            value=d.get("value"),
            count=d.get("count"),
        )


@dataclass
class Step:
    """A single recorded action."""
    seq: int                          # 1-indexed sequence number
    action: ActionType
    # For navigate
    url: Optional[str] = None
    # For fill / select
    locator: Optional[Locator] = None
    value: Optional[str] = None       # may contain {{param}} placeholders
    # For extract: which output key to populate
    output_key: Optional[str] = None
    # For wait
    wait_ms: Optional[int] = None
    # Checkpoint evaluated after this step
    checkpoint: Optional[Checkpoint] = None
    # Metadata
    description: Optional[str] = None  # LLM's reasoning for this step
    # Error handling hints
    on_not_found: Literal["fail", "business_outcome"] = "fail"
    business_outcome_signal: Optional[str] = None  # text indicating this is a known non-success

    def to_dict(self) -> dict:
        d: dict = {
            "seq": self.seq,
            "action": self.action,
        }
        if self.url is not None:
            d["url"] = self.url
        if self.locator is not None:
            d["locator"] = self.locator.to_dict()
        if self.value is not None:
            d["value"] = self.value
        if self.output_key is not None:
            d["output_key"] = self.output_key
        if self.wait_ms is not None:
            d["wait_ms"] = self.wait_ms
        if self.checkpoint is not None:
            d["checkpoint"] = self.checkpoint.to_dict()
        if self.description:
            d["description"] = self.description
        if self.on_not_found != "fail":
            d["on_not_found"] = self.on_not_found
        if self.business_outcome_signal:
            d["business_outcome_signal"] = self.business_outcome_signal
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        return cls(
            seq=d["seq"],
            action=d["action"],
            url=d.get("url"),
            locator=Locator.from_dict(d["locator"]) if "locator" in d else None,
            value=d.get("value"),
            output_key=d.get("output_key"),
            wait_ms=d.get("wait_ms"),
            checkpoint=Checkpoint.from_dict(d["checkpoint"]) if "checkpoint" in d else None,
            description=d.get("description"),
            on_not_found=d.get("on_not_found", "fail"),
            business_outcome_signal=d.get("business_outcome_signal"),
        )


# ─── Artifact ─────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "1.0"


@dataclass
class ArtifactSafety:
    allowed_domains: list[str] = field(default_factory=list)
    reversibility: Literal["read-only", "write-reversible", "write-irreversible"] = "read-only"
    requires_confirmation: bool = False

    def to_dict(self) -> dict:
        return {
            "allowed_domains": self.allowed_domains,
            "reversibility": self.reversibility,
            "requires_confirmation": self.requires_confirmation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ArtifactSafety":
        return cls(
            allowed_domains=d.get("allowed_domains", []),
            reversibility=d.get("reversibility", "read-only"),
            requires_confirmation=d.get("requires_confirmation", False),
        )


@dataclass
class Artifact:
    """
    A portable, versioned record of a UI flow.

    Contains everything needed to replay the flow deterministically:
    - Typed, parameterized steps
    - Checkpoints for every mutating step
    - Output declarations
    - Safety policy embedded in the artifact
    """
    goal: str
    surface: Literal["web", "desktop"] = "web"
    base_url: str = ""
    steps: list[Step] = field(default_factory=list)
    params: dict[str, str] = field(default_factory=dict)      # name → type hint (e.g. "string")
    outputs: list[str] = field(default_factory=list)           # declared output keys
    safety: ArtifactSafety = field(default_factory=ArtifactSafety)
    # Auto-set fields
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    discovery_model: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "created_at": self.created_at,
            "goal": self.goal,
            "surface": self.surface,
            "base_url": self.base_url,
            "discovery_model": self.discovery_model,
            "params": self.params,
            "outputs": self.outputs,
            "steps": [s.to_dict() for s in self.steps],
            "safety": self.safety.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "Artifact":
        return cls(
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            id=d.get("id", str(uuid.uuid4())),
            created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
            goal=d["goal"],
            surface=d.get("surface", "web"),
            base_url=d.get("base_url", ""),
            discovery_model=d.get("discovery_model"),
            params=d.get("params", {}),
            outputs=d.get("outputs", []),
            steps=[Step.from_dict(s) for s in d.get("steps", [])],
            safety=ArtifactSafety.from_dict(d.get("safety", {})),
        )

    @classmethod
    def from_json(cls, text: str) -> "Artifact":
        return cls.from_dict(json.loads(text))


# ─── Artifact Store ───────────────────────────────────────────────────────────

class ArtifactStore:
    """Saves and loads artifacts from a directory as JSON files."""

    def __init__(self, directory: str = "artifacts"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, artifact: Artifact) -> Path:
        """Save artifact, return path."""
        # Use a slug of the goal + id for readability
        slug = artifact.goal[:40].lower()
        slug = "".join(c if c.isalnum() else "_" for c in slug).strip("_")
        slug = slug[:30]
        filename = f"{slug}_{artifact.id[:8]}.json"
        path = self.directory / filename
        path.write_text(artifact.to_json(), encoding="utf-8")
        return path

    def load(self, path: str) -> Artifact:
        """Load artifact from a JSON file path."""
        text = Path(path).read_text(encoding="utf-8")
        return Artifact.from_json(text)

    def list(self) -> list[Path]:
        """List all saved artifact files."""
        return sorted(self.directory.glob("*.json"))
