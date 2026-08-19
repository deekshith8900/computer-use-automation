"""
agent/safety.py — Policy guardrail enforcement.

Every step is checked against the policy BEFORE execution.
The policy encodes:
  1. Domain allowlist — agent may not navigate outside approved domains
  2. Reversibility — read-only vs. write-reversible vs. write-irreversible steps
  3. PII/secret redaction — values are scrubbed before persisting to artifacts/logs
  4. Risky action gating — irreversible steps are blocked or flagged

Design choice: fail-closed. If unsure, block and surface a clear error.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse


# ─── Policy Config ────────────────────────────────────────────────────────────

@dataclass
class Policy:
    """
    Configurable safety policy.
    Loaded from environment or artifact safety metadata.
    """
    allowed_domains: list[str] = field(default_factory=list)
    # Action types that are allowed at all
    allowed_actions: set[str] = field(default_factory=lambda: {
        "navigate", "fill", "click", "select", "wait", "assert", "extract"
    })
    # How to handle irreversible (write-irreversible) actions
    # "block" — never allow | "confirm" — require flag | "allow" — allow with logging
    irreversible_policy: str = "block"

    @classmethod
    def from_env(cls) -> "Policy":
        domains_env = os.environ.get("ALLOWED_DOMAINS", "localhost:5000")
        domains = [d.strip() for d in domains_env.split(",") if d.strip()]
        return cls(allowed_domains=domains)

    @classmethod
    def from_artifact_safety(cls, safety_dict: dict, env_override: bool = True) -> "Policy":
        base = cls.from_env() if env_override else cls()
        artifact_domains = safety_dict.get("allowed_domains", [])
        # Intersection of artifact domains and env-allowed domains (stricter wins)
        if env_override and base.allowed_domains:
            domains = [d for d in artifact_domains if d in base.allowed_domains] or base.allowed_domains
        else:
            domains = artifact_domains or base.allowed_domains
        return cls(allowed_domains=domains)


# ─── Policy Violations ────────────────────────────────────────────────────────

@dataclass
class PolicyViolation:
    reason: str
    action: str
    detail: str


# ─── PII Redaction ────────────────────────────────────────────────────────────

# Patterns for values that should never appear in artifacts or logs
_REDACT_PATTERNS = [
    (re.compile(r'Bearer\s+[A-Za-z0-9\-._~+/]+=*', re.I), "Bearer [REDACTED]"),
    (re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}'), "[EMAIL_REDACTED]"),
    (re.compile(r'\b(?:\d[ -]?){13,16}\b'), "[CARD_REDACTED]"),           # credit card numbers
    (re.compile(r'\bsk-[A-Za-z0-9]{20,}'), "[API_KEY_REDACTED]"),          # OpenAI-style keys
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), "[SSN_REDACTED]"),              # SSN
]

# Field names whose values should always be redacted
_REDACT_FIELD_NAMES = {"password", "passwd", "secret", "token", "api_key", "apikey", "credential"}


def redact(value: str, field_name: str = "") -> str:
    """Redact PII and secrets from a string value."""
    if field_name.lower() in _REDACT_FIELD_NAMES:
        return "[REDACTED]"
    result = value
    for pattern, replacement in _REDACT_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def redact_step_value(value: Optional[str], field_name: str = "") -> Optional[str]:
    """Redact a step's fill/value field before persisting."""
    if value is None:
        return None
    return redact(value, field_name)


# ─── Reversibility ────────────────────────────────────────────────────────────

# Actions that write/mutate state — used to determine reversibility
_WRITE_ACTIONS = {"click", "fill", "select"}
_LIKELY_IRREVERSIBLE_PATTERNS = [
    # URL patterns that suggest irreversible commits
    re.compile(r"/transfer", re.I),
    re.compile(r"/confirm", re.I),
    re.compile(r"/execute", re.I),
    re.compile(r"/submit", re.I),
    re.compile(r"/delete", re.I),
    re.compile(r"/payment", re.I),
]

def classify_reversibility(action: str, url: str = "", value: str = "") -> str:
    """
    Classify an action as:
    - 'read-only': navigate/assert/extract
    - 'write-reversible': fill/select a form field (undoable by navigating away)
    - 'write-irreversible': confirm/submit a transfer, delete, payment
    """
    if action in {"navigate", "assert", "extract", "wait"}:
        return "read-only"
    if action in {"fill", "select"}:
        return "write-reversible"
    if action == "click":
        # Check if we're on a page that suggests commitment
        for pattern in _LIKELY_IRREVERSIBLE_PATTERNS:
            if pattern.search(url):
                return "write-irreversible"
        # Button labels that suggest commitment
        if value and re.search(r'\b(confirm|execute|transfer|submit|delete|pay)\b', value, re.I):
            return "write-irreversible"
        return "write-reversible"
    return "read-only"


# ─── Guardrail ────────────────────────────────────────────────────────────────

class Guardrail:
    """
    Enforces the policy before each step executes.

    Raises PolicyViolation if blocked.
    Returns a reversibility classification for logging.
    """

    def __init__(self, policy: Policy):
        self.policy = policy

    def check_navigate(self, url: str) -> Optional[PolicyViolation]:
        """Verify the target URL is within the allowed domain list."""
        if not self.policy.allowed_domains:
            return None  # No restriction configured
        parsed = urlparse(url)
        host = parsed.netloc or parsed.path  # handle bare hostnames
        for allowed in self.policy.allowed_domains:
            if host == allowed or host.endswith("." + allowed):
                return None
        return PolicyViolation(
            reason="domain_not_allowed",
            action="navigate",
            detail=f"URL '{url}' is not in the allowed domain list: {self.policy.allowed_domains}",
        )

    def check_action(
        self,
        action: str,
        current_url: str = "",
        value: str = "",
        explicit_reversibility: str = "",
    ) -> tuple[Optional[PolicyViolation], str]:
        """
        Check an action against the policy.
        Returns (violation_or_None, reversibility_classification).
        """
        if action not in self.policy.allowed_actions:
            return PolicyViolation(
                reason="action_not_allowed",
                action=action,
                detail=f"Action '{action}' is not in the allowed action set.",
            ), "unknown"

        rev = explicit_reversibility or classify_reversibility(action, current_url, value)

        if rev == "write-irreversible":
            if self.policy.irreversible_policy == "block":
                return PolicyViolation(
                    reason="irreversible_action_blocked",
                    action=action,
                    detail=(
                        f"Action '{action}' on '{current_url}' classified as write-irreversible. "
                        f"Policy: block. Set irreversible_policy='allow' or 'confirm' to permit."
                    ),
                ), rev
            # "confirm" or "allow" — return None (no hard block) but flag it
        return None, rev
