"""Unit tests for artifact schema, serialization, and round-trip."""

import json
import tempfile
import pytest
from pathlib import Path

from agent.artifact import (
    Artifact, Step, Locator, Checkpoint, ArtifactSafety, ArtifactStore
)


def make_sample_artifact() -> Artifact:
    return Artifact(
        goal="Find member Jane Doe and return her balance",
        surface="web",
        base_url="http://localhost:5000",
        steps=[
            Step(
                seq=1,
                action="navigate",
                url="http://localhost:5000/members",
                description="Go to member search",
            ),
            Step(
                seq=2,
                action="fill",
                locator=Locator(strategy="aria-label", value="Search members"),
                value="{{member_name}}",
                description="Enter member name",
            ),
            Step(
                seq=3,
                action="click",
                locator=Locator(strategy="data-testid", value="search-button"),
                checkpoint=Checkpoint(type="element_visible", locator=Locator(strategy="data-testid", value="search-results")),
                description="Submit search",
            ),
            Step(
                seq=4,
                action="extract",
                locator=Locator(strategy="data-testid", value="current-balance"),
                output_key="account_balance",
                description="Extract account balance",
            ),
        ],
        params={"member_name": "string"},
        outputs=["account_balance"],
        safety=ArtifactSafety(
            allowed_domains=["localhost:5000"],
            reversibility="read-only",
        ),
        discovery_model="openai/gpt-4o",
    )


class TestLocator:
    def test_round_trip(self):
        loc = Locator(strategy="aria-label", value="Search members")
        d = loc.to_dict()
        restored = Locator.from_dict(d)
        assert restored.strategy == loc.strategy
        assert restored.value == loc.value

    def test_with_fallbacks(self):
        loc = Locator(
            strategy="aria-label",
            value="Search members",
            fallbacks=[{"strategy": "data-testid", "value": "search-input"}],
        )
        d = loc.to_dict()
        assert "fallbacks" in d
        restored = Locator.from_dict(d)
        assert len(restored.fallbacks) == 1

    def test_role_with_name(self):
        loc = Locator(strategy="role", value="button", name="Search")
        d = loc.to_dict()
        assert d["name"] == "Search"
        restored = Locator.from_dict(d)
        assert restored.name == "Search"


class TestCheckpoint:
    def test_url_contains(self):
        cp = Checkpoint(type="url_contains", value="/members/")
        d = cp.to_dict()
        assert d["type"] == "url_contains"
        assert d["value"] == "/members/"
        restored = Checkpoint.from_dict(d)
        assert restored.type == cp.type
        assert restored.value == cp.value

    def test_element_visible_with_locator(self):
        cp = Checkpoint(
            type="element_visible",
            locator=Locator(strategy="data-testid", value="search-results"),
        )
        d = cp.to_dict()
        assert "locator" in d
        restored = Checkpoint.from_dict(d)
        assert restored.locator.strategy == "data-testid"


class TestStep:
    def test_navigate_round_trip(self):
        step = Step(seq=1, action="navigate", url="http://localhost:5000")
        d = step.to_dict()
        restored = Step.from_dict(d)
        assert restored.seq == 1
        assert restored.action == "navigate"
        assert restored.url == "http://localhost:5000"

    def test_fill_with_param_placeholder(self):
        step = Step(
            seq=2,
            action="fill",
            locator=Locator(strategy="aria-label", value="Search members"),
            value="{{member_name}}",
        )
        d = step.to_dict()
        restored = Step.from_dict(d)
        assert restored.value == "{{member_name}}"

    def test_click_with_checkpoint(self):
        step = Step(
            seq=3,
            action="click",
            locator=Locator(strategy="data-testid", value="search-button"),
            checkpoint=Checkpoint(type="url_contains", value="/members/"),
        )
        d = step.to_dict()
        restored = Step.from_dict(d)
        assert restored.checkpoint is not None
        assert restored.checkpoint.type == "url_contains"

    def test_on_not_found_defaults(self):
        step = Step(seq=1, action="navigate", url="http://x.com")
        assert step.on_not_found == "fail"
        d = step.to_dict()
        # Default is not serialized (to keep files clean)
        assert "on_not_found" not in d

    def test_business_outcome_step(self):
        step = Step(
            seq=4,
            action="click",
            locator=Locator(strategy="data-testid", value="search-button"),
            on_not_found="business_outcome",
            business_outcome_signal="no results",
        )
        d = step.to_dict()
        restored = Step.from_dict(d)
        assert restored.on_not_found == "business_outcome"
        assert restored.business_outcome_signal == "no results"


class TestArtifact:
    def test_full_round_trip(self):
        art = make_sample_artifact()
        d = art.to_dict()
        restored = Artifact.from_dict(d)

        assert restored.goal == art.goal
        assert restored.surface == art.surface
        assert restored.base_url == art.base_url
        assert len(restored.steps) == len(art.steps)
        assert restored.params == art.params
        assert restored.outputs == art.outputs
        assert restored.safety.reversibility == art.safety.reversibility
        assert restored.discovery_model == art.discovery_model

    def test_json_round_trip(self):
        art = make_sample_artifact()
        json_str = art.to_json()
        restored = Artifact.from_json(json_str)
        assert restored.id == art.id
        assert len(restored.steps) == 4

    def test_schema_version_set(self):
        art = make_sample_artifact()
        assert art.schema_version == "1.0"

    def test_id_auto_generated(self):
        art1 = Artifact(goal="test", base_url="http://x.com")
        art2 = Artifact(goal="test", base_url="http://x.com")
        assert art1.id != art2.id

    def test_empty_artifact(self):
        art = Artifact(goal="empty", base_url="http://x.com")
        restored = Artifact.from_json(art.to_json())
        assert restored.steps == []
        assert restored.params == {}
        assert restored.outputs == []


class TestArtifactStore:
    def test_save_and_load(self):
        art = make_sample_artifact()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(tmpdir)
            path = store.save(art)
            assert path.exists()
            restored = store.load(str(path))
            assert restored.id == art.id
            assert len(restored.steps) == len(art.steps)

    def test_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(tmpdir)
            art1 = Artifact(goal="first", base_url="http://x.com")
            art2 = Artifact(goal="second", base_url="http://x.com")
            store.save(art1)
            store.save(art2)
            paths = store.list()
            assert len(paths) == 2

    def test_filename_slug(self):
        art = Artifact(goal="Find member Jane Doe", base_url="http://x.com")
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(tmpdir)
            path = store.save(art)
            assert "find_member_jane_doe" in path.name
