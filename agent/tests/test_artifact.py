"""Unit tests for artifact schema, serialization, validation, and round-trip."""

import json
import tempfile
import pytest
from pathlib import Path

from agent.artifact import (
    Artifact, ArtifactParam, ArtifactOutput, ArtifactSafety,
    Step, Locator, Checkpoint, ArtifactStore, SCHEMA_VERSION,
)


def make_sample_artifact() -> Artifact:
    return Artifact(
        goal="Find member Jane Doe and return her balance",
        surface="web",
        base_url="http://localhost:5001",
        steps=[
            Step(
                seq=1,
                action="navigate",
                url="http://localhost:5001/members",
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
                locator=Locator(strategy="data-testid", value="search-btn"),
                checkpoint=Checkpoint(
                    type="element_visible",
                    locator=Locator(strategy="data-testid", value="search-results"),
                ),
                description="Submit search",
            ),
            Step(
                seq=4,
                action="extract",
                locator=Locator(strategy="data-testid", value="total-balance"),
                output_key="account_balance",
                description="Extract account balance",
            ),
        ],
        param_defs=[
            ArtifactParam(name="member_name", type="string", required=True,
                          description="Member name to search for"),
        ],
        output_defs=[
            ArtifactOutput(key="account_balance", type="string",
                           description="Total balance on member detail page"),
        ],
        safety=ArtifactSafety(
            allowed_domains=["localhost:5001"],
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
        step = Step(seq=1, action="navigate", url="http://localhost:5001")
        d = step.to_dict()
        restored = Step.from_dict(d)
        assert restored.seq == 1
        assert restored.action == "navigate"
        assert restored.url == "http://localhost:5001"

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
            locator=Locator(strategy="data-testid", value="search-btn"),
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
            locator=Locator(strategy="data-testid", value="search-btn"),
            on_not_found="business_outcome",
            business_outcome_signal="no results",
        )
        d = step.to_dict()
        restored = Step.from_dict(d)
        assert restored.on_not_found == "business_outcome"
        assert restored.business_outcome_signal == "no results"


class TestArtifactParamOutput:
    def test_param_round_trip(self):
        p = ArtifactParam(name="member_name", type="string", required=True,
                          description="Full name")
        d = p.to_dict()
        restored = ArtifactParam.from_dict(d)
        assert restored.name == "member_name"
        assert restored.type == "string"
        assert restored.required is True
        assert restored.description == "Full name"

    def test_output_round_trip(self):
        o = ArtifactOutput(key="account_balance", type="string", description="Balance")
        d = o.to_dict()
        restored = ArtifactOutput.from_dict(d)
        assert restored.key == "account_balance"
        assert restored.type == "string"

    def test_legacy_param_compat(self):
        """from_dict with legacy string format should produce ArtifactParam."""
        p = ArtifactParam.from_dict("string")
        assert p.type == "string"

    def test_legacy_output_compat(self):
        """from_dict with legacy string key should produce ArtifactOutput."""
        o = ArtifactOutput.from_dict("account_balance")
        assert o.key == "account_balance"


class TestArtifact:
    def test_full_round_trip(self):
        art = make_sample_artifact()
        d = art.to_dict()
        restored = Artifact.from_dict(d)

        assert restored.goal == art.goal
        assert restored.surface == art.surface
        assert restored.base_url == art.base_url
        assert len(restored.steps) == len(art.steps)
        # Check via legacy compat properties
        assert restored.params == art.params
        assert restored.outputs == art.outputs
        # Check typed defs
        assert len(restored.param_defs) == 1
        assert restored.param_defs[0].name == "member_name"
        assert len(restored.output_defs) == 1
        assert restored.output_defs[0].key == "account_balance"
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
        assert art.schema_version == SCHEMA_VERSION

    def test_id_auto_generated(self):
        art1 = Artifact(goal="test", base_url="http://x.com")
        art2 = Artifact(goal="test", base_url="http://x.com")
        assert art1.id != art2.id

    def test_empty_artifact_params_outputs(self):
        art = Artifact(goal="empty", base_url="http://x.com")
        restored = Artifact.from_json(art.to_json())
        assert restored.steps == []
        assert restored.params == {}
        assert restored.outputs == []

    def test_legacy_params_dict_deserialized(self):
        """Artifacts saved with old schema (params: {name: type}) still load."""
        old_dict = {
            "schema_version": "1.0",
            "id": "test-id",
            "created_at": "2026-01-01T00:00:00+00:00",
            "goal": "test goal",
            "base_url": "http://localhost:5001",
            "params": {"member_name": "string"},
            "outputs": ["account_balance"],
            "steps": [
                {"seq": 1, "action": "navigate", "url": "http://localhost:5001"},
            ],
            "safety": {"allowed_domains": [], "reversibility": "read-only",
                       "requires_confirmation": False},
        }
        art = Artifact.from_dict(old_dict)
        assert art.params == {"member_name": "string"}
        assert art.outputs == ["account_balance"]
        assert len(art.param_defs) == 1
        assert art.param_defs[0].name == "member_name"

    def test_params_legacy_property(self):
        art = make_sample_artifact()
        assert art.params == {"member_name": "string"}

    def test_outputs_legacy_property(self):
        art = make_sample_artifact()
        assert art.outputs == ["account_balance"]


class TestArtifactValidation:
    def test_valid_artifact_no_errors(self):
        art = make_sample_artifact()
        errors = art.validate()
        assert errors == [], f"Unexpected errors: {errors}"

    def test_missing_goal_fails(self):
        art = Artifact(goal="", base_url="http://x.com",
                       steps=[Step(seq=1, action="navigate", url="http://x.com")])
        errors = art.validate()
        assert any("goal" in e for e in errors)

    def test_missing_base_url_fails(self):
        art = Artifact(goal="test", base_url="",
                       steps=[Step(seq=1, action="navigate", url="http://x.com")])
        errors = art.validate()
        assert any("base_url" in e for e in errors)

    def test_empty_steps_fails(self):
        art = Artifact(goal="test", base_url="http://x.com")
        errors = art.validate()
        assert any("at least one step" in e for e in errors)

    def test_fill_without_locator_fails(self):
        art = Artifact(
            goal="test", base_url="http://x.com",
            steps=[Step(seq=1, action="fill", value="hello")],  # no locator
        )
        errors = art.validate()
        assert any("locator" in e for e in errors)

    def test_navigate_without_url_fails(self):
        art = Artifact(
            goal="test", base_url="http://x.com",
            steps=[Step(seq=1, action="navigate")],  # no url
        )
        errors = art.validate()
        assert any("url" in e for e in errors)

    def test_extract_without_output_key_fails(self):
        art = Artifact(
            goal="test", base_url="http://x.com",
            steps=[Step(
                seq=1, action="extract",
                locator=Locator(strategy="css", value=".balance"),
                # no output_key
            )],
        )
        errors = art.validate()
        assert any("output_key" in e for e in errors)

    def test_duplicate_seq_fails(self):
        art = Artifact(
            goal="test", base_url="http://x.com",
            steps=[
                Step(seq=1, action="navigate", url="http://x.com"),
                Step(seq=1, action="navigate", url="http://x.com/2"),  # dup seq
            ],
        )
        errors = art.validate()
        assert any("duplicate" in e for e in errors)

    def test_declared_output_without_extract_step_fails(self):
        art = Artifact(
            goal="test", base_url="http://x.com",
            steps=[Step(seq=1, action="navigate", url="http://x.com")],
            output_defs=[ArtifactOutput(key="balance")],
        )
        errors = art.validate()
        assert any("balance" in e and "extract" in e for e in errors)

    def test_checkpoint_element_visible_without_locator_fails(self):
        art = Artifact(
            goal="test", base_url="http://x.com",
            steps=[Step(
                seq=1, action="navigate", url="http://x.com",
                checkpoint=Checkpoint(type="element_visible"),  # no locator
            )],
        )
        errors = art.validate()
        assert any("locator" in e for e in errors)

    def test_store_save_validates(self):
        """ArtifactStore.save() raises ValueError for invalid artifacts."""
        art = Artifact(goal="test", base_url="http://x.com")  # no steps
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(tmpdir)
            with pytest.raises(ValueError, match="validation failed"):
                store.save(art)


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
            art1 = Artifact(
                goal="first", base_url="http://x.com",
                steps=[Step(seq=1, action="navigate", url="http://x.com")],
            )
            art2 = Artifact(
                goal="second", base_url="http://x.com",
                steps=[Step(seq=1, action="navigate", url="http://x.com")],
            )
            store.save(art1)
            store.save(art2)
            paths = store.list()
            assert len(paths) == 2

    def test_filename_slug(self):
        art = Artifact(
            goal="Find member Jane Doe", base_url="http://x.com",
            steps=[Step(seq=1, action="navigate", url="http://x.com")],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(tmpdir)
            path = store.save(art)
            assert "find_member_jane_doe" in path.name
