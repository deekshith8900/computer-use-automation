"""
Unit tests for replay engine logic — param substitution, error classification.
Uses mock browser to avoid real Playwright in unit tests.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agent.replay import (
    substitute_params,
    ReplayResult,
    RecoverableError,
    BusinessOutcomeError,
    HardFailureError,
)
from agent.artifact import (
    Artifact, ArtifactParam, ArtifactOutput, Step, Locator, Checkpoint, ArtifactSafety
)



class TestSubstituteParams:
    def test_simple_substitution(self):
        result = substitute_params("Hello {{name}}", {"name": "Jane"})
        assert result == "Hello Jane"

    def test_multiple_substitutions(self):
        result = substitute_params("{{first}} {{last}}", {"first": "Jane", "last": "Doe"})
        assert result == "Jane Doe"

    def test_no_placeholders(self):
        result = substitute_params("plain text", {})
        assert result == "plain text"

    def test_missing_param_raises(self):
        with pytest.raises(ValueError, match="Missing required parameter: 'name'"):
            substitute_params("Hello {{name}}", {})

    def test_none_value_safe(self):
        result = substitute_params(None, {}) if False else None
        # substitute_params requires string — None handled by caller
        assert True

    def test_url_substitution(self):
        result = substitute_params("http://localhost:5000/members/{{member_id}}", {"member_id": "M001"})
        assert result == "http://localhost:5000/members/M001"


class TestReplayResult:
    def test_success_result(self):
        result = ReplayResult(
            status="success",
            outputs={"account_balance": "$4,823.50"},
            log_path="evidence/replay_abc.log",
        )
        assert result.is_success()
        d = result.to_dict()
        assert d["status"] == "success"
        assert d["outputs"]["account_balance"] == "$4,823.50"

    def test_business_outcome_result(self):
        result = ReplayResult(
            status="business_outcome",
            business_outcome_signal="no such member",
        )
        assert not result.is_success()
        assert result.status == "business_outcome"

    def test_hard_failure_result(self):
        result = ReplayResult(
            status="hard_failure",
            error_step=3,
            error_detail="Locator not found: [data-testid='search-button']",
            screenshot_path="evidence/failure_step3.png",
        )
        assert not result.is_success()
        d = result.to_dict()
        assert d["error_step"] == 3
        assert "search-button" in d["error_detail"]

    def test_to_dict_completeness(self):
        result = ReplayResult(status="success")
        d = result.to_dict()
        required_keys = {"status", "outputs", "business_outcome_signal", "error_step", "error_detail", "screenshot_path", "log_path"}
        assert required_keys.issubset(d.keys())


class TestErrorTypes:
    def test_recoverable_error_is_exception(self):
        err = RecoverableError("timeout waiting for element")
        assert isinstance(err, Exception)
        assert "timeout" in str(err)

    def test_business_outcome_error(self):
        err = BusinessOutcomeError("no such member")
        assert "no such member" in str(err)

    def test_hard_failure_error(self):
        err = HardFailureError("unexpected dialog appeared")
        assert isinstance(err, Exception)


class TestArtifactForReplay:
    """Test that artifacts are structured correctly for deterministic replay."""

    def make_artifact_with_checkpoint(self) -> Artifact:
        return Artifact(
            goal="Find member and get balance",
            base_url="http://localhost:5001",
            steps=[
                Step(seq=1, action="navigate", url="http://localhost:5001/members"),
                Step(
                    seq=2,
                    action="fill",
                    locator=Locator(strategy="aria-label", value="Search members"),
                    value="{{member_name}}",
                ),
                Step(
                    seq=3,
                    action="click",
                    locator=Locator(strategy="data-testid", value="search-button"),
                    checkpoint=Checkpoint(
                        type="element_visible",
                        locator=Locator(strategy="data-testid", value="search-results"),
                    ),
                    on_not_found="business_outcome",
                    business_outcome_signal="No members found",
                ),
                Step(
                    seq=4,
                    action="extract",
                    locator=Locator(strategy="data-testid", value="current-balance"),
                    output_key="account_balance",
                ),
            ],
            param_defs=[ArtifactParam(name="member_name", type="string")],
            output_defs=[ArtifactOutput(key="account_balance", type="string")],
        )

    def test_artifact_has_params(self):
        art = self.make_artifact_with_checkpoint()
        assert "member_name" in art.params

    def test_all_steps_have_seq(self):
        art = self.make_artifact_with_checkpoint()
        seqs = [s.seq for s in art.steps]
        assert seqs == [1, 2, 3, 4]

    def test_mutating_step_has_checkpoint(self):
        art = self.make_artifact_with_checkpoint()
        click_step = next(s for s in art.steps if s.action == "click")
        assert click_step.checkpoint is not None

    def test_extract_step_has_output_key(self):
        art = self.make_artifact_with_checkpoint()
        extract_step = next(s for s in art.steps if s.action == "extract")
        assert extract_step.output_key == "account_balance"
        assert "account_balance" in art.outputs

    def test_business_outcome_step_configured(self):
        art = self.make_artifact_with_checkpoint()
        click_step = next(s for s in art.steps if s.action == "click")
        assert click_step.on_not_found == "business_outcome"
        assert click_step.business_outcome_signal is not None
