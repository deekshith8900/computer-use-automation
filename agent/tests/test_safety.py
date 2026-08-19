"""Unit tests for safety guardrails — allowlist, PII redaction, reversibility."""

import pytest
from agent.safety import (
    Policy, Guardrail, PolicyViolation,
    redact, redact_step_value, classify_reversibility,
)


class TestRedact:
    def test_redacts_email(self):
        result = redact("Contact jane.doe@example.com for help")
        assert "jane.doe@example.com" not in result
        assert "[EMAIL_REDACTED]" in result

    def test_redacts_api_key(self):
        result = redact("My key is sk-abcdefghijklmnopqrstuvwxyz123456")
        assert "sk-" not in result

    def test_redacts_by_field_name(self):
        result = redact_step_value("super_secret_password", field_name="password")
        assert result == "[REDACTED]"

    def test_redacts_ssn(self):
        result = redact("SSN: 123-45-6789")
        assert "123-45-6789" not in result

    def test_preserves_normal_text(self):
        text = "Search for Jane Doe in the member list"
        result = redact(text)
        assert result == text

    def test_none_value_safe(self):
        assert redact_step_value(None) is None

    def test_template_params_preserved(self):
        # Parameterized values like {{member_name}} should pass through
        result = redact("{{member_name}}")
        assert result == "{{member_name}}"


class TestReversibility:
    def test_navigate_is_readonly(self):
        assert classify_reversibility("navigate", "http://localhost:5000") == "read-only"

    def test_assert_is_readonly(self):
        assert classify_reversibility("assert") == "read-only"

    def test_extract_is_readonly(self):
        assert classify_reversibility("extract") == "read-only"

    def test_fill_is_write_reversible(self):
        assert classify_reversibility("fill") == "write-reversible"

    def test_select_is_write_reversible(self):
        assert classify_reversibility("select") == "write-reversible"

    def test_click_on_normal_page_is_reversible(self):
        rev = classify_reversibility("click", "http://localhost:5000/members", "Search")
        assert rev == "write-reversible"

    def test_click_on_transfer_page_is_irreversible(self):
        rev = classify_reversibility("click", "http://localhost:5000/transfer", "Confirm Transfer")
        assert rev == "write-irreversible"

    def test_click_confirm_button_is_irreversible(self):
        rev = classify_reversibility("click", "http://localhost:5000/anything", "Confirm")
        assert rev == "write-irreversible"

    def test_click_delete_is_irreversible(self):
        rev = classify_reversibility("click", "http://localhost:5000/delete", "Delete")
        assert rev == "write-irreversible"


class TestDomainAllowlist:
    def setup_method(self):
        self.policy = Policy(allowed_domains=["localhost:5000"])
        self.guardrail = Guardrail(self.policy)

    def test_allowed_domain_passes(self):
        violation = self.guardrail.check_navigate("http://localhost:5000/members")
        assert violation is None

    def test_disallowed_domain_blocked(self):
        violation = self.guardrail.check_navigate("http://evil.com/steal-data")
        assert violation is not None
        assert violation.reason == "domain_not_allowed"

    def test_subdomain_of_allowed_passes(self):
        policy = Policy(allowed_domains=["example.com"])
        guardrail = Guardrail(policy)
        violation = guardrail.check_navigate("http://api.example.com/v1")
        assert violation is None

    def test_no_allowlist_permits_all(self):
        policy = Policy(allowed_domains=[])
        guardrail = Guardrail(policy)
        violation = guardrail.check_navigate("http://anywhere.com")
        assert violation is None

    def test_localhost_5001_blocked_when_not_in_list(self):
        violation = self.guardrail.check_navigate("http://localhost:5001/evil")
        assert violation is not None


class TestActionPolicy:
    def setup_method(self):
        self.policy = Policy(
            allowed_domains=["localhost:5000"],
            irreversible_policy="block",
        )
        self.guardrail = Guardrail(self.policy)

    def test_fill_allowed(self):
        violation, rev = self.guardrail.check_action("fill", "http://localhost:5000/members")
        assert violation is None
        assert rev == "write-reversible"

    def test_click_on_normal_page_allowed(self):
        violation, rev = self.guardrail.check_action("click", "http://localhost:5000/members", "Search")
        assert violation is None

    def test_irreversible_click_blocked_by_default(self):
        violation, rev = self.guardrail.check_action(
            "click", "http://localhost:5000/transfer", "Confirm Transfer"
        )
        assert violation is not None
        assert violation.reason == "irreversible_action_blocked"
        assert rev == "write-irreversible"

    def test_irreversible_click_allowed_when_policy_permits(self):
        policy = Policy(
            allowed_domains=["localhost:5000"],
            irreversible_policy="allow",
        )
        guardrail = Guardrail(policy)
        violation, rev = guardrail.check_action(
            "click", "http://localhost:5000/transfer", "Confirm Transfer"
        )
        assert violation is None
        assert rev == "write-irreversible"

    def test_unknown_action_blocked(self):
        violation, rev = self.guardrail.check_action("execute_code", "http://localhost:5000")
        assert violation is not None
        assert violation.reason == "action_not_allowed"
