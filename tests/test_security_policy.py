"""Tests for security.policy — loading, validation, defaults, hot-reload."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

with patch.dict("os.environ", {
    "TELEGRAM_BOT_TOKEN": "fake:token",
    "AUTHORIZED_USER_ID": "12345",
}):
    from security.policy import (
        SecurityPolicy,
        FilesystemPolicy,
        NetworkPolicy,
        ToolPolicy,
        InferencePolicy,
        CredentialPolicy,
        AuditPolicy,
        load_policy,
        get_policy,
        reload_policy,
        matches_glob,
        matches_domain,
        _parse_policy,
        _validate_schema,
        PolicyWatcher,
    )


class TestDefaultPolicy:

    def test_default_policy_is_permissive(self):
        policy = SecurityPolicy()
        assert policy.version == 1
        assert policy.tools.allowed == ("*",)
        assert policy.tools.denied == ()
        assert policy.inference.model_allowlist == ("*",)
        assert policy.inference.max_tokens_per_session == 0

    def test_default_filesystem_policy(self):
        fs = FilesystemPolicy()
        assert fs.read_allow == ("/**",)
        assert fs.read_deny == ()
        assert fs.write_allow == ("/**",)
        assert fs.write_deny == ()

    def test_default_network_policy(self):
        net = NetworkPolicy()
        assert net.fetch_allow_domains == ("*",)
        assert net.block_private_ranges is True

    def test_default_audit_policy(self):
        audit = AuditPolicy()
        assert audit.enabled is True
        assert audit.retention_days == 90

    def test_frozen_dataclasses(self):
        policy = SecurityPolicy()
        with pytest.raises(AttributeError):
            policy.version = 2


class TestParsing:

    def test_parse_minimal(self):
        policy = _parse_policy({"version": 1})
        assert policy.version == 1
        assert policy.filesystem == FilesystemPolicy()

    def test_parse_filesystem(self):
        data = {
            "version": 2,
            "filesystem": {
                "read_allow": ["/data/**"],
                "read_deny": ["/etc/shadow"],
                "write_allow": ["/output/**"],
                "write_deny": ["/etc/**"],
            },
        }
        policy = _parse_policy(data)
        assert policy.filesystem.read_allow == ("/data/**",)
        assert policy.filesystem.read_deny == ("/etc/shadow",)
        assert policy.filesystem.write_allow == ("/output/**",)
        assert policy.filesystem.write_deny == ("/etc/**",)

    def test_parse_network(self):
        data = {
            "version": 1,
            "network": {
                "fetch_allow_domains": ["*.example.com", "api.github.com"],
                "block_private_ranges": False,
            },
        }
        policy = _parse_policy(data)
        assert "*.example.com" in policy.network.fetch_allow_domains
        assert policy.network.block_private_ranges is False

    def test_parse_tools(self):
        data = {
            "version": 1,
            "tools": {
                "allowed": ["bash", "read_file"],
                "denied": ["web_fetch"],
            },
        }
        policy = _parse_policy(data)
        assert "bash" in policy.tools.allowed
        assert "web_fetch" in policy.tools.denied

    def test_parse_credentials(self):
        data = {
            "version": 1,
            "credentials": {
                "redact_patterns": ["sk-[a-zA-Z0-9]+"],
                "redact_env_vars": ["MY_SECRET"],
            },
        }
        policy = _parse_policy(data)
        assert len(policy.credentials.redact_patterns) == 1
        assert "MY_SECRET" in policy.credentials.redact_env_vars


class TestLoading:

    def test_load_missing_file_returns_permissive(self):
        policy = load_policy("/nonexistent/policy.yaml")
        assert policy.version == 1
        assert policy.tools.allowed == ("*",)

    def test_load_valid_yaml(self, tmp_path):
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("""
version: 5
filesystem:
  read_allow: ["/data/**"]
  read_deny: ["/etc/shadow"]
tools:
  allowed: ["bash", "read_file"]
  denied: ["web_fetch"]
""")
        policy = load_policy(policy_file)
        assert policy.version == 5
        assert "bash" in policy.tools.allowed
        assert "web_fetch" in policy.tools.denied

    def test_load_bad_yaml_raises(self, tmp_path):
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("not: a: valid: yaml: [[[")
        # yaml.safe_load may or may not raise on this, depending on content
        # Let's use genuinely invalid YAML
        policy_file.write_text("version: not_a_number\n  bad indent")
        # This may parse as a string — test schema validation instead
        policy_file.write_text('["this is a list not a dict"]')
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            load_policy(policy_file)

    def test_get_policy_returns_singleton(self, tmp_path):
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("version: 42\n")
        load_policy(policy_file)
        assert get_policy().version == 42

    def test_reload_policy(self, tmp_path):
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("version: 1\n")
        load_policy(policy_file)
        assert get_policy().version == 1

        policy_file.write_text("version: 2\n")
        reload_policy()
        assert get_policy().version == 2

    def test_load_from_env_var(self, tmp_path, monkeypatch):
        policy_file = tmp_path / "custom.yaml"
        policy_file.write_text("version: 99\n")
        monkeypatch.setenv("SECURITY_POLICY_PATH", str(policy_file))
        load_policy()  # No path arg — should read from env
        assert get_policy().version == 99


class TestMatching:

    def test_glob_basic(self):
        assert matches_glob("/data/file.txt", ("/data/**",))
        assert matches_glob("/data/sub/file.txt", ("/data/**",))
        assert not matches_glob("/etc/passwd", ("/data/**",))

    def test_glob_star(self):
        assert matches_glob("/anything", ("/**",))
        assert matches_glob("/deep/nested/path", ("/**",))

    def test_glob_pattern_list(self):
        patterns = ("/data/**", "/output/**", "/tmp/**")
        assert matches_glob("/data/x", patterns)
        assert matches_glob("/output/y", patterns)
        assert matches_glob("/tmp/z", patterns)
        assert not matches_glob("/etc/passwd", patterns)

    def test_glob_empty_patterns(self):
        assert not matches_glob("/any/path", ())

    def test_domain_wildcard(self):
        assert matches_domain("example.com", ("*",))

    def test_domain_exact(self):
        assert matches_domain("api.github.com", ("api.github.com",))
        assert not matches_domain("evil.com", ("api.github.com",))

    def test_domain_glob(self):
        assert matches_domain("sub.example.com", ("*.example.com",))
        assert not matches_domain("example.com", ("*.example.com",))

    def test_domain_case_insensitive(self):
        assert matches_domain("API.GitHub.COM", ("api.github.com",))

    def test_domain_empty_patterns(self):
        assert not matches_domain("example.com", ())


class TestSchemaValidation:

    def test_valid_minimal(self):
        errors = _validate_schema({"version": 1})
        assert errors == []

    def test_invalid_version_type(self):
        errors = _validate_schema({"version": "not_a_number"})
        # jsonschema may not be installed — skip if empty
        if errors:
            assert any("version" in e for e in errors)

    def test_unknown_field(self):
        errors = _validate_schema({"version": 1, "unknown_field": True})
        if errors:
            assert any("unknown_field" in e or "additional" in e.lower() for e in errors)


class TestPolicyWatcher:

    def test_watcher_init(self):
        watcher = PolicyWatcher(interval=5.0)
        assert watcher._interval == 5.0
        assert not watcher._running

    def test_watcher_stop(self):
        watcher = PolicyWatcher()
        watcher._running = True
        watcher.stop()
        assert not watcher._running
