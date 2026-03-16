"""Tests for security.network — SSRF protection, domain allow/deny, private IP blocking."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

with patch.dict("os.environ", {
    "TELEGRAM_BOT_TOKEN": "fake:token",
    "AUTHORIZED_USER_ID": "12345",
}):
    from security.policy import _parse_policy
    from security.network import check_url_access, _check_private_ip, _is_blocked_ip
    import security.policy as _pol

_NET_POLICY = _parse_policy({
    "version": 1,
    "network": {
        "fetch_allow_domains": ["*.example.com", "api.github.com"],
        "fetch_deny_domains": ["evil.example.com"],
        "block_private_ranges": True,
    },
})


@pytest.fixture(autouse=True)
def _set_policy():
    saved = _pol._current_policy
    _pol._current_policy = _NET_POLICY
    yield
    _pol._current_policy = saved


class TestUrlAccess:

    def test_allowed_domain(self):
        allowed, _ = check_url_access("https://api.github.com/repos")
        assert allowed

    def test_allowed_subdomain(self):
        allowed, _ = check_url_access("https://docs.example.com/page")
        assert allowed

    def test_denied_domain(self):
        allowed, reason = check_url_access("https://evil.example.com/")
        assert not allowed
        assert "fetch_deny_domains" in reason

    def test_not_in_allowlist(self):
        allowed, reason = check_url_access("https://random-site.com/")
        assert not allowed
        assert "not in fetch_allow_domains" in reason

    def test_no_hostname(self):
        allowed, reason = check_url_access("not-a-url")
        assert not allowed

    def test_empty_url(self):
        allowed, reason = check_url_access("")
        assert not allowed


class TestSSRFProtection:

    def test_localhost_blocked(self):
        with patch("security.network.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [(2, 1, 6, "", ("127.0.0.1", 0))]
            allowed, reason = check_url_access("https://docs.example.com/")
            assert not allowed
            assert "blocked" in reason.lower() or "loopback" in reason.lower()

    def test_private_ip_blocked(self):
        with patch("security.network.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [(2, 1, 6, "", ("192.168.1.1", 0))]
            allowed, reason = check_url_access("https://docs.example.com/")
            assert not allowed
            assert "blocked" in reason.lower() or "private" in reason.lower()

    def test_link_local_blocked(self):
        with patch("security.network.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [(2, 1, 6, "", ("169.254.1.1", 0))]
            allowed, reason = check_url_access("https://docs.example.com/")
            assert not allowed

    def test_ip_literal_blocked(self):
        allowed, reason = check_url_access("http://127.0.0.1:8080/admin")
        # 127.0.0.1 not in allow list and is private
        assert not allowed

    def test_public_ip_allowed(self):
        with patch("security.network.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [(2, 1, 6, "", ("1.2.3.4", 0))]
            allowed, _ = check_url_access("https://docs.example.com/page")
            assert allowed

    def test_dns_failure_allowed(self):
        """Can't resolve → allow (HTTP request will fail naturally)."""
        import socket
        with patch("security.network.socket.getaddrinfo", side_effect=socket.gaierror("NXDOMAIN")):
            allowed, _ = check_url_access("https://docs.example.com/page")
            assert allowed


class TestWildcardPolicy:

    def test_wildcard_allows_all(self):
        import security.policy as _pol
        saved = _pol._current_policy
        _pol._current_policy = _parse_policy({
            "version": 1,
            "network": {
                "fetch_allow_domains": ["*"],
                "block_private_ranges": False,
            },
        })
        allowed, _ = check_url_access("https://anything.com/")
        assert allowed
        _pol._current_policy = saved

    def test_wildcard_with_private_blocking(self):
        import security.policy as _pol
        saved = _pol._current_policy
        _pol._current_policy = _parse_policy({
            "version": 1,
            "network": {
                "fetch_allow_domains": ["*"],
                "block_private_ranges": True,
            },
        })
        # Public should pass
        with patch("security.network.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [(2, 1, 6, "", ("8.8.8.8", 0))]
            allowed, _ = check_url_access("https://example.com/")
            assert allowed
        _pol._current_policy = saved


class TestBlockedIP:

    def test_private_v4(self):
        import ipaddress
        assert _is_blocked_ip(ipaddress.ip_address("192.168.1.1"))
        assert _is_blocked_ip(ipaddress.ip_address("10.0.0.1"))
        assert _is_blocked_ip(ipaddress.ip_address("172.16.0.1"))

    def test_loopback(self):
        import ipaddress
        assert _is_blocked_ip(ipaddress.ip_address("127.0.0.1"))

    def test_link_local(self):
        import ipaddress
        assert _is_blocked_ip(ipaddress.ip_address("169.254.1.1"))

    def test_public_not_blocked(self):
        import ipaddress
        assert not _is_blocked_ip(ipaddress.ip_address("8.8.8.8"))
        assert not _is_blocked_ip(ipaddress.ip_address("1.1.1.1"))
