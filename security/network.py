"""Network egress control — SSRF protection and domain allowlisting.

Validates URLs before HTTP requests. Blocks RFC 1918, loopback, and
link-local addresses. Matches hostnames against allow/deny domain lists.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

from security.policy import get_policy, matches_domain

log = logging.getLogger("nexus.security")


def check_url_access(url: str) -> tuple[bool, str]:
    """Check if a URL is allowed by network policy.

    Parses URL, resolves hostname to IP, checks against private ranges
    and domain allow/deny lists. Deny takes precedence.

    Returns (allowed: bool, reason: str).
    """
    policy = get_policy().network

    # Parse URL
    try:
        parsed = urlparse(url)
    except Exception:
        return False, f"Malformed URL: {url}"

    hostname = parsed.hostname
    if not hostname:
        return False, "URL has no hostname"

    port = parsed.port

    # Check deny domains first (takes precedence)
    if matches_domain(hostname, policy.fetch_deny_domains):
        return False, f"Domain '{hostname}' matches fetch_deny_domains policy"

    # Check allow domains
    if not matches_domain(hostname, policy.fetch_allow_domains):
        return False, f"Domain '{hostname}' not in fetch_allow_domains policy"

    # SSRF protection: block private/reserved IP ranges
    if policy.block_private_ranges:
        blocked, reason = _check_private_ip(hostname)
        if blocked:
            return False, reason

    return True, "allowed"


def _check_private_ip(hostname: str) -> tuple[bool, str]:
    """Resolve hostname and check if it points to a private/reserved IP.

    Returns (blocked: bool, reason: str).
    """
    # Check if hostname is already an IP address
    try:
        addr = ipaddress.ip_address(hostname)
        if _is_blocked_ip(addr):
            return True, f"IP {hostname} is in a blocked range (private/loopback/link-local)"
        return False, ""
    except ValueError:
        pass  # Not an IP literal, need to resolve

    # DNS resolution
    try:
        results = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        # Can't resolve — allow (the HTTP request will fail naturally)
        return False, ""

    for family, _, _, _, sockaddr in results:
        ip_str = sockaddr[0]
        try:
            addr = ipaddress.ip_address(ip_str)
            if _is_blocked_ip(addr):
                return True, f"Domain '{hostname}' resolves to blocked IP {ip_str} (private/loopback/link-local)"
        except ValueError:
            continue

    return False, ""


def _is_blocked_ip(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Check if an IP address is in a blocked range."""
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
    )
