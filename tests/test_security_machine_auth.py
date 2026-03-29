"""Tests for security.machine_auth."""

from pathlib import Path
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from security.machine_auth import sign_payload, verify_headers


def test_sign_and_verify_round_trip():
    body = b'{"ok":true}'
    headers = sign_payload(body, 'secret', timestamp='1700000000')
    with patch('security.machine_auth.time.time', return_value=1700000000):
        allowed, reason = verify_headers(body, headers, 'secret')
    assert allowed is True
    assert reason == 'ok'


def test_verify_rejects_missing_secret():
    allowed, reason = verify_headers(b'{}', {}, '')
    assert allowed is False
    assert 'not configured' in reason


def test_verify_rejects_stale_timestamp():
    body = b'{}'
    headers = sign_payload(body, 'secret', timestamp='100')
    with patch('security.machine_auth.time.time', return_value=1000):
        allowed, reason = verify_headers(body, headers, 'secret', max_skew_secs=10)
    assert allowed is False
    assert reason == 'stale timestamp'
