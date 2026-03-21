"""Authority levels and escalation policy for the PureClaw mesh."""

from __future__ import annotations

from enum import Enum


class AuthorityLevel(Enum):
    """What a Claw is authorized to do without user approval."""
    ALERT_ONLY = "alert_only"   # Observe and report, no action
    AUTO_FIX = "auto_fix"       # Execute pre-approved remediation
    ESCALATE = "escalate"       # Must ask Prime (-> user) before acting
    FULL = "full"               # Unrestricted (Prime only)

    @classmethod
    def from_str(cls, s: str) -> AuthorityLevel:
        try:
            return cls(s)
        except ValueError:
            return cls.ESCALATE  # safe default


# Default action-to-authority mapping per Claw role.
# Keys are action categories, values are minimum authority required.
DEFAULT_AUTHORITY_MATRIX = {
    "infra": {
        "service_restart": AuthorityLevel.AUTO_FIX,
        "gpu_management": AuthorityLevel.AUTO_FIX,
        "ipmi_command": AuthorityLevel.AUTO_FIX,
        "fan_adjust": AuthorityLevel.AUTO_FIX,
        "config_change": AuthorityLevel.ESCALATE,
        "network_change": AuthorityLevel.ESCALATE,
        "data_deletion": AuthorityLevel.ESCALATE,
    },
    "ops": {
        "ceph_rebalance": AuthorityLevel.AUTO_FIX,
        "osd_recovery": AuthorityLevel.AUTO_FIX,
        "scrub_schedule": AuthorityLevel.AUTO_FIX,
        "backup_verify": AuthorityLevel.AUTO_FIX,
        "pool_change": AuthorityLevel.ESCALATE,
        "data_deletion": AuthorityLevel.ESCALATE,
        "config_change": AuthorityLevel.ESCALATE,
    },
    "sentinel": {
        "alert_classify": AuthorityLevel.ALERT_ONLY,
        "alert_route": AuthorityLevel.ALERT_ONLY,
        "status_check": AuthorityLevel.ALERT_ONLY,
    },
}


def check_authority(
    claw_id: str,
    claw_authority: AuthorityLevel,
    action: str,
) -> tuple[bool, str]:
    """Check if a Claw is authorized for an action.

    Returns (allowed, reason).
    """
    matrix = DEFAULT_AUTHORITY_MATRIX.get(claw_id, {})
    required = matrix.get(action)

    if required is None:
        # Unknown action -- default to escalate unless Claw has FULL authority
        if claw_authority == AuthorityLevel.FULL:
            return True, "full authority"
        return False, f"Unknown action '{action}' requires escalation"

    # Authority hierarchy: FULL > AUTO_FIX > ESCALATE > ALERT_ONLY
    hierarchy = [AuthorityLevel.ALERT_ONLY, AuthorityLevel.ESCALATE, AuthorityLevel.AUTO_FIX, AuthorityLevel.FULL]
    claw_rank = hierarchy.index(claw_authority)
    required_rank = hierarchy.index(required)

    if claw_rank >= required_rank:
        return True, f"authorized ({claw_authority.value} >= {required.value})"
    return False, f"insufficient authority ({claw_authority.value} < {required.value})"
