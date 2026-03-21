"""ClawMessage — the inter-agent message envelope for the PureClaw mesh."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass
class ClawMessage:
    """Message envelope for inter-Claw communication."""

    msg_type: str               # alert, task, escalation, report, query, ack
    payload: dict               # type-specific data
    from_claw: str = ""         # sender ID (e.g. "prime", "infra", "ops", "sentinel")
    to_claw: str = ""           # target ID or "broadcast"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    priority: int = 1           # 0=routine, 1=normal, 2=high, 3=critical
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reply_to: str | None = None # message ID this replies to
    ttl_seconds: int = 3600     # expiry

    # Valid message types
    TYPES = ("alert", "task", "escalation", "report", "query", "ack")

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []
        if self.msg_type not in self.TYPES:
            errors.append(f"Invalid msg_type: {self.msg_type}")
        if not self.payload:
            errors.append("Empty payload")
        if not self.from_claw:
            errors.append("Missing from_claw")
        return errors

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> ClawMessage:
        # Filter to only known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_json(cls, raw: str) -> ClawMessage:
        return cls.from_dict(json.loads(raw))

    def is_expired(self) -> bool:
        try:
            sent = datetime.fromisoformat(self.timestamp)
            age = (datetime.now(timezone.utc) - sent).total_seconds()
            return age > self.ttl_seconds
        except Exception:
            return False
