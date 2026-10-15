from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolPermission(str, Enum):
    ALLOW = 'ALLOW'
    ASK = 'ASK'
    DENY = 'DENY'


class ToolInvocationOutcome(str, Enum):
    DENIED = 'DENIED'
    SUCCEEDED = 'SUCCEEDED'
    FAILED = 'FAILED'


class ToolApproval(BaseModel):
    """One-use approval bound to the exact canonical invocation arguments."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    schema_version: str = 'astrograph-tool-approval.v2'
    approval_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    tool_id: str = Field(min_length=1)
    approved_args_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    approved: bool
    approved_by: str = Field(min_length=1)
    expires_at: datetime

    @model_validator(mode='after')
    def validate_expiry(self) -> 'ToolApproval':
        if self.expires_at.tzinfo is None or self.expires_at.utcoffset() is None:
            raise ValueError('expires_at must be timezone-aware')
        return self


def canonical_args_sha256(args: dict[str, Any]) -> str:
    """Hash JSON-compatible arguments with deterministic key ordering."""
    raw = json.dumps(
        args,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


class RegisteredTool(BaseModel):
    """B3: one controlled tool available to the LLM agent (S1-gated)."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-controlled-tool.v1'
    tool_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    permission: ToolPermission = ToolPermission.ALLOW
    audit_required: bool = True


class ToolInvocationRecord(BaseModel):
    """B3: auditable record of one tool invocation by an agent."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-tool-invocation.v1'
    invocation_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    tool_id: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    permitted: bool
    outcome: ToolInvocationOutcome
    failure_type: str | None = None
    approval_id: str | None = None
    invoked_at: str = Field(default_factory=lambda: time.strftime('%Y-%m-%dT%H:%M:%S+08:00'))


class ControlledToolRegistry:
    """B3: registry of S1-authorized tools with call audit (Hook/Runtime Policy)."""

    schema_version = 'astrograph-controlled-tool-registry.v1'

    def __init__(
        self,
        *,
        max_calls_per_agent: int = 20,
        approval_validator: Callable[[ToolApproval], bool] | None = None,
    ) -> None:
        self._tools: dict[str, tuple[RegisteredTool, Callable]] = {}
        self._audit: list[ToolInvocationRecord] = []
        self._calls: dict[str, int] = {}
        self._consumed_approval_ids: set[str] = set()
        self.max_calls_per_agent = max_calls_per_agent
        self._approval_validator = approval_validator

    def register(self, tool: RegisteredTool, handler: Callable) -> None:
        self._tools[tool.tool_id] = (tool, handler)

    def tools(self) -> list[RegisteredTool]:
        return [t for t, _ in self._tools.values()]

    def invoke(
        self,
        agent_id: str,
        tool_id: str,
        args: dict[str, Any] | None = None,
        *,
        approval: ToolApproval | None = None,
    ) -> dict[str, Any]:
        entry = self._tools.get(tool_id)
        args = args or {}
        if entry is None:
            record = ToolInvocationRecord(
                invocation_id=f'INV-{len(self._audit)+1}', agent_id=agent_id,
                tool_id=tool_id, args=args, permitted=False,
                outcome=ToolInvocationOutcome.DENIED,
            )
            self._audit.append(record)
            return {'permitted': False, 'reason': 'unknown-tool', 'record': record.model_dump(mode='json')}
        tool, handler = entry
        calls = self._calls.get(agent_id, 0)
        permitted = tool.permission != ToolPermission.DENY and calls < self.max_calls_per_agent
        if tool.permission == ToolPermission.ASK:
            approval_matches = (
                approval is not None
                and approval.approved
                and approval.agent_id == agent_id
                and approval.tool_id == tool_id
                and approval.approved_args_sha256 == canonical_args_sha256(args)
                and approval.expires_at > datetime.now(timezone.utc)
                and approval.approval_id not in self._consumed_approval_ids
            )
            permitted = bool(
                permitted
                and approval_matches
                and self._approval_validator is not None
                and self._approval_validator(approval)
            )
        if permitted:
            self._calls[agent_id] = calls + 1
            if tool.permission == ToolPermission.ASK and approval is not None:
                self._consumed_approval_ids.add(approval.approval_id)
        record = ToolInvocationRecord(
            invocation_id=f'INV-{len(self._audit)+1}', agent_id=agent_id, tool_id=tool_id,
            args=args, permitted=permitted,
            outcome=(ToolInvocationOutcome.SUCCEEDED if permitted else ToolInvocationOutcome.DENIED),
            approval_id=approval.approval_id if approval is not None else None,
        )
        self._audit.append(record)
        if not permitted:
            return {'permitted': False, 'reason': 'permission-or-budget-denied', 'record': record.model_dump(mode='json')}
        try:
            result = handler(args)
        except Exception as exc:
            failed = record.model_copy(
                update={
                    'outcome': ToolInvocationOutcome.FAILED,
                    'failure_type': type(exc).__name__,
                }
            )
            self._audit[-1] = failed
            raise
        return {'permitted': True, 'result': result, 'record': record.model_dump(mode='json')}

    def audit_log(self) -> list[ToolInvocationRecord]:
        return list(self._audit)

    def audit_summary(self) -> dict[str, Any]:
        return {
            'status': 'PASS',
            'tool_count': len(self._tools),
            'invocation_count': len(self._audit),
            'denied_count': sum(1 for r in self._audit if not r.permitted),
            'failed_count': sum(1 for r in self._audit if r.outcome == ToolInvocationOutcome.FAILED),
            'per_agent_calls': dict(self._calls),
        }


__all__ = [
    'ToolPermission', 'ToolInvocationOutcome', 'ToolApproval', 'canonical_args_sha256', 'RegisteredTool',
    'ToolInvocationRecord', 'ControlledToolRegistry',
]
