"""B3: controlled tool registry (constrained Agentic) tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from astrograph.agents.controlled_tools import (
    ControlledToolRegistry,
    RegisteredTool,
    ToolApproval,
    ToolInvocationOutcome,
    ToolPermission,
    canonical_args_sha256,
)


def _registry(*, validate_approval=False) -> ControlledToolRegistry:
    validator = (lambda approval: approval.approval_id == 'APPROVED-1') if validate_approval else None
    r = ControlledToolRegistry(max_calls_per_agent=3, approval_validator=validator)
    r.register(RegisteredTool(tool_id='kg.query', description='query knowledge graph'), lambda a: {'nodes': 2})
    r.register(RegisteredTool(tool_id='evidence.fetch', description='fetch evidence', permission=ToolPermission.ASK), lambda a: {'facts': 5})
    r.register(RegisteredTool(tool_id='config.write', description='write config', permission=ToolPermission.DENY), lambda a: {})
    return r


def _approval(*, agent_id='agent-1', args=None, expires_delta=timedelta(hours=1)):
    return ToolApproval(
        approval_id='APPROVED-1', agent_id=agent_id, tool_id='evidence.fetch',
        approved_args_sha256=canonical_args_sha256(args or {}),
        approved=True, approved_by='human-reviewer',
        expires_at=datetime.now(timezone.utc) + expires_delta,
    )


def test_allow_tool_invoked():
    r = _registry()
    result = r.invoke('agent-1', 'kg.query', {})
    assert result['permitted'] is True
    assert result['result'] == {'nodes': 2}


def test_deny_tool_blocked():
    r = _registry()
    result = r.invoke('agent-1', 'config.write', {})
    assert result['permitted'] is False


def test_ask_tool_requires_human():
    r = _registry()
    denied = r.invoke('agent-1', 'evidence.fetch', {})
    assert denied['permitted'] is False
    spoofed = r.invoke('agent-1', 'evidence.fetch', {'human_approved': True})
    assert spoofed['permitted'] is False

    r = _registry(validate_approval=True)
    approval = _approval()
    approved = r.invoke('agent-1', 'evidence.fetch', {}, approval=approval)
    assert approved['permitted'] is True


def test_ask_tool_rejects_approval_for_another_agent():
    r = _registry(validate_approval=True)
    approval = _approval(agent_id='agent-2')
    assert r.invoke('agent-1', 'evidence.fetch', {}, approval=approval)['permitted'] is False


def test_ask_approval_is_bound_to_args_expiry_and_one_use():
    r = _registry(validate_approval=True)
    approval = _approval(args={'case_id': 'C1'})
    assert r.invoke('agent-1', 'evidence.fetch', {'case_id': 'C2'}, approval=approval)['permitted'] is False
    assert r.invoke('agent-1', 'evidence.fetch', {'case_id': 'C1'}, approval=approval)['permitted'] is True
    assert r.invoke('agent-1', 'evidence.fetch', {'case_id': 'C1'}, approval=approval)['permitted'] is False

    expired = _approval(expires_delta=timedelta(seconds=-1))
    assert r.invoke('agent-1', 'evidence.fetch', {}, approval=expired)['permitted'] is False


def test_handler_failure_is_audited_without_leaking_exception_message():
    r = ControlledToolRegistry()
    r.register(RegisteredTool(tool_id='bad', description='fails'), lambda args: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        r.invoke('agent-1', 'bad', {'secret': 'not-an-error-message'})
    record = r.audit_log()[-1]
    assert record.outcome == ToolInvocationOutcome.FAILED
    assert record.failure_type == 'ZeroDivisionError'
    assert r.audit_summary()['failed_count'] == 1


def test_budget_enforced():
    r = _registry()
    for _ in range(3):
        r.invoke('agent-1', 'kg.query', {})
    fourth = r.invoke('agent-1', 'kg.query', {})
    assert fourth['permitted'] is False
    assert 'budget' in fourth['reason']


def test_unknown_tool_denied_and_audited():
    r = _registry()
    result = r.invoke('agent-1', 'nope', {})
    assert result['permitted'] is False
    assert result['reason'] == 'unknown-tool'


def test_audit_summary():
    r = _registry()
    r.invoke('agent-1', 'kg.query', {})
    r.invoke('agent-1', 'config.write', {})
    summary = r.audit_summary()
    assert summary['invocation_count'] == 2
    assert summary['denied_count'] == 1
