from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DecisionIndependenceReport(BaseModel):
    """B4: verifies the decision (arbitration) path is independent of the LLM generation path."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-decision-independence.v1'
    checked_at: str = Field(default_factory=lambda: time.strftime('%Y-%m-%dT%H:%M:%S+08:00'))
    checks: list[dict] = Field(default_factory=list)
    overall: str = Field(pattern=r'^(PASS|FAIL)$')


def audit_decision_independence(project_root: str | Path) -> DecisionIndependenceReport:
    """Deterministically verify the arbitration path independence (B4).

    Checks (against source layout, no execution):
    1. DiagnosticDecisionGovernor exists in the investigation package.
    2. The governor's decide() reads deterministic core comparison (no LLM dependency declared).
    3. The workflow constructs the governor independently of the reasoning backend.
    """
    root = Path(project_root)
    checks: list[dict] = []
    governor_path = root / 'src/astrograph/investigation/governor.py'
    checks.append({
        'check': 'governor_module_present',
        'passed': governor_path.is_file(),
        'detail': str(governor_path.relative_to(root)) if governor_path.is_file() else 'missing investigation/governor.py',
    })
    if governor_path.is_file():
        src = governor_path.read_text(encoding='utf-8')
        has_core = 'deterministic_core' in src or 'core_comparison' in src or 'comparison_status' in src
        checks.append({'check': 'governor_uses_deterministic_core', 'passed': has_core, 'detail': 'governor references deterministic core comparison' if has_core else 'no deterministic-core reference found'})
        # The governor must not import the LLM backend.
        no_llm_import = 'openai_compatible' not in src and 'StructuredReasoningBackend' not in src
        checks.append({'check': 'governor_no_llm_dependency', 'passed': no_llm_import, 'detail': 'governor has no LLM backend import' if no_llm_import else 'governor imports LLM backend (independence violated)'})
    workflow_path = root / 'src/astrograph/orchestration/whole_spacecraft_online_workflow.py'
    if workflow_path.is_file():
        wsrc = workflow_path.read_text(encoding='utf-8')
        independent = 'self.governor = DiagnosticDecisionGovernor()' in wsrc and 'reasoning_backend' not in wsrc.split('self.governor')[1][:500] if 'self.governor' in wsrc else False
        checks.append({'check': 'workflow_governor_independent', 'passed': independent, 'detail': 'workflow builds governor independently of reasoning backend' if independent else 'governor construction may couple to reasoning backend'})
    failed = [c for c in checks if not c['passed']]
    return DecisionIndependenceReport(checks=checks, overall='FAIL' if failed else 'PASS')


__all__ = ['DecisionIndependenceReport', 'audit_decision_independence']