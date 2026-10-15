"""B4: decision-path independence audit tests."""
from __future__ import annotations

from pathlib import Path

from astrograph.governance.decision_independence import audit_decision_independence

ROOT = Path(__file__).resolve().parents[1]


def test_real_repo_passes_independence():
    report = audit_decision_independence(ROOT)
    assert report.overall == 'PASS'
    by_name = {c['check']: c['passed'] for c in report.checks}
    assert by_name['governor_module_present']
    assert by_name['governor_uses_deterministic_core']
    assert by_name['governor_no_llm_dependency']


def test_missing_governor_fails(tmp_path):
    report = audit_decision_independence(tmp_path)
    assert report.overall == 'FAIL'
    assert any(c['check'] == 'governor_module_present' and not c['passed'] for c in report.checks)


def test_llm_dependency_detected(tmp_path):
    (tmp_path / 'src/astrograph/investigation').mkdir(parents=True)
    (tmp_path / 'src/astrograph/investigation/governor.py').write_text(
        'from astrograph.llm.openai_compatible import OpenAICompatibleStructuredBackend\n', encoding='utf-8',
    )
    report = audit_decision_independence(tmp_path)
    assert any(c['check'] == 'governor_no_llm_dependency' and not c['passed'] for c in report.checks)