"""C5: capability access four-dimension assessment tests."""
from __future__ import annotations

from astrograph.data_driven.capability_access import assess_capability_access


def test_admitted_when_all_meet_minima():
    d = assess_capability_access('m', 'v1', data_coverage=0.9, robustness=0.8, explainability=0.7, governance_compliance=True)
    assert d.overall == 'ADMITTED'


def test_rejected_when_robustness_low():
    d = assess_capability_access('m', 'v1', data_coverage=0.9, robustness=0.3, explainability=0.7, governance_compliance=True)
    assert d.overall == 'REJECTED'
    assert d.dimensions['robustness']['passed'] is False


def test_rejected_when_governance_noncompliant():
    d = assess_capability_access('m', 'v1', data_coverage=0.9, robustness=0.8, explainability=0.7, governance_compliance=False)
    assert d.overall == 'REJECTED'


def test_rejected_when_coverage_low():
    d = assess_capability_access('m', 'v1', data_coverage=0.2, robustness=0.8, explainability=0.7, governance_compliance=True)
    assert d.overall == 'REJECTED'


def test_custom_minima():
    d = assess_capability_access('m', 'v1', data_coverage=0.6, robustness=0.6, explainability=0.6, governance_compliance=True, minimum_coverage=0.7)
    assert d.overall == 'REJECTED'
    assert d.dimensions['data_coverage']['passed'] is False