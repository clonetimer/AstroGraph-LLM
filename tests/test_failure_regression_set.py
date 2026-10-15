"""D2: failure-case regression set tests."""
from __future__ import annotations

import pytest

from astrograph.data_driven.failure_regression import FailureCase, FailureRegressionSet


def _case(case_id: str = 'F-1', truth: str = 'ADCS_RW_JAM', pred: str = 'ADCS_GYRO_BIAS_DRIFT') -> FailureCase:
    return FailureCase(
        case_id=case_id, episode_id='EP-1', failure_type='MISDIAGNOSIS',
        truth_fault_id=truth, predicted_fault_id=pred,
        evidence_summary='confused gyro bias with wheel jam on shared innovation band',
    )


def test_add_and_cases(tmp_path):
    s = FailureRegressionSet(tmp_path)
    s.add(_case())
    assert len(s.cases()) == 1
    assert s.cases()[0].truth_fault_id == 'ADCS_RW_JAM'


def test_replay_fixed(tmp_path):
    s = FailureRegressionSet(tmp_path)
    s.add(_case())
    result = s.replay(lambda c: c.truth_fault_id)
    assert result['case_count'] == 1
    assert result['fixed_count'] == 1
    assert result['fix_rate'] == 1.0


def test_replay_still_failing(tmp_path):
    s = FailureRegressionSet(tmp_path)
    s.add(_case())
    result = s.replay(lambda c: 'WRONG')
    assert result['fixed_count'] == 0
    assert result['fix_rate'] == 0.0


def test_empty_set_replay(tmp_path):
    s = FailureRegressionSet(tmp_path)
    result = s.replay(lambda c: c.truth_fault_id)
    assert result['case_count'] == 0
    assert result['fix_rate'] == 1.0


def test_failure_type_enum(tmp_path):
    with pytest.raises(Exception):
        FailureCase(case_id='F-x', episode_id='E', failure_type='UNKNOWN', truth_fault_id='A', predicted_fault_id='B')


def test_append_only(tmp_path):
    s = FailureRegressionSet(tmp_path)
    s.add(_case('F-1'))
    s.add(_case('F-2'))
    assert len(s.cases()) == 2