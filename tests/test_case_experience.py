"""D4: human-confirmed case experience library tests."""
from __future__ import annotations

import pytest

from astrograph.data_driven.case_experience import CaseExperienceLibrary, ConfirmedCase


def _case(case_id: str = 'C-1', confirmed: bool = True) -> ConfirmedCase:
    return ConfirmedCase(
        case_id=case_id, episode_id='EP-GYRO', truth_fault_id='ADCS_GYRO_BIAS_DRIFT',
        predicted_fault_id='ADCS_GYRO_BIAS_DRIFT', source_hashes=['a' * 64],
        human_confirmed=confirmed, confirmed_by='human-20260824',
    )


def test_add_requires_human_confirmation(tmp_path):
    lib = CaseExperienceLibrary(tmp_path)
    with pytest.raises(ValueError):
        lib.add(_case(confirmed=False))


def test_add_and_cases(tmp_path):
    lib = CaseExperienceLibrary(tmp_path)
    lib.add(_case())
    assert len(lib.cases()) == 1
    assert lib.cases()[0].human_confirmed


def test_duplicate_case_rejected(tmp_path):
    lib = CaseExperienceLibrary(tmp_path)
    lib.add(_case())
    with pytest.raises(ValueError, match='duplicate case'):
        lib.add(_case())


def test_invalid_source_hash_and_missing_confirmer_rejected(tmp_path):
    lib = CaseExperienceLibrary(tmp_path)
    invalid = _case()
    invalid.source_hashes = ['not-a-sha256']
    with pytest.raises(ValueError, match='SHA-256'):
        lib.add(invalid)
    missing_confirmer = _case('C-2')
    missing_confirmer.confirmed_by = ''
    with pytest.raises(ValueError, match='confirmed_by'):
        lib.add(missing_confirmer)


def test_negative_case_recorded(tmp_path):
    lib = CaseExperienceLibrary(tmp_path)
    neg = _case('C-NEG')
    neg.is_negative = True
    lib.add(neg)
    audit = lib.audit()
    assert audit['negative_samples'] == 1


def test_retrieval_excludes_negative_cases_by_default(tmp_path):
    lib = CaseExperienceLibrary(tmp_path)
    lib.add(_case('C-POS'))
    negative = _case('C-NEG')
    negative.is_negative = True
    lib.add(negative)
    retrieved = lib.retrieve(truth_fault_id='ADCS_GYRO_BIAS_DRIFT')
    assert [item.case.case_id for item in retrieved] == ['C-POS']
    assert retrieved[0].match_score == 1.0
    assert len(lib.retrieve(truth_fault_id='ADCS_GYRO_BIAS_DRIFT', include_negative=True)) == 2


def test_audit(tmp_path):
    lib = CaseExperienceLibrary(tmp_path)
    lib.add(_case('C-1'))
    lib.add(_case('C-2'))
    audit = lib.audit()
    assert audit['case_count'] == 2
    assert audit['confirmed'] == 2


def test_empty_audit(tmp_path):
    lib = CaseExperienceLibrary(tmp_path)
    audit = lib.audit()
    assert audit['case_count'] == 0
    assert audit['status'] == 'PASS'
