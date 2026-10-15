from copy import deepcopy
import pytest
from astrograph.oag.complementary import mechanism_confirmation_candidate

PACKAGE = '6ee2ab138a1001e00db509796eef68df6540c93fb457f543aa6bd9a24ca74148'


def decision():
    return {'status': 'CANDIDATE_SET', 'detector_mechanism_disagreement': True,
            'blocking_conflicts': [], 'candidate_fault_ids': ['ADCS_RW_JAM'],
            'data_candidate_source': 'PRE_HIERARCHY_CLASS_DISTRIBUTION', 'data_candidate_count': 2,
            'support_by_fault': {'ADCS_RW_JAM': {'complete_mechanism_support': True, 'data_candidate': True}}}


def test_pinned_complete_mechanism_requires_raw_class_agreement():
    assert mechanism_confirmation_candidate(decision(), evidence_package_sha256=PACKAGE) == 'ADCS_RW_JAM'
    d = decision()
    d['support_by_fault']['ADCS_RW_JAM']['data_candidate'] = False
    assert mechanism_confirmation_candidate(d, evidence_package_sha256=PACKAGE) is None


@pytest.mark.parametrize('change', [
    {'status': 'NORMAL_REFERENCE_MATCH'}, {'status': 'BLOCKED'},
    {'detector_mechanism_disagreement': False}, {'blocking_conflicts': ['CONFLICT']},
    {'candidate_fault_ids': ['ADCS_RW_JAM', 'ADCS_RW_MOTOR_FAILURE']},
    {'candidate_fault_ids': ['ADCS_RW_MOTOR_FAILURE']},
    {'data_candidate_source': 'EXISTING_CLASS_RANKING'}, {'data_candidate_count': 3},
    {'support_by_fault': {'ADCS_RW_JAM': {'complete_mechanism_support': False, 'data_candidate': True}}},
])
def test_policy_does_not_become_blanket_detector_override(change):
    d = deepcopy(decision())
    d.update(change)
    assert mechanism_confirmation_candidate(d, evidence_package_sha256=PACKAGE) is None


def test_unreviewed_evidence_package_cannot_confirm():
    assert mechanism_confirmation_candidate(decision(), evidence_package_sha256='0'*64) is None
