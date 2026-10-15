"""A5: independent output verification (generation-evaluation separation) tests."""
from __future__ import annotations

from astrograph.governance.output_verifier import VerificationVerdict, verify_hypothesis_set


def _valid_output() -> dict:
    return {
        'episode_id': 'EP-1',
        'hypotheses': [
            {'hypothesis_id': 'H-1', 'target_fault_id': 'ADCS_RW_JAM', 'status': 'SUPPORTED',
             'supporting_evidence_fact_ids': ['EF-1', 'EF-2']},
            {'hypothesis_id': 'H-2', 'target_fault_id': 'ADCS_GYRO_BIAS_DRIFT', 'status': 'CONTRADICTED',
             'supporting_evidence_fact_ids': []},
        ],
    }


def test_valid_hypothesis_set_passes():
    record = verify_hypothesis_set(_valid_output(), candidate_universe=['ADCS_RW_JAM', 'ADCS_GYRO_BIAS_DRIFT'])
    assert record.verdict == VerificationVerdict.PASS
    assert all(c['passed'] for c in record.checks)


def test_unknown_fault_rejected():
    output = _valid_output()
    output['hypotheses'][0]['target_fault_id'] = 'NOT_A_FAULT'
    record = verify_hypothesis_set(output, candidate_universe=['ADCS_RW_JAM'])
    assert record.verdict == VerificationVerdict.FAIL
    assert any('candidate_universe' in c['check'] and not c['passed'] for c in record.checks)


def test_supported_without_evidence_fails():
    output = _valid_output()
    output['hypotheses'][0]['supporting_evidence_fact_ids'] = []
    record = verify_hypothesis_set(output, candidate_universe=['ADCS_RW_JAM'])
    assert record.verdict == VerificationVerdict.FAIL
    assert any('supporting_evidence' in c['check'] and not c['passed'] for c in record.checks)


def test_status_contradiction_detected():
    output = _valid_output()
    output['hypotheses'][1]['status'] = 'CONTRADICTED'
    output['hypotheses'][1]['supporting_evidence_fact_ids'] = ['EF-9']
    record = verify_hypothesis_set(output, candidate_universe=['ADCS_RW_JAM'])
    assert record.verdict == VerificationVerdict.FAIL
    assert any('no_status_contradiction' in c['check'] and not c['passed'] for c in record.checks)


def test_empty_hypotheses_passes_vacuously():
    record = verify_hypothesis_set({'episode_id': 'EP-2', 'hypotheses': []}, candidate_universe=['ADCS_RW_JAM'])
    assert record.verdict == VerificationVerdict.PASS