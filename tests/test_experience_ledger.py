"""D3: experience-asset ledger (outcome backfill + lesson) tests."""
from __future__ import annotations

import pytest

from astrograph.data_driven.experience import ExperienceLedger, ExperienceRecord, ExperienceStatus


def _record(experience_id: str = 'EXP-1') -> ExperienceRecord:
    return ExperienceRecord(
        experience_id=experience_id, source_cycle_id='cycle-1', dataset_id='phm',
        proposal_summary='promote unified PHM adapter', decision='WITHHELD',
    )


def test_record_and_audit(tmp_path):
    ledger = ExperienceLedger(tmp_path)
    ledger.record(_record())
    audit = ledger.audit()
    assert audit['status'] == 'PASS'
    assert audit['record_count'] == 1
    assert audit['lessons_extracted'] == 0


def test_backfill_outcome_and_lesson(tmp_path):
    ledger = ExperienceLedger(tmp_path)
    ledger.record(_record())
    ledger.backfill_outcome('EXP-1', outcome='5-fold CV showed not uniformly better', lesson='single-split gains must be multi-fold validated before promotion')
    audit = ledger.audit()
    assert audit['lessons_extracted'] == 1
    assert audit['record_count'] == 1
    assert audit['event_count'] == 2
    assert len(ledger._path().read_text(encoding='utf-8').splitlines()) == 2


def test_duplicate_initial_record_rejected(tmp_path):
    ledger = ExperienceLedger(tmp_path)
    ledger.record(_record())
    with pytest.raises(ValueError, match='duplicate experience'):
        ledger.record(_record())


def test_backfill_unknown_rejected(tmp_path):
    ledger = ExperienceLedger(tmp_path)
    ledger.record(_record())
    with pytest.raises(KeyError):
        ledger.backfill_outcome('NOPE', outcome='x', lesson='y')


def test_empty_ledger_audit(tmp_path):
    ledger = ExperienceLedger(tmp_path)
    audit = ledger.audit()
    assert audit['record_count'] == 0
    assert audit['status'] == 'PASS'


def test_status_flow(tmp_path):
    ledger = ExperienceLedger(tmp_path)
    ledger.record(_record())
    path = ledger._path()
    payload = __import__('json').loads(path.read_text(encoding='utf-8').splitlines()[0])
    assert payload['status'] == ExperienceStatus.RECORDED.value


def test_record_requires_summary(tmp_path):
    with pytest.raises(Exception):
        ExperienceRecord(experience_id='EXP-x', source_cycle_id='c', dataset_id='d', proposal_summary='', decision='W')


def test_reuse_requires_extracted_lesson_and_appends_human_approval(tmp_path):
    ledger = ExperienceLedger(tmp_path)
    ledger.record(_record())
    with pytest.raises(ValueError, match='extracted'):
        ledger.approve_reuse('EXP-1', reviewer='reviewer', fault_ids=['F-1'])
    ledger.backfill_outcome('EXP-1', outcome='confirmed', lesson='check correlated telemetry')
    approved = ledger.approve_reuse(
        'EXP-1', reviewer='reviewer', fault_ids=['F-1'], source_refs=['run:1']
    )
    assert approved.status is ExperienceStatus.REUSE_APPROVED
    assert ledger.audit()['reuse_approved'] == 1
