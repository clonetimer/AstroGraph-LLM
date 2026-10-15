"""AI-Native Phase 5: capability federation / insight-sharing registry tests."""
from __future__ import annotations

import pytest

from astrograph.federation import (
    CapabilityFederationRecord, CapabilityFederationRegistry, FederationStatus, TransferInsight,
)


def test_positive_transfer_insight_is_recorded(tmp_path):
    insight = TransferInsight(
        insight_id='TI-1', source_domain='adcs', target_domain='phm',
        shared_backbone='unified-dilated-residual-v1', metric='e2e_macro_f1',
        delta_macro_f1=0.3292, target_baseline_macro_f1=0.5147,
        evidence_ref='governance/evidence/ai_native/unified_baseline_20260823/unified_baseline_report.json',
    )
    record = CapabilityFederationRecord(
        record_id='fed-001', backbone_id='unified-dilated-residual-v1',
        member_domains=['adcs', 'phm'], insights=[insight],
        rationale='PHM gains strongly from the shared backbone; ADCS regresses (recorded separately).',
    )
    registry = CapabilityFederationRegistry(tmp_path)
    path = registry.append(record)
    assert path.is_file()
    loaded = registry.read('fed-001')
    assert len(loaded) == 1
    assert loaded[0].insights[0].relative_change > 0.6


def test_negative_insight_is_recorded_honestly(tmp_path):
    insight = TransferInsight(
        insight_id='TI-2', source_domain='adcs', target_domain='phm',
        shared_backbone='unified-dilated-residual-v1',
        delta_macro_f1=-0.0681, target_baseline_macro_f1=0.8974,
        evidence_ref='unified_baseline_report.json',
    )
    record = CapabilityFederationRecord(
        record_id='fed-002', backbone_id='unified-dilated-residual-v1',
        member_domains=['adcs'], insights=[insight],
        rationale='ADCS regresses on the shared backbone; not a federation candidate for ADCS.',
    )
    registry = CapabilityFederationRegistry(tmp_path)
    registry.append(record)
    loaded = registry.read('fed-002')[0]
    assert loaded.insights[0].delta_macro_f1 < 0
    assert loaded.insights[0].relative_change < 0


def test_audit_reports_domains_and_count(tmp_path):
    registry = CapabilityFederationRegistry(tmp_path)
    for i in range(2):
        insight = TransferInsight(
            insight_id=f'TI-{i}', source_domain='adcs', target_domain='phm',
            shared_backbone='b1', delta_macro_f1=0.1, target_baseline_macro_f1=0.5,
            evidence_ref='e.json',
        )
        record = CapabilityFederationRecord(
            record_id=f'fed-{i:03d}', backbone_id='b1',
            member_domains=['adcs', 'phm'], insights=[insight],
        )
        registry.append(record)
    audit = registry.audit()
    assert audit['status'] == 'PASS'
    assert audit['record_count'] == 2
    assert audit['domains'] == ['adcs', 'phm']


def test_human_review_is_default_gate(tmp_path):
    record = CapabilityFederationRecord(
        record_id='fed-003', backbone_id='b1', member_domains=['adcs'],
        insights=[], rationale='no decision yet',
    )
    assert record.decision == FederationStatus.HUMAN_REVIEW


def test_append_is_append_only(tmp_path):
    registry = CapabilityFederationRegistry(tmp_path)
    insight = TransferInsight(
        insight_id='TI-x', source_domain='adcs', target_domain='phm',
        shared_backbone='b1', delta_macro_f1=0.05, target_baseline_macro_f1=0.5,
        evidence_ref='e.json',
    )
    r1 = CapabilityFederationRecord(record_id='fed-004', backbone_id='b1', member_domains=['adcs'], insights=[insight])
    r2 = CapabilityFederationRecord(record_id='fed-004', backbone_id='b1', member_domains=['adcs'], insights=[insight], decision=FederationStatus.VERIFIED)
    registry.append(r1)
    registry.append(r2)
    loaded = registry.read('fed-004')
    assert len(loaded) == 2  # append-only: both snapshots retained


def test_rejects_empty_domains():
    with pytest.raises(Exception):
        CapabilityFederationRecord(
            record_id='fed-x', backbone_id='b1', member_domains=[], insights=[],
        )