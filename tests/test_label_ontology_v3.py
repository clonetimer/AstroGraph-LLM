from __future__ import annotations

import pytest

from astrograph.assurance.label_ontology_v3 import (
    CausalRole,
    LabelOntologyBundle,
    LabelOntologyRecord,
    LabelSource,
    LabelSourceKind,
    Multiplicity,
    ReviewStatus,
    TimeInterval,
    UNRESOLVED,
)

H = "b" * 64


def source(review: ReviewStatus = ReviewStatus.SOURCE_PINNED) -> LabelSource:
    return LabelSource("source", LabelSourceKind.EXPERIMENT_MANIFEST, H, "manifest.json#fault", review)


def record(**changes) -> LabelOntologyRecord:
    values = dict(
        label_id="fault-a",
        fault_universe_id="U1",
        mechanism="open circuit",
        component="relay K1",
        manifestation="downstream voltage loss",
        injection_location="K1 contacts",
        onset_interval=TimeInterval(1.0, 1.1, "s", "experiment_start"),
        onset_uncertainty=0.1,
        active_interval=TimeInterval(1.0, 5.0, "s", "experiment_start"),
        severity="complete",
        causal_role=CausalRole.ROOT_CAUSE,
        upstream_causes=(),
        downstream_effects=("load voltage loss",),
        multiplicity=Multiplicity.SINGLE,
        label_source=source(),
        label_confidence=0.95,
        supporting_evidence_ids=("rule-k1",),
    )
    values.update(changes)
    return LabelOntologyRecord(**values)


def test_content_hash_is_order_stable() -> None:
    a = record(upstream_causes=("a", "b"))
    b = LabelOntologyRecord.from_dict(a.to_dict())
    assert a.sha256 == b.sha256


def test_unresolved_temporal_label_cannot_support_localization() -> None:
    item = record(onset_interval=None, active_interval=None, onset_uncertainty=None)
    support = item.support_for_task("TEMPORAL_LOCALIZATION")
    assert not support.supported


def test_fault_identification_requires_semantics() -> None:
    item = record(mechanism=UNRESOLVED, component=UNRESOLVED)
    assert not item.support_for_task("FAULT_IDENTIFICATION").supported


def test_root_cause_requires_causal_role() -> None:
    item = record(causal_role=CausalRole.SYMPTOM)
    assert not item.support_for_task("ROOT_CAUSE_ATTRIBUTION").supported


def test_root_cause_requires_source_review_and_confidence() -> None:
    item = record(label_source=source(ReviewStatus.UNREVIEWED), label_confidence=None)
    support = item.support_for_task("ROOT_CAUSE_ATTRIBUTION")
    assert not support.supported
    assert len(support.errors) == 2


def test_explanation_fidelity_requires_executed_evidence_ids() -> None:
    item = record(supporting_evidence_ids=())
    assert not item.support_for_task("EXPLANATION_FIDELITY").supported


def test_bad_interval_is_rejected() -> None:
    with pytest.raises(ValueError):
        TimeInterval(2.0, 1.0, "s", "start")


def test_bundle_rejects_duplicate_label_ids() -> None:
    with pytest.raises(ValueError):
        LabelOntologyBundle("U1", (record(), record()))


def test_bundle_rejects_cross_universe_record() -> None:
    with pytest.raises(ValueError):
        LabelOntologyBundle("U2", (record(),))


def test_unknown_fields_fail_closed() -> None:
    raw = record().to_dict()
    raw["inferred_by_llm"] = True
    with pytest.raises(ValueError):
        LabelOntologyRecord.from_dict(raw)
