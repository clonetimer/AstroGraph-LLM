from __future__ import annotations

import json
from dataclasses import replace

import pytest

from astrograph.assurance.claim_registry import (
    ArtifactRef,
    ArtifactRole,
    ClaimRecord,
    ClaimScope,
    ClaimValidationError,
    DatasetEvidenceProfile,
    EvaluationUse,
    EvidenceTier,
    LabelAccess,
    ProjectClaimPolicy,
    ProjectStatus,
    RecordState,
    StatisticalUnit,
    TaskType,
    append_claim_record,
    require_valid_claim,
    validate_claim,
    verify_claim_ledger,
)
from astrograph.assurance.label_ontology_v3 import (
    CausalRole,
    LabelOntologyRecord,
    LabelSource,
    LabelSourceKind,
    Multiplicity,
    ReviewStatus,
    TimeInterval,
    UNRESOLVED,
)

H = "a" * 64


def policy() -> ProjectClaimPolicy:
    return ProjectClaimPolicy(
        project_version="B006",
        project_status=ProjectStatus.WIP_NOT_RELEASED,
        permanently_post_hoc_groups=("4", "9"),
        globally_frozen_claim_scopes=(
            ClaimScope.UNIVERSAL_KG_GAIN,
            ClaimScope.HIL_VALIDITY,
            ClaimScope.FLIGHT_VALIDITY,
            ClaimScope.FLIGHT_QUALIFICATION,
            ClaimScope.SAFETY_QUALIFICATION,
            ClaimScope.G0_G4_STATUS,
        ),
    )


def profile() -> DatasetEvidenceProfile:
    return DatasetEvidenceProfile(
        dataset_id="ADCS",
        evidence_tier=EvidenceTier.S0_INTERNAL_SYNTHETIC,
        domain_scope="internal",
        allowed_tasks=(TaskType.TEMPORAL_LOCALIZATION,),
        allowed_evaluation_uses=(EvaluationUse.DEVELOPMENT, EvaluationUse.POST_HOC_AUDIT),
        allowed_claim_scopes=(ClaimScope.INTERNAL_METHOD_VALIDITY, ClaimScope.POST_HOC_EVIDENCE),
        forbidden_claim_scopes=(ClaimScope.FLIGHT_VALIDITY,),
        allowed_statistical_units=(StatisticalUnit.MONTE_CARLO_GROUP,),
        confirmatory_eligible=False,
        previously_accessed=True,
    )


def label() -> LabelOntologyRecord:
    return LabelOntologyRecord(
        label_id="F1",
        fault_universe_id="ADCS14",
        mechanism="bias",
        component="gyro",
        manifestation="residual shift",
        injection_location="gyro.x",
        onset_interval=TimeInterval(10.0, 10.5, "s", "trajectory_start"),
        onset_uncertainty=0.5,
        active_interval=TimeInterval(10.0, 50.0, "s", "trajectory_start"),
        severity="fixed",
        causal_role=CausalRole.ROOT_CAUSE,
        upstream_causes=(),
        downstream_effects=("attitude_error",),
        multiplicity=Multiplicity.SINGLE,
        label_source=LabelSource("sim", LabelSourceKind.SIMULATOR_DEFINITION, H, "config/f1", ReviewStatus.SOURCE_PINNED),
        label_confidence=1.0,
        supporting_evidence_ids=("E1",),
    )


def artifacts() -> tuple[ArtifactRef, ...]:
    return (
        ArtifactRef("trace", ArtifactRole.SCORE_TRACE, "1" * 64, LabelAccess.LABEL_FREE_RUNTIME, True),
        ArtifactRef("pred", ArtifactRole.PREDICTIONS, "2" * 64, LabelAccess.LABEL_FREE_RUNTIME, True),
        ArtifactRef("truth", ArtifactRole.GROUND_TRUTH, "3" * 64, LabelAccess.EVALUATOR_ONLY, False),
        ArtifactRef("metrics", ArtifactRole.METRICS, "4" * 64, LabelAccess.EVALUATOR_ONLY, False),
        ArtifactRef("manifest", ArtifactRole.RUN_MANIFEST, "5" * 64, LabelAccess.METADATA_ONLY, True),
    )


def valid_dev_claim() -> ClaimRecord:
    p = policy()
    d = profile()
    lab = label()
    return ClaimRecord(
        claim_id="dev-temporal-v2",
        claim_text="Development-only temporal localisation on groups 3/8.",
        project_version="B006",
        project_status=ProjectStatus.WIP_NOT_RELEASED,
        project_policy_sha256=p.sha256,
        dataset_id="ADCS",
        dataset_profile_sha256=d.sha256,
        task_type=TaskType.TEMPORAL_LOCALIZATION,
        evaluation_use=EvaluationUse.DEVELOPMENT,
        claim_scope=ClaimScope.INTERNAL_METHOD_VALIDITY,
        statistical_unit=StatisticalUnit.MONTE_CARLO_GROUP,
        groups_used=("3", "8"),
        parameter_selection_groups=("3", "8"),
        policy_frozen_before_evaluation=False,
        truth_join_after_prediction_freeze=True,
        data_previously_used_for_development=True,
        qualification_change_requested=False,
        record_state=RecordState.EVIDENCE_BOUND,
        artifacts=artifacts(),
        label_ontology_record_sha256=lab.sha256,
    )


def error_codes(record: ClaimRecord) -> set[str]:
    return {x.code for x in validate_claim(record, profile(), policy(), label()) if x.severity.value == "ERROR"}


def test_valid_development_claim_groups_3_8() -> None:
    assert not error_codes(valid_dev_claim())


def test_groups_4_9_cannot_select_policy() -> None:
    record = replace(valid_dev_claim(), groups_used=("4", "9"), parameter_selection_groups=("4", "9"))
    codes = error_codes(record)
    assert "POST_HOC_GROUP_USED_FOR_SELECTION" in codes
    assert "EXPOSED_GROUP_MISCLASSIFIED" in codes


def test_groups_4_9_must_be_post_hoc_scope() -> None:
    record = replace(
        valid_dev_claim(),
        groups_used=("4", "9"),
        parameter_selection_groups=(),
        evaluation_use=EvaluationUse.POST_HOC_AUDIT,
        claim_scope=ClaimScope.POST_HOC_EVIDENCE,
    )
    assert not error_codes(record)


def test_runtime_trace_must_be_label_free() -> None:
    broken = list(artifacts())
    broken[0] = replace(broken[0], label_access=LabelAccess.EVALUATOR_ONLY)
    record = replace(valid_dev_claim(), artifacts=tuple(broken))
    assert "RUNTIME_ARTIFACT_LABEL_ACCESS" in error_codes(record)


def test_prediction_must_freeze_before_truth_join() -> None:
    broken = list(artifacts())
    broken[1] = replace(broken[1], frozen_before_truth_join=False)
    record = replace(valid_dev_claim(), artifacts=tuple(broken))
    assert "PREDICTION_NOT_FROZEN_BEFORE_TRUTH" in error_codes(record)


def test_flight_claim_fails_closed() -> None:
    d = profile()
    expanded = replace(
        d,
        allowed_claim_scopes=d.allowed_claim_scopes + (ClaimScope.FLIGHT_VALIDITY,),
        forbidden_claim_scopes=(),
    )
    record = replace(valid_dev_claim(), claim_scope=ClaimScope.FLIGHT_VALIDITY)
    codes = {x.code for x in validate_claim(record, expanded, policy(), label()) if x.severity.value == "ERROR"}
    assert "CLAIM_SCOPE_GLOBALLY_FROZEN" in codes


def test_confirmatory_cannot_use_current_profile() -> None:
    record = replace(
        valid_dev_claim(),
        evaluation_use=EvaluationUse.CONFIRMATORY,
        record_state=RecordState.FROZEN,
        parameter_selection_groups=(),
        policy_frozen_before_evaluation=True,
        data_previously_used_for_development=False,
    )
    codes = error_codes(record)
    assert "EVALUATION_USE_OUTSIDE_DATASET_SCOPE" in codes
    assert "DATASET_NOT_CONFIRMATORY_ELIGIBLE" in codes


def test_invalid_window_as_statistical_unit_is_rejected_at_parse() -> None:
    raw = valid_dev_claim().to_dict()
    raw["statistical_unit"] = "WINDOW"
    with pytest.raises(ValueError):
        ClaimRecord.from_dict(raw)


def test_append_only_hash_chain_detects_tamper(tmp_path) -> None:
    path = tmp_path / "claims.jsonl"
    record = valid_dev_claim()
    first = append_claim_record(path, record, profile(), policy(), label())
    assert verify_claim_ledger(path).valid
    assert verify_claim_ledger(path).final_entry_sha256 == first
    second_record = replace(record, claim_id="dev-temporal-v2-second")
    append_claim_record(path, second_record, profile(), policy(), label())
    assert verify_claim_ledger(path).entry_count == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    payload["record"]["claim_text"] = "tampered"
    lines[0] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not verify_claim_ledger(path).valid


def test_unknown_fields_fail_closed() -> None:
    raw = valid_dev_claim().to_dict()
    raw["accuracy"] = 1.0
    with pytest.raises(ValueError):
        ClaimRecord.from_dict(raw)
