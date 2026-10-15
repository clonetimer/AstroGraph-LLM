from __future__ import annotations

import importlib.util
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from astrograph.physics import default_dynamics_contracts
from astrograph.state_estimation import GrayBoxStateEstimator, StateEstimateResult
from astrograph.diagnosis.generic import (
    AnomalyDecision,
    AssetTopologyContext,
    CalibratedMechanismModel,
    CalibratedSegmentedMechanismModel,
    DataQualityContext,
    DataQualityStatus,
    FailureMechanism,
    GenericDiagnosisPipeline,
    GenericDecisionGovernor,
    GenericDisposition,
    GenericHeadManifest,
    HeadExecutionStatus,
    LocalizationDecision,
    LocalizationCandidate,
    LocalizationHead,
    LocalizationRole,
    MechanismCandidate,
    MechanismCalibrationArtifact,
    MechanismFeatureVector,
    MechanismDecision,
    OpenSetDecision,
    ReasoningHead,
    ReasoningDecision,
    RootCauseEvidenceContext,
    SEGMENTED_FEATURE_NAMES,
    SegmentedMechanismCalibrationArtifact,
    extract_mechanism_features,
    extract_segmented_mechanism_features,
)


ROOT = Path(__file__).resolve().parents[1]


def _calibration_module():
    path = ROOT / "scripts/acceptance/calibrate_gray_box_uncertainty.py"
    spec = importlib.util.spec_from_file_location("generic_head_synthetic_cases", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _storage_calibrated_estimator():
    """Estimator for the frozen generic lineage: legacy storage contract + a
    synthetic storage calibration profile.

    The mainline storage contract (v1.1) is NOT_CALIBRATED under A2, which would
    make the frozen generic heads fail closed.  The generic lineage (ADR-004 ①)
    regression frames use the legacy v1.0 single-slope contract with a synthetic
    storage calibration so the four-head logic is still exercised (this is a
    test-local synthetic profile, not the frozen production calibration).
    """
    from astrograph.physics.archetypes import legacy_storage_contract
    from astrograph.state_estimation.contracts import (
        ArchetypeUncertaintyCalibration,
        UncertaintyCalibrationProfile,
    )

    synthetic = ArchetypeUncertaintyCalibration(
        component_archetype_id="storage.energy_buffer",
        confidence_level=0.95,
        state_interval_multiplier=1.96,
        parameter_interval_multiplier=1.96,
        calibration_trajectory_count=10,
        calibration_split_sha256="0" * 64,
        conservation_change_point_fractions=[0.2, 0.4, 0.6, 0.8],
        conservation_development_episode_scores=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
    )
    profile = UncertaintyCalibrationProfile(
        calibration_id="generic-test-synthetic-storage.v1",
        version="test",
        estimator_id="astrograph.gray-box.bounded-balance",
        estimator_version="1.2.0",
        random_seed=20260802,
        archetypes=[synthetic],
    )
    return GrayBoxStateEstimator(profile, use_frozen_calibration=False), legacy_storage_contract()


def _normal_state_estimate() -> StateEstimateResult:
    """Normal storage estimate under the legacy single-slope contract."""
    module = _calibration_module()
    timestamps = np.arange(0.0, 240.0, 1.0)
    power = 35.0 + 10.0 * np.sin(timestamps / 20.0)  # single-slope (positive) profile
    capacity = 20_000.0
    state = np.empty_like(timestamps)
    state[0] = 0.25
    for index, dt in enumerate(np.diff(timestamps)):
        state[index + 1] = state[index] + dt * 0.5 * (power[index] + power[index + 1]) / capacity
    frame = module._frame(
        "storage.energy_buffer",
        timestamps,
        [
            ("state.energy_fraction", "energy.state_of_charge", "1", state),
            ("power.exchange", "electrical.active_power", "W", power),
        ],
    )
    estimator, contract = _storage_calibrated_estimator()
    return estimator.estimate(frame, contract)


def _significant_state_estimate() -> StateEstimateResult:
    result = _normal_state_estimate()
    payload = result.model_dump(mode="json")
    payload["ood_score"] = 0.989
    payload["constraint_violations"] = [
        {
            "code": "CONSERVATION_RESIDUAL_EXCEEDED",
            "index": 5,
            "magnitude": 0.002,
            "limit": 0.001,
        }
    ]
    payload["conservation_assessment"].update(
        {
            "raw_violation_count": 1,
            "raw_violation_fraction": 1.0 / 159.0,
            "episode_score": 12.0,
            "empirical_p_value": 1.0 / 257.0,
            "p_value_upper_confidence": 0.011,
            "statistically_significant": True,
        }
    )
    return StateEstimateResult.model_validate(payload)


def _topology() -> AssetTopologyContext:
    return AssetTopologyContext(
        episode_id="episode:generic-001",
        asset_instance_id="asset:vehicle-001",
        component_instance_id="component:storage-001",
        component_archetype_id="storage.energy_buffer",
        asset_path=[
            "asset:vehicle-001",
            "subsystem:energy-001",
            "component:storage-001",
        ],
    )


def test_normal_episode_runs_all_four_heads_and_returns_normal() -> None:
    result = GenericDiagnosisPipeline().diagnose(
        state_estimate=_normal_state_estimate(),
        topology=_topology(),
        data_quality=DataQualityContext(status=DataQualityStatus.VALID),
    )

    assert result.anomaly.decision == AnomalyDecision.NORMAL
    assert result.mechanism.decision == MechanismDecision.NO_ANOMALY
    assert result.localization.decision == LocalizationDecision.NO_ANOMALY
    assert result.reasoning.decision == ReasoningDecision.NO_ANOMALY
    assert result.decision.disposition == GenericDisposition.NORMAL
    receipts = [
        result.anomaly.receipt,
        result.mechanism.receipt,
        result.localization.receipt,
        result.reasoning.receipt,
    ]
    assert all(receipt.labels_used is False for receipt in receipts)
    assert all(receipt.truth_used is False for receipt in receipts)


def test_anomaly_preserves_generic_candidate_but_refuses_uncalibrated_root_cause() -> None:
    result = GenericDiagnosisPipeline().diagnose(
        state_estimate=_significant_state_estimate(),
        topology=_topology(),
        data_quality=DataQualityContext(status=DataQualityStatus.VALID),
    )

    assert result.anomaly.decision == AnomalyDecision.ANOMALOUS
    assert result.anomaly.raw_violation_count == 1
    assert result.mechanism.decision == MechanismDecision.CANDIDATES
    assert [item.mechanism_id for item in result.mechanism.candidates] == [
        FailureMechanism.UNMODELED_BALANCE_DISTURBANCE
    ]
    assert result.mechanism.candidates[0].calibrated is False
    assert result.localization.decision == (
        LocalizationDecision.AFFECTED_COMPONENT_IDENTIFIED
    )
    assert result.localization.candidates[0].role == LocalizationRole.AFFECTED_COMPONENT
    assert result.localization.candidates[0].root_cause_supported is False
    assert result.reasoning.decision == ReasoningDecision.INSUFFICIENT_EVIDENCE
    assert set(result.reasoning.hypotheses[0].missing_requirements) == {
        "MECHANISM_CALIBRATION",
        "ROOT_CAUSE_LOCALIZATION_EVIDENCE",
    }
    assert result.decision.disposition == GenericDisposition.INSUFFICIENT_EVIDENCE
    assert "OPEN_SET_OOD_NOT_EVALUATED" in result.decision.reason_codes
    assert result.mechanism.open_set_decision == OpenSetDecision.NOT_EVALUATED
    assert result.decision.formal_release_allowed is False


def test_invalid_data_short_circuits_downstream_heads() -> None:
    result = GenericDiagnosisPipeline().diagnose(
        state_estimate=_normal_state_estimate(),
        topology=_topology(),
        data_quality=DataQualityContext(
            status=DataQualityStatus.INVALID,
            reason_codes=["TIMESTAMPS_NOT_MONOTONIC"],
        ),
    )

    assert result.anomaly.decision == AnomalyDecision.DATA_INVALID
    assert result.mechanism.execution_status == HeadExecutionStatus.NOT_EVALUATED
    assert result.localization.execution_status == HeadExecutionStatus.NOT_EVALUATED
    assert result.reasoning.execution_status == HeadExecutionStatus.NOT_EVALUATED
    assert result.decision.disposition == GenericDisposition.DATA_INVALID
    assert result.decision.unresolved_requirements == ["TIMESTAMPS_NOT_MONOTONIC"]


def test_pipeline_rejects_topology_archetype_mismatch() -> None:
    topology = _topology().model_copy(
        update={"component_archetype_id": "thermal.lumped_capacity"}
    )
    try:
        GenericDiagnosisPipeline().diagnose(
            state_estimate=_normal_state_estimate(),
            topology=topology,
            data_quality=DataQualityContext(status=DataQualityStatus.VALID),
        )
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("topology mismatch must fail closed")


def test_frozen_head_manifest_discloses_incomplete_calibration() -> None:
    manifest = GenericHeadManifest.model_validate(
        json.loads(
            (ROOT / "configs/diagnosis/generic_heads.v1.json").read_text(
                encoding="utf-8"
            )
        )
    )
    assert [item.sequence for item in manifest.bindings] == [1, 2, 3, 4]
    assert manifest.bindings[0].formal_output_allowed is True
    assert all(
        item.formal_output_allowed is False for item in manifest.bindings[1:]
    )
    assert manifest.open_set_ood_calibrated is False
    assert manifest.known_mechanism_classification_calibrated is False
    assert manifest.root_cause_localization_calibrated is False
    assert manifest.labels_used is False and manifest.truth_used is False


def _future_calibrated_results(*, two_mechanisms: bool = False):
    baseline = GenericDiagnosisPipeline().diagnose(
        state_estimate=_significant_state_estimate(),
        topology=_topology(),
        data_quality=DataQualityContext(status=DataQualityStatus.VALID),
    )
    anomaly = baseline.anomaly

    original_mechanism = baseline.mechanism.candidates[0]
    candidate_payload = original_mechanism.model_dump(mode="json")
    candidate_payload["calibrated"] = True
    candidates = [MechanismCandidate.model_validate(candidate_payload)]
    if two_mechanisms:
        second_payload = dict(candidate_payload)
        second_payload["mechanism_id"] = FailureMechanism.DRIFT.value
        candidates.append(MechanismCandidate.model_validate(second_payload))
    mechanism_payload = baseline.mechanism.model_dump(mode="json")
    mechanism_payload["open_set_decision"] = OpenSetDecision.IN_DISTRIBUTION.value
    mechanism_payload["candidates"] = [item.model_dump(mode="json") for item in candidates]
    mechanism = type(baseline.mechanism).model_validate(mechanism_payload)

    localization_payload = baseline.localization.candidates[0].model_dump(mode="json")
    localization_payload.update(
        {"role": LocalizationRole.ROOT_CAUSE.value, "root_cause_supported": True}
    )
    root_cause = LocalizationCandidate.model_validate(localization_payload)
    head_payload = baseline.localization.model_dump(mode="json")
    head_payload["decision"] = LocalizationDecision.ROOT_CAUSE_IDENTIFIED.value
    head_payload["candidates"] = [root_cause.model_dump(mode="json")]
    localization = type(baseline.localization).model_validate(head_payload)
    reasoning = ReasoningHead().evaluate(
        episode_id=_topology().episode_id,
        mechanism=mechanism,
        localization=localization,
    )
    return baseline, anomaly, mechanism, localization, reasoning


def test_governor_handles_one_supported_hypothesis_and_conflict_separately() -> None:
    baseline, anomaly, mechanism, localization, reasoning = _future_calibrated_results()
    decision = GenericDecisionGovernor().decide(
        episode_id=baseline.episode_id,
        data_quality=DataQualityContext(status=DataQualityStatus.VALID),
        anomaly=anomaly,
        mechanism=mechanism,
        localization=localization,
        reasoning=reasoning,
    )
    assert decision.disposition == GenericDisposition.DIAGNOSED
    assert decision.accepted_hypothesis_id == reasoning.hypotheses[0].hypothesis_id

    baseline, anomaly, mechanism, localization, reasoning = _future_calibrated_results(
        two_mechanisms=True
    )
    conflict = GenericDecisionGovernor().decide(
        episode_id=baseline.episode_id,
        data_quality=DataQualityContext(status=DataQualityStatus.VALID),
        anomaly=anomaly,
        mechanism=mechanism,
        localization=localization,
        reasoning=reasoning,
    )
    assert conflict.disposition == GenericDisposition.AMBIGUOUS
    assert conflict.accepted_hypothesis_id is None


def test_governor_rejects_calibrated_open_set_ood_before_classification() -> None:
    baseline, anomaly, mechanism, localization, reasoning = _future_calibrated_results()
    payload = mechanism.model_dump(mode="json")
    payload["open_set_decision"] = OpenSetDecision.OOD.value
    ood_mechanism = type(mechanism).model_validate(payload)
    decision = GenericDecisionGovernor().decide(
        episode_id=baseline.episode_id,
        data_quality=DataQualityContext(status=DataQualityStatus.VALID),
        anomaly=anomaly,
        mechanism=ood_mechanism,
        localization=localization,
        reasoning=reasoning,
    )
    assert decision.disposition == GenericDisposition.OOD
    assert decision.accepted_hypothesis_id is None
    assert decision.reason_codes == ["CALIBRATED_OPEN_SET_REJECTION"]


def test_mechanism_feature_vector_is_fixed_finite_and_truth_blind() -> None:
    features = extract_mechanism_features(_normal_state_estimate())
    assert len(features.feature_names) == len(features.values) == 12
    assert features.labels_used is False
    assert features.truth_used is False
    assert all(value == value and abs(value) != float("inf") for value in features.values)


def test_segmented_feature_vector_is_fixed_finite_and_truth_blind() -> None:
    module = _calibration_module()
    config = module._configurations(20270802, 1, validation=False)[
        "storage.energy_buffer"
    ][0]
    frame, _, _ = module._storage_case(config)
    features = extract_segmented_mechanism_features(
        frame, default_dynamics_contracts()["storage.energy_buffer"]
    )
    assert features.feature_names == list(SEGMENTED_FEATURE_NAMES)
    assert len(features.values) == 22
    assert features.anomaly_score >= 0.0
    assert features.labels_used is False and features.truth_used is False
    assert all(value == value and abs(value) != float("inf") for value in features.values)


def test_segmented_candidate_is_one_logistic_head_plus_nonclassifying_envelope() -> None:
    dimension = len(SEGMENTED_FEATURE_NAMES)
    artifact = SegmentedMechanismCalibrationArtifact(
        calibration_id="test-segmented-calibration",
        calibration_version="1.0.0",
        protocol_sha256="a" * 64,
        model_family="L2_REGULARIZED_MULTINOMIAL_LOGISTIC_REGRESSION",
        model_id="test-segmented-model",
        model_version="1.0.0",
        feature_names=list(SEGMENTED_FEATURE_NAMES),
        class_order=[FailureMechanism.CAPACITY_LOSS, FailureMechanism.DRIFT],
        feature_mean=[0.0] * dimension,
        feature_scale=[1.0] * dimension,
        coefficients=[[1.0] + [0.0] * (dimension - 1), [-1.0] + [0.0] * (dimension - 1)],
        intercepts=[0.0, 0.0],
        temperature=1.0,
        anomaly_threshold=1.0,
        class_centers=[[1.0] + [0.0] * (dimension - 1), [-1.0] + [0.0] * (dimension - 1)],
        class_inverse_covariances=[
            np.eye(dimension).tolist(),
            np.eye(dimension).tolist(),
        ],
        class_distance_thresholds=[0.5, 0.5],
        open_set_alpha=0.02,
        anomaly_false_alarm_alpha=0.02,
        covariance_shrinkage=0.1,
        covariance_eigenvalue_floor=1e-6,
        development_split_sha256="b" * 64,
        calibration_split_sha256="c" * 64,
        normal_calibration_split_sha256="d" * 64,
        development_case_count=10,
        calibration_case_count=10,
        normal_calibration_case_count=10,
    )
    prediction = CalibratedSegmentedMechanismModel(artifact).predict(
        extract_segmented_mechanism_features(
            _calibration_module()._storage_case(
                _calibration_module()._configurations(20270802, 1, validation=False)[
                    "storage.energy_buffer"
                ][0]
            )[0],
            default_dynamics_contracts()["storage.energy_buffer"],
        )
    )
    assert prediction.mechanism_id in artifact.class_order
    assert prediction.open_set_decision in {
        OpenSetDecision.IN_DISTRIBUTION,
        OpenSetDecision.OOD,
    }


def test_calibrated_mechanism_model_keeps_open_set_separate_from_class_choice() -> None:
    feature_count = 12
    artifact = MechanismCalibrationArtifact(
        calibration_id="test-calibration",
        calibration_version="1.0.0",
        protocol_sha256="a" * 64,
        model_family="L2_REGULARIZED_MULTINOMIAL_LOGISTIC_REGRESSION",
        model_id="test-model",
        model_version="1.0.0",
        feature_names=extract_mechanism_features(_normal_state_estimate()).feature_names,
        class_order=[FailureMechanism.CAPACITY_LOSS, FailureMechanism.DRIFT],
        feature_mean=[0.0] * feature_count,
        feature_scale=[1.0] * feature_count,
        coefficients=[[1.0] + [0.0] * 11, [-1.0] + [0.0] * 11],
        intercepts=[0.0, 0.0],
        temperature=1.0,
        class_centroids=[[1.0] + [0.0] * 11, [-1.0] + [0.0] * 11],
        class_distance_thresholds=[0.5, 0.5],
        open_set_alpha=0.02,
        development_split_sha256="b" * 64,
        calibration_split_sha256="c" * 64,
        development_case_count=10,
        calibration_case_count=10,
    )
    names = artifact.feature_names
    in_distribution = CalibratedMechanismModel(artifact).predict(
        MechanismFeatureVector(
            feature_names=names,
            values=[1.1] + [0.0] * 11,
            input_sha256="d" * 64,
        )
    )
    assert in_distribution.mechanism_id == FailureMechanism.CAPACITY_LOSS
    assert in_distribution.open_set_decision == OpenSetDecision.IN_DISTRIBUTION

    out_of_distribution = CalibratedMechanismModel(artifact).predict(
        MechanismFeatureVector(
            feature_names=names,
            values=[4.0] + [0.0] * 11,
            input_sha256="e" * 64,
        )
    )
    assert out_of_distribution.mechanism_id == FailureMechanism.CAPACITY_LOSS
    assert out_of_distribution.open_set_decision == OpenSetDecision.OOD


def test_root_cause_requires_calibrated_mechanism_and_complete_boundary_evidence() -> None:
    baseline, anomaly, mechanism, _, _ = _future_calibrated_results()
    evidence = RootCauseEvidenceContext(
        boundary_inputs_observed=True,
        state_channel_integrity_verified=True,
        input_channel_integrity_verified=True,
        time_alignment_verified=True,
        independent_repetitions=2,
        evidence_ids=["EV-INDEPENDENT-REPEAT-1", "EV-INDEPENDENT-REPEAT-2"],
        input_sha256=anomaly.receipt.input_sha256,
    )
    localized = LocalizationHead().evaluate(
        topology=_topology(),
        anomaly=anomaly,
        mechanism=mechanism,
        root_cause_evidence=evidence,
    )
    assert localized.decision == LocalizationDecision.ROOT_CAUSE_IDENTIFIED
    assert localized.candidates[0].role == LocalizationRole.ROOT_CAUSE
    assert localized.candidates[0].root_cause_supported is True

    incomplete = evidence.model_copy(update={"independent_repetitions": 1})
    affected_only = LocalizationHead().evaluate(
        topology=_topology(),
        anomaly=anomaly,
        mechanism=mechanism,
        root_cause_evidence=incomplete,
    )
    assert affected_only.decision == LocalizationDecision.AFFECTED_COMPONENT_IDENTIFIED
    assert affected_only.candidates[0].root_cause_supported is False
    assert baseline.decision.formal_release_allowed is False


def test_failed_frozen_candidate_is_retained_but_not_bound_to_production() -> None:
    artifact_path = ROOT / "src/astrograph/resources/generic_mechanism_calibration.v1.json"
    artifact = MechanismCalibrationArtifact.model_validate_json(
        artifact_path.read_text(encoding="utf-8")
    )
    evidence = json.loads(
        (
            ROOT
            / "governance/evidence/0.7.1-extension/generic_head_technical_validation.v1.json"
        ).read_text(encoding="utf-8")
    )
    canonical_artifact = json.dumps(
        artifact.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert hashlib.sha256(canonical_artifact).hexdigest() == evidence[
        "calibration_artifact_sha256"
    ]
    assert evidence["denominators"] == {
        "normal": 240,
        "known": 720,
        "open_set": 720,
        "total": 1680,
    }
    assert evidence["technical_status"] == "FAIL"
    assert evidence["p014_status"] == "OPEN"
    assert evidence["checks"]["denominators_preserved"] is True
    assert all(
        passed is False
        for name, passed in evidence["checks"].items()
        if name != "denominators_preserved"
    )
    manifest = GenericHeadManifest.model_validate_json(
        (ROOT / "configs/diagnosis/generic_heads.v1.json").read_text(encoding="utf-8")
    )
    assert manifest.known_mechanism_classification_calibrated is False
    assert manifest.open_set_ood_calibrated is False
    assert manifest.root_cause_localization_calibrated is False
    assert manifest.bindings[1].formal_output_allowed is False


def test_segmented_candidate_fit_is_reproducibly_bound_but_not_promoted() -> None:
    artifact_path = ROOT / "src/astrograph/resources/generic_mechanism_calibration.v2.json"
    artifact = SegmentedMechanismCalibrationArtifact.model_validate_json(
        artifact_path.read_text(encoding="utf-8")
    )
    evidence = json.loads(
        (
            ROOT
            / "governance/evidence/0.7.1-extension/generic_mechanism_fit_calibration.v2.json"
        ).read_text(encoding="utf-8")
    )
    protocol_path = ROOT / "configs/diagnosis/generic_head_calibration_protocol.v2.json"
    canonical_artifact = json.dumps(
        artifact.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert hashlib.sha256(canonical_artifact).hexdigest() == evidence[
        "calibration_artifact_sha256"
    ]
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == artifact.protocol_sha256
    assert evidence["candidate_ordinal"] == 2
    assert evidence["development_case_count"] == 720
    assert evidence["calibration_case_count"] == 360
    assert evidence["normal_calibration_case_count"] == 120
    assert evidence["development_failure_count"] == 0
    assert evidence["calibration_failure_count"] == 0
    assert evidence["normal_calibration_failure_count"] == 0
    assert evidence["technical_validation_read"] is False
    assert evidence["exhausted_validation_seed_used"] is False
    assert evidence["new_validation_seed_used"] is False
    assert evidence["candidate_bound_to_production"] is False


def test_segmented_candidate_single_use_validation_failure_is_retained() -> None:
    evidence = json.loads(
        (
            ROOT
            / "governance/evidence/0.7.1-extension/generic_head_technical_validation.v2.json"
        ).read_text(encoding="utf-8")
    )
    query = json.loads(
        (
            ROOT
            / "governance/evidence/0.7.1-extension/generic_head_technical_validation_query.v2.json"
        ).read_text(encoding="utf-8")
    )
    artifact = SegmentedMechanismCalibrationArtifact.model_validate_json(
        (
            ROOT / "src/astrograph/resources/generic_mechanism_calibration.v2.json"
        ).read_text(encoding="utf-8")
    )
    canonical_artifact = json.dumps(
        artifact.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert query["status"] == "STARTED_AND_SEED_CONSUMED"
    assert query["maximum_invocations"] == 1
    assert query["validation_seed"] == 20290820
    assert query["exhausted_validation_seed_used"] is False
    assert query["calibration_artifact_sha256"] == hashlib.sha256(
        canonical_artifact
    ).hexdigest()
    assert evidence["validation_run_count"] == 1
    assert evidence["denominators"] == {
        "normal": 240,
        "known": 720,
        "open_set": 720,
        "total": 1680,
    }
    assert evidence["failed_run_count"] == 0
    assert evidence["technical_status"] == "FAIL"
    assert evidence["checks"]["denominators_preserved"] is True
    assert evidence["checks"]["runtime_budget"] is True
    assert all(
        evidence["checks"][name] is False
        for name in (
            "detection_accuracy",
            "classification_accuracy",
            "localization_accuracy",
            "reasoning_correctness",
            "known_class_retention",
            "open_set_rejection",
        )
    )
    assert evidence["candidate_bound_to_production"] is False
    assert evidence["p014_status"] == "OPEN"
    assert evidence["p015_status"] == "OPEN"
    assert evidence["g3_status"] == "INCOMPLETE"


def test_final_candidate_failure_and_single_use_seed_are_retained() -> None:
    root = ROOT / "governance/evidence/0.7.1-extension"
    evidence = json.loads((root / "generic_head_technical_validation.v3.json").read_text(encoding="utf-8"))
    query = json.loads((root / "generic_head_technical_validation_query.v3.json").read_text(encoding="utf-8"))
    assert query["validation_seed"] == 20300820
    assert query["maximum_invocations"] == 1
    assert evidence["validation_run_count"] == 1
    assert evidence["denominators"] == {"normal": 240, "known": 720, "open_set": 720, "total": 1680}
    assert evidence["failed_run_count"] == 0
    assert evidence["technical_status"] == "FAIL"
    assert evidence["candidate_bound_to_production"] is False
    assert evidence["checks"]["denominators_preserved"] is True
    assert all(evidence["checks"][key] is False for key in (
        "detection_accuracy", "classification_accuracy", "localization_accuracy",
        "reasoning_correctness", "known_class_retention", "open_set_rejection",
    ))


def test_final_candidate_open_set_rule_is_retained_as_structurally_unreachable() -> None:
    artifact = json.loads(
        (ROOT / "src/astrograph/resources/generic_mechanism_calibration.v3.json").read_text(
            encoding="utf-8"
        )
    )
    evidence = json.loads(
        (ROOT / "governance/evidence/0.7.1-extension/generic_head_technical_validation.v3.json")
        .read_text(encoding="utf-8")
    )
    root_cause = json.loads(
        (ROOT / "governance/evidence/0.7.1-extension/generic_diagnosis_failure_root_cause.v1.json")
        .read_text(encoding="utf-8")
    )
    maximum_runtime_nonconformity = math.log(len(artifact["class_order"]))
    assert all(
        threshold > maximum_runtime_nonconformity
        for threshold in artifact["class_nonconformity_thresholds"]
    )
    assert sum(row["open_set_ood"] for row in evidence["case_results"]) == 0
    finding = next(item for item in root_cause["findings"] if item["finding_id"] == "F-001")
    assert finding["severity"] == "CRITICAL"
    assert root_cause["solution_status"] == "ROOT_CAUSE_IDENTIFIED_REPAIR_NOT_AUTHORIZED"


def test_new_research_phase_is_fail_closed_pending_timer_approval() -> None:
    request = json.loads(
        (ROOT / "protocols/acceptance/p017_new_generic_diagnosis_research_authorization_request.json")
        .read_text(encoding="utf-8")
    )
    assert request["status"] == "PENDING_TIMER_APPROVAL"
    root_cause_path = ROOT / request["bound_root_cause_evidence"]
    assert hashlib.sha256(root_cause_path.read_bytes()).hexdigest() == request[
        "bound_root_cause_evidence_sha256"
    ]
    assert request["permanently_exhausted_validation_seeds"] == [20280820, 20290820, 20300820]
    assert request["requested_stage_a_authority"]["technical_validation_seed"] is None
    assert request["stage_b_gate"]["status"] == "NOT_AUTHORIZED_BY_THIS_REQUEST"


def test_stage_a_research_protocol_is_authorized_and_has_no_validation_seed() -> None:
    protocol = json.loads(
        (ROOT / "configs/diagnosis/generic_head_research_protocol.stage_a.v1.json")
        .read_text(encoding="utf-8")
    )
    attestation_path = ROOT / protocol["authorization"]["path"]
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    assert hashlib.sha256(attestation_path.read_bytes()).hexdigest() == protocol[
        "authorization"
    ]["sha256"]
    assert attestation["status"] == "APPROVED_STAGE_A_ONLY"
    assert protocol["development_seed"] == 20310818
    assert protocol["calibration_seed"] == 20310819
    assert protocol["technical_validation_seed"] is None
    assert protocol["production_candidate_fit"] is False
    assert protocol["technical_validation_authorized"] is False
    assert protocol["permanently_exhausted_validation_seeds"] == [
        20280820,
        20290820,
        20300820,
    ]
    assert protocol["formal_open_set_classes_not_generated"] == [
        "BIAS",
        "INTERMITTENT",
        "STUCK",
    ]


def test_stage_a_v2_oracle_is_nonproduction_and_retains_v1_failure() -> None:
    protocol = json.loads(
        (ROOT / "configs/diagnosis/generic_head_research_protocol.stage_a.v2.json")
        .read_text(encoding="utf-8")
    )
    retained = protocol["retained_stage_a_v1_failure"]
    evidence_path = (
        ROOT / "governance/evidence/0.7.1-extension/generic_diagnosis_stage_a.v1.json"
    )
    assert hashlib.sha256(evidence_path.read_bytes()).hexdigest() == retained[
        "evidence_sha256"
    ]
    assert retained["status"] == "FAIL"
    assert protocol["method_role"].endswith("NOT_A_PRODUCTION_CANDIDATE")
    assert protocol["technical_validation_seed"] is None
    assert protocol["formal_open_set_classes_generated"] == []


def test_stage_a_v2_failure_and_engineering_retry_are_retained() -> None:
    root = ROOT / "governance/evidence/0.7.1-extension"
    error = json.loads(
        (root / "generic_diagnosis_stage_a.v2.attempt1_error.json").read_text(
            encoding="utf-8"
        )
    )
    evidence = json.loads(
        (root / "generic_diagnosis_stage_a.v2.json").read_text(encoding="utf-8")
    )
    assert error["status"] == "FAILED_RETAINED"
    assert error["scientific_evidence_written"] is False
    assert evidence["stage_a_v2_status"] == "FAIL"
    assert evidence["technical_validation_seed"] is None
    assert evidence["exhausted_validation_seeds_used"] == []
    assert evidence["formal_open_set_classes_generated"] == []
    assert evidence["production_candidate_fitted"] is False


def test_stage_a_v3_switching_physics_is_frozen_without_open_classes() -> None:
    protocol = json.loads(
        (ROOT / "configs/diagnosis/generic_head_research_protocol.stage_a.v3.json")
        .read_text(encoding="utf-8")
    )
    assert protocol["technical_validation_seed"] is None
    assert protocol["formal_open_set_classes_generated"] == []
    assert protocol["method_role"].endswith("NOT_A_PRODUCTION_CANDIDATE")
    assert protocol["adequacy_and_open_set_design"]["score_calibration"] == protocol[
        "adequacy_and_open_set_design"
    ]["score_runtime"]
    assert protocol["adequacy_and_open_set_design"]["threshold_fit_split"] == "DEVELOPMENT"
    assert all(item["status"] == "FAIL" for item in protocol["retained_stage_a_failures"])


def test_stage_a_v3_failure_blocks_stage_b_and_preserves_seed_isolation() -> None:
    evidence = json.loads(
        (
            ROOT
            / "governance/evidence/0.7.1-extension/generic_diagnosis_stage_a.v3.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["stage_a_v3_status"] == "FAIL"
    assert evidence["stage_b_authorized"] is False
    assert evidence["technical_validation_seed"] is None
    assert evidence["exhausted_validation_seeds_used"] == []
    assert evidence["formal_open_set_classes_generated"] == []
    assert evidence["production_candidate_fitted"] is False
    assert evidence["same_score_semantics"] is True
    assert all(evidence["open_set_rule_reachable"].values())


def test_downstream_metrics_cannot_alias_classification() -> None:
    contract = json.loads(
        (ROOT / "configs/diagnosis/generic_downstream_evaluation.stage_a.v1.json")
        .read_text(encoding="utf-8")
    )
    assert "classification_correct" in contract["localization"]["forbidden_aliases"]
    assert "classification_correct" in contract["reasoning"]["forbidden_aliases"]
    assert "classification_correct" not in contract["localization"]["correctness_fields"]
    assert "classification_correct" not in contract["reasoning"]["required_fields"]
    assert contract["reasoning"]["llm_formal_gate_allowed"] is False
    assert contract["stage_b_authorized"] is False


def test_observability_intervention_is_pending_and_has_no_validation_seed() -> None:
    request = json.loads(
        (
            ROOT
            / "protocols/acceptance/p017_observability_intervention_authorization_request.json"
        ).read_text(encoding="utf-8")
    )
    summary_path = ROOT / request["bound_stage_a_summary"]
    assert hashlib.sha256(summary_path.read_bytes()).hexdigest() == request[
        "bound_stage_a_summary_sha256"
    ]
    assert request["status"] == "PENDING_EXTERNAL_APPROVAL"
    assert request["requested_authority"]["technical_validation_seed"] is None
    assert "Create or reserve a technical-validation seed" in request["not_authorized"]


def test_timer_observability_attestation_is_scoped_to_development_and_calibration() -> None:
    request_path = (
        ROOT
        / "protocols/acceptance/p017_observability_intervention_authorization_request.json"
    )
    attestation = json.loads(
        (
            ROOT
            / "protocols/acceptance/p017_observability_intervention_timer_attestation.json"
        ).read_text(encoding="utf-8")
    )
    assert attestation["status"] == "APPROVED_OBSERVABILITY_INTERVENTION_ONLY"
    assert attestation["bound_request"]["sha256"] == hashlib.sha256(
        request_path.read_bytes()
    ).hexdigest()
    assert attestation["authorized_seeds"] == {
        "development": 20320818,
        "calibration": 20320819,
        "technical_validation": None,
    }
    assert not attestation["authorization_limits"]["start_stage_b"]
    assert not attestation["authorization_limits"]["fit_production_candidate"]


def test_observability_protocol_freezes_paired_arms_before_execution() -> None:
    protocol = json.loads(
        (
            ROOT
            / "configs/diagnosis/generic_observability_intervention_protocol.v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["development_seed"] == 20320818
    assert protocol["calibration_seed"] == 20320819
    assert protocol["technical_validation_seed"] is None
    assert protocol["permanently_exhausted_validation_seeds"] == [
        20280820,
        20290820,
        20300820,
    ]
    assert [item["arm_id"] for item in protocol["thermal_arms"]] == [
        "PASSIVE_SHORT",
        "PASSIVE_LONG",
        "ACTIVE_MULTISCALE_LONG",
    ]
    assert protocol["thermal_arms"][1]["observation_horizon_seconds"] == 3600.0
    assert protocol["thermal_arms"][2]["dwell_seconds"] == [90.0, 360.0]
    assert protocol["root_cause_decision_thresholds"] == {
        "material_accuracy_improvement_absolute": 0.10,
        "material_generalization_gap_absolute": 0.10,
        "numerical_observed_order_minimum": 1.0,
        "numerical_fine_grid_relative_error_maximum": 0.0001,
        "rms_match_relative_tolerance": 0.05,
    }
    assert protocol["formal_open_set_classes_generated"] == []
    assert not protocol["stage_b_authorized"]


def test_observability_intervention_failure_is_complete_and_blocks_stage_b() -> None:
    evidence = json.loads(
        (
            ROOT
            / "governance/evidence/0.7.1-extension/generic_thermal_observability_intervention.v1.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["observability_intervention_status"] == "FAIL"
    assert evidence["development_seed"] == 20320818
    assert evidence["calibration_seed"] == 20320819
    assert evidence["technical_validation_seed"] is None
    assert evidence["exhausted_validation_seeds_used"] == []
    assert evidence["formal_open_set_classes_generated"] == []
    assert evidence["failed_case_count"] == 0
    assert evidence["paired_controls_pass"] is True
    assert evidence["rms_match_pass"] is True
    assert evidence["simulation_numerical_verification"]["pass"] is True
    assert evidence["production_candidate_fitted"] is False
    assert evidence["stage_b_proposal_eligible"] is False
    assert evidence["stage_b_authorized"] is False


def test_observability_root_cause_retains_storage_generalization_correction() -> None:
    evidence_path = (
        ROOT
        / "governance/evidence/0.7.1-extension/generic_thermal_observability_intervention.v1.json"
    )
    analysis = json.loads(
        (
            ROOT
            / "governance/evidence/0.7.1-extension/p017_observability_root_cause_analysis.v1.json"
        ).read_text(encoding="utf-8")
    )
    assert analysis["bound_intervention_evidence"]["sha256"] == hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    roles = {
        item["factor"]: item["causal_role"] for item in analysis["factor_assessment"]
    }
    assert roles["ORIGINAL_THERMAL_OBSERVATION_HORIZON"] == "PRIMARY_CAUSE"
    assert roles["CURRENT_GOVERNED_SWITCHING_PHYSICS_METHOD"] == (
        "SECONDARY_BLOCKING_CAUSE"
    )
    correction = analysis["automated_attribution_correction"]
    assert correction["scientific_evidence_modified"] is False
    assert "storage/model risk" in correction["corrected_interpretation"]
    assert analysis["solution_result"]["stage_b_proposal_eligible"] is False


def test_method_readiness_request_is_single_model_and_pending() -> None:
    request = json.loads(
        (
            ROOT / "protocols/acceptance/p017_method_readiness_authorization_request.json"
        ).read_text(encoding="utf-8")
    )
    analysis_path = ROOT / request["bound_root_cause_analysis"]["path"]
    assert request["bound_root_cause_analysis"]["sha256"] == hashlib.sha256(
        analysis_path.read_bytes()
    ).hexdigest()
    authority = request["requested_authority"]
    assert request["status"] == "PENDING_EXTERNAL_APPROVAL"
    assert authority["technical_validation_seed"] is None
    assert authority["maximum_research_alternatives"] == 1
    assert authority["frozen_model_before_fit"]["single_head"] is True
    assert authority["frozen_model_before_fit"]["total_dimension"] == 45


def test_method_readiness_attestation_and_protocol_preserve_scope() -> None:
    request_path = ROOT / "protocols/acceptance/p017_method_readiness_authorization_request.json"
    attestation_path = ROOT / "protocols/acceptance/p017_method_readiness_timer_attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    protocol = json.loads(
        (ROOT / "configs/diagnosis/generic_method_readiness_protocol.v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert attestation["status"] == "APPROVED_METHOD_READINESS_ONLY"
    assert attestation["bound_request"]["sha256"] == hashlib.sha256(
        request_path.read_bytes()
    ).hexdigest()
    assert protocol["authorization"]["sha256"] == hashlib.sha256(
        attestation_path.read_bytes()
    ).hexdigest()
    assert protocol["development_seed"] == 20330818
    assert protocol["calibration_seed"] == 20330819
    assert protocol["technical_validation_seed"] is None
    assert protocol["single_model"]["model_count"] == 1
    assert protocol["feature_mapping"]["total_dimension"] == 45
    assert protocol["formal_open_set_classes_generated"] == []
    assert protocol["conformal_rule"]["score_calibration"] == protocol[
        "conformal_rule"
    ]["score_runtime"]
    assert not protocol["stage_b_authorized"]


def test_method_literature_review_does_not_preapprove_result() -> None:
    review = json.loads(
        (
            ROOT
            / "governance/evidence/0.7.1-extension/p017_method_literature_review.v1.json"
        ).read_text(encoding="utf-8")
    )
    assessment = review["method_assessment"]
    assert assessment["from_scratch_algorithm"] is False
    assert assessment["not_a_concept_bottleneck_model"] is True
    assert review["risk_and_exit"]["literature_does_not_preapprove_pass"] is True
    assert review["risk_and_exit"]["model_or_protocol_changed_after_review"] is False


def test_method_readiness_failure_retains_denominators_and_blocks_stage_b() -> None:
    evidence_path = (
        ROOT / "governance/evidence/0.7.1-extension/generic_method_readiness.v1.json"
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    summary = json.loads(
        (
            ROOT
            / "governance/evidence/0.7.1-extension/p017_method_readiness_summary.v1.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["method_readiness_status"] == "FAIL"
    assert evidence["development_seed"] == 20330818
    assert evidence["calibration_seed"] == 20330819
    assert evidence["technical_validation_seed"] is None
    assert evidence["exhausted_validation_seeds_used"] == []
    assert evidence["formal_open_set_classes_generated"] == []
    assert evidence["model_count"] == 1
    assert evidence["production_candidate_fitted"] is False
    assert evidence["stage_b_started"] is False
    assert evidence["stage_b_proposal_eligible"] is False
    assert evidence["stage_b_authorized"] is False
    assert evidence["same_score_semantics"] is True
    assert summary["bound_evidence"]["sha256"] == hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    assert summary["calibration_denominators"] == {
        "normal": 160,
        "known_total": 480,
        "known_per_archetype": 240,
    }
    assert summary["calibration_results"]["thermal_known_classification"]["pass"]
    assert not summary["calibration_results"]["storage_known_classification"]["pass"]


def test_project_gate_status_does_not_claim_prior_gates_complete() -> None:
    status = json.loads(
        (
            ROOT
            / "governance/evidence/0.7.1-extension/project_gate_status_2026-08-02.v2.json"
        ).read_text(encoding="utf-8")
    )
    actual = {
        item["gate"]: item["status"]
        for item in status["gates"]
        if item["gate"] in {"G0", "G1", "G2", "G3", "G4"}
    }
    assert actual == {
        "G0": "INCOMPLETE",
        "G1": "INCOMPLETE",
        "G2": "BLOCKED",
        "G3": "IN_PROGRESS_NOT_PASSED",
        "G4": "NOT_AUTHORIZED",
    }
    assert status["sequential_completion_claim"] is False


def test_p018_stage_x0_request_is_zero_fit_and_preserves_g11() -> None:
    request = json.loads(
        (
            ROOT
            / "protocols/acceptance/p018_cross_component_data_readiness_authorization_request.json"
        ).read_text(encoding="utf-8")
    )
    landscape_path = ROOT / request["bound_landscape"]["path"]
    adr_path = ROOT / request["bound_architecture_decision"]["path"]
    prior_path = ROOT / request["preserves_prior_failure"]["path"]
    assert request["status"] == "PENDING_EXTERNAL_APPROVAL"
    assert request["requested_stage"] == "STAGE_X0_DATA_READINESS_ONLY"
    assert request["bound_landscape"]["sha256"] == hashlib.sha256(
        landscape_path.read_bytes()
    ).hexdigest()
    assert request["bound_architecture_decision"]["sha256"] == hashlib.sha256(
        adr_path.read_bytes()
    ).hexdigest()
    assert request["preserves_prior_failure"]["sha256"] == hashlib.sha256(
        prior_path.read_bytes()
    ).hexdigest()
    authority = request["requested_authority"]
    assert authority["authorized_model_fit_count"] == 0
    assert authority["development_seed"] is None
    assert authority["calibration_seed"] is None
    assert authority["technical_validation_seed"] is None
    assert authority["production_candidate_count"] == 0
    assert authority["stage_b_authorized"] is False
    direction = authority["frozen_architecture_direction_for_design_only"]
    assert direction["detection_and_classification_share_one_required_model"] is False
    assert direction["dynamic_gating_status"] == (
        "CHALLENGER_TO_BE_FROZEN_ONLY_IN_LATER_FIT_REQUEST"
    )
    assert direction["current_45d_logistic_role"] == "RETAINED_FAILED_BASELINE_ONLY"
    assert direction["unique_production_estimator_manifest_preserved"] is True
    assert direction["estimator_ensemble_fallback_or_second_slot_authorized"] is False
    assert set(request["g1_1_preserved"]["exhausted_validation_seeds"]) == {
        20280820,
        20290820,
        20300820,
    }
    assert request["g1_1_preserved"]["thresholds_and_denominators_unchanged"]
    preserved = request["g1_1_preserved"]
    approval_path = ROOT / preserved["approval_binding"]["path"]
    assert preserved["approval_binding"]["sha256"] == hashlib.sha256(
        approval_path.read_bytes()
    ).hexdigest()
    for binding in preserved["artifact_bindings"].values():
        assert binding["sha256"] == hashlib.sha256(
            (ROOT / binding["path"]).read_bytes()
        ).hexdigest()
    contract = preserved["generic_technical_contract"]
    assert contract["source_sha256"] == hashlib.sha256(
        (ROOT / contract["source_path"]).read_bytes()
    ).hexdigest()
    assert contract["denominators"] == {
        "normal": 240,
        "known": 720,
        "open_set": 720,
        "total": 1680,
    }
    assert contract["maximum_runtime_seconds"] == 600
    denied = " ".join(authority["gate_input_denylist"]).lower()
    for forbidden in ("dataset", "filename", "case", "seed", "fault label", "truth"):
        assert forbidden in denied


def test_p018_landscape_requires_heterogeneous_features_and_external_sources() -> None:
    landscape = json.loads(
        (
            ROOT
            / "governance/evidence/0.7.1-extension/p018_cross_component_data_method_landscape.v1.json"
        ).read_text(encoding="utf-8")
    )
    assert landscape["status"] == "DESIGN_EVIDENCE_ONLY"
    assert landscape["evidence_status"] == "WAITING_FOR_DATA_AND_TIMER_APPROVAL"
    assert len(landscape["proposed_hierarchy"]) == 5
    feature_policy = landscape["archetype_specific_feature_policy"]
    assert set(feature_policy) == {
        "shared_only",
        "storage",
        "thermal",
        "actuator",
        "sensor",
        "converter",
        "transport_link",
        "queue",
    }
    assert feature_policy["storage"] != feature_policy["thermal"]
    assert feature_policy["actuator"] != feature_policy["sensor"]
    portfolio = {item["dataset_id"]: item for item in landscape["data_portfolio"]}
    assert portfolio["LOCAL_SAT_AGENT_WHOLE_SPACECRAFT"]["source_type"] == (
        "SYNTHETIC_SINGLE_SOURCE"
    )
    for dataset_id in ("ESA_ANOMALY_DATASET_2024", "OPS_SAT_AD_2024", "NASA_ADAPT_EPS"):
        assert portfolio[dataset_id]["local_status"] == "NOT_ACQUIRED"
    assert [item["level"] for item in landscape["validation_ladder"]] == [
        "V0",
        "V1",
        "V2",
        "V3",
        "V4",
    ]
    assert "No external dataset was downloaded" in landscape["claim_boundary"]
