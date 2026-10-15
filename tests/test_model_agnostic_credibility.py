from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from astrograph.data_driven import (
    CASE_TO_FAULT_ID,
    CallableRawModelProvider,
    CredibilityStatus,
    DataDrivenModelManifest,
    DataDrivenModelRegistry,
    DataDrivenPrediction,
    DataDrivenTaskType,
    ExternalServiceModelProvider,
    IndependentRunRecord,
    ModelCapability,
    ModelInferenceRequest,
    ModelOutputSemantics,
    ModelRuntime,
    OpenSetPolicy,
    OpenSetStatus,
    RawModelOutput,
    SklearnProbabilityProvider,
    SplitName,
    TemperatureScaler,
    assess_open_set,
    audit_independent_runs,
)
from astrograph.evidence.executable import stable_id
from astrograph.orchestration.models import (
    ExecutionPolicy,
    PermissionScope,
    TaskEnvelope,
    WorkflowContext,
)
from astrograph.orchestration.whole_spacecraft_online_workflow import WholeSpacecraftOnlineDiagnosisWorkflow
from astrograph.simulation import SimulationRunCompleted, WholeSpacecraftRuntimeEvidenceBuilder

ROOT = Path(__file__).resolve().parents[1]
SCENE_ROOT = ROOT / "tests/fixtures/spacecraft_scenarios"
PROTOCOL_ROOT = ROOT / "protocols/phm_project_a"
FAULTS = sorted(set(CASE_TO_FAULT_ID.values()) - {"NORMAL"})


def _manifest(
    *,
    model_id: str,
    runtime: ModelRuntime,
    semantics: ModelOutputSemantics,
    task: DataDrivenTaskType = DataDrivenTaskType.MULTICLASS_FAULT_CLASSIFICATION,
    calibrated: bool = False,
) -> DataDrivenModelManifest:
    capabilities = [ModelCapability.FAULT_RANKING, ModelCapability.NORMAL_CLASS]
    if calibrated:
        capabilities.append(ModelCapability.CALIBRATED_PROBABILITY)
    return DataDrivenModelManifest(
        model_id=model_id,
        model_version="1.0.0",
        display_name=model_id,
        task_type=task,
        runtime=runtime,
        output_semantics=semantics,
        capabilities=capabilities,
        supported_fault_ids=FAULTS,
        feature_schema_ids=["FEATURE-V1"],
        calibration_id="CAL-V1" if calibrated else None,
    )


def _request(model_id: str, *, episode_id: str = "EP-V0515") -> ModelInferenceRequest:
    return ModelInferenceRequest(
        episode_id=episode_id,
        model_id=model_id,
        model_version="1.0.0",
        feature_schema_id="FEATURE-V1",
        input_data_sha256="0" * 64,
        registered_fault_ids=FAULTS,
        payload={"features": {"x": 1.0, "y": 2.0}, "sequence": [[1.0, 2.0]]},
    )


def test_registry_accepts_pytorch_style_logits_provider() -> None:
    manifest = _manifest(
        model_id="rw-tcn-v1",
        runtime=ModelRuntime.PYTORCH,
        semantics=ModelOutputSemantics.LOGITS,
    )

    def predict(request: ModelInferenceRequest) -> RawModelOutput:
        values = {fault_id: -5.0 for fault_id in FAULTS}
        values["ADCS_RW_FRICTION_INCREASE"] = 3.0
        values["NORMAL"] = -2.0
        return RawModelOutput(
            episode_id=request.episode_id,
            model_id=manifest.model_id,
            model_version=manifest.model_version,
            output_semantics=manifest.output_semantics,
            class_values=values,
            metadata={"architecture": "TCN"},
        )

    registry = DataDrivenModelRegistry.for_research()
    registry.register(CallableRawModelProvider(manifest, predict))
    prediction = registry.predict(_request(manifest.model_id))
    assert prediction.model_family == "PYTORCH"
    assert prediction.output_semantics == "LOGITS"
    assert prediction.classes[0].class_id == "ADCS_RW_FRICTION_INCREASE"
    assert sum(item.probability for item in prediction.classes) == pytest.approx(1.0)
    assert prediction.calibrated_probabilities is False


def test_anomaly_attribution_model_maps_anomaly_mass_to_fault_ranking() -> None:
    manifest = _manifest(
        model_id="transformer-autoencoder-attribution",
        runtime=ModelRuntime.TENSORFLOW,
        semantics=ModelOutputSemantics.ANOMALY_ATTRIBUTION,
        task=DataDrivenTaskType.ANOMALY_ATTRIBUTION,
    )

    def predict(request: ModelInferenceRequest) -> RawModelOutput:
        return RawModelOutput(
            episode_id=request.episode_id,
            model_id=manifest.model_id,
            model_version=manifest.model_version,
            output_semantics=manifest.output_semantics,
            anomaly_score=0.8,
            fault_attribution={
                "THERMAL_HEATER_STUCK_ON": 3.0,
                "THERMAL_RADIATOR_REJECTION_LOSS": 1.0,
            },
        )

    registry = DataDrivenModelRegistry.for_research()
    registry.register(CallableRawModelProvider(manifest, predict))
    prediction = registry.predict(_request(manifest.model_id))
    probabilities = {item.class_id: item.probability for item in prediction.classes}
    assert probabilities["NORMAL"] == pytest.approx(0.2)
    assert probabilities["THERMAL_HEATER_STUCK_ON"] == pytest.approx(0.6)
    assert probabilities["THERMAL_RADIATOR_REJECTION_LOSS"] == pytest.approx(0.2)


def test_external_model_service_uses_same_prediction_protocol() -> None:
    manifest = _manifest(
        model_id="external-onnx-sequence-model",
        runtime=ModelRuntime.EXTERNAL_SERVICE,
        semantics=ModelOutputSemantics.PROBABILITIES,
        calibrated=True,
    )
    calls: list[dict] = []

    def transport(endpoint: str, payload: dict, timeout_s: float) -> dict:
        calls.append({"endpoint": endpoint, "payload": payload, "timeout_s": timeout_s})
        values = {fault_id: 0.0 for fault_id in FAULTS}
        values["ADCS_STAR_TRACKER_DROPOUT"] = 0.9
        values["NORMAL"] = 0.1
        return RawModelOutput(
            episode_id=payload["request"]["episode_id"],
            model_id=manifest.model_id,
            model_version=manifest.model_version,
            output_semantics=manifest.output_semantics,
            class_values=values,
        ).model_dump(mode="json")

    registry = DataDrivenModelRegistry.for_research()
    registry.register(
        ExternalServiceModelProvider(
            manifest=manifest,
            endpoint="http://model-gateway.invalid/predict",
            transport=transport,
        )
    )
    prediction = registry.predict(_request(manifest.model_id))
    assert calls and calls[0]["endpoint"].endswith("/predict")
    assert prediction.classes[0].class_id == "ADCS_STAR_TRACKER_DROPOUT"
    assert prediction.calibrated_probabilities is True
    assert prediction.calibration_id == "CAL-V1"


def test_sklearn_compatible_adapter_is_not_hgb_specific() -> None:
    class FakeRandomForest:
        classes_ = np.asarray(["NORMAL", "EPS_BATTERY_CAPACITY_LOSS"])

        @staticmethod
        def predict_proba(rows: list[list[float]]) -> np.ndarray:
            assert rows == [[1.0, 2.0]]
            return np.asarray([[0.25, 0.75]])

    manifest = DataDrivenModelManifest(
        model_id="rf-baseline",
        model_version="1",
        display_name="Random Forest fixture",
        task_type=DataDrivenTaskType.MULTICLASS_FAULT_CLASSIFICATION,
        runtime=ModelRuntime.SCIKIT_LEARN,
        output_semantics=ModelOutputSemantics.PROBABILITIES,
        capabilities=[ModelCapability.FAULT_RANKING, ModelCapability.NORMAL_CLASS],
        supported_fault_ids=["EPS_BATTERY_CAPACITY_LOSS"],
        feature_schema_ids=["FEATURE-V1"],
    )
    request = ModelInferenceRequest(
        episode_id="EP-RF",
        model_id=manifest.model_id,
        model_version=manifest.model_version,
        feature_schema_id="FEATURE-V1",
        input_data_sha256="1" * 64,
        registered_fault_ids=["EPS_BATTERY_CAPACITY_LOSS"],
        payload={"features": {"x": 1.0, "y": 2.0}},
    )
    registry = DataDrivenModelRegistry.for_research()
    registry.register(SklearnProbabilityProvider(manifest, FakeRandomForest(), ["x", "y"]))
    prediction = registry.predict(request)
    assert prediction.classes[0].class_id == "EPS_BATTERY_CAPACITY_LOSS"
    assert prediction.classes[0].probability == pytest.approx(0.75)


def test_registry_versions_are_explicit_and_duplicate_safe() -> None:
    manifest = _manifest(
        model_id="versioned-model",
        runtime=ModelRuntime.CUSTOM_CALLABLE,
        semantics=ModelOutputSemantics.NONNEGATIVE_SCORES,
    )
    provider = CallableRawModelProvider(
        manifest,
        lambda request: RawModelOutput(
            episode_id=request.episode_id,
            model_id=manifest.model_id,
            model_version=manifest.model_version,
            output_semantics=manifest.output_semantics,
            class_values={"NORMAL": 1.0, **{fault_id: 0.0 for fault_id in FAULTS}},
        ),
    )
    registry = DataDrivenModelRegistry.for_research()
    registry.register(provider)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(provider)
    assert registry.get("versioned-model").manifest.manifest_id == "versioned-model@1.0.0"


def test_non_ranking_model_cannot_feed_birrf_without_adapter() -> None:
    manifest = DataDrivenModelManifest(
        model_id="battery-rul",
        model_version="1",
        display_name="Battery RUL",
        task_type=DataDrivenTaskType.PROGNOSTICS,
        runtime=ModelRuntime.PYTORCH,
        output_semantics=ModelOutputSemantics.NONNEGATIVE_SCORES,
        capabilities=[],
        feature_schema_ids=["FEATURE-V1"],
    )
    provider = CallableRawModelProvider(
        manifest,
        lambda request: RawModelOutput(
            episode_id=request.episode_id,
            model_id=manifest.model_id,
            model_version=manifest.model_version,
            output_semantics=manifest.output_semantics,
            class_values={"RUL_HOURS": 120.0},
        ),
    )
    registry = DataDrivenModelRegistry.for_research()
    registry.register(provider)
    request = ModelInferenceRequest(
        episode_id="EP-RUL",
        model_id=manifest.model_id,
        model_version=manifest.model_version,
        feature_schema_id="FEATURE-V1",
        input_data_sha256="2" * 64,
        payload={},
    )
    with pytest.raises(ValueError, match="does not declare FAULT_RANKING"):
        registry.predict(request)


def test_temperature_scaling_improves_or_preserves_validation_nll() -> None:
    probabilities = np.asarray(
        [
            [0.99, 0.01],
            [0.99, 0.01],
            [0.01, 0.99],
            [0.99, 0.01],
        ]
    )
    truth = np.asarray([0, 1, 1, 0])
    scaler, record = TemperatureScaler.fit(probabilities, truth)
    assert scaler.temperature >= 1.0
    assert (
        record.validation_after.negative_log_likelihood
        <= record.validation_before.negative_log_likelihood + 1e-12
    )


def test_current_e0_inventory_is_blocked_for_independent_run_claims() -> None:
    records = [
        IndependentRunRecord(
            run_id=case_id,
            class_ids=[fault_id],
            split=SplitName.UNASSIGNED,
        )
        for case_id, fault_id in sorted(CASE_TO_FAULT_ID.items())
    ]
    audit = audit_independent_runs(records)
    assert audit.status == CredibilityStatus.BLOCKED
    assert any(reason.startswith("INSUFFICIENT_INDEPENDENT_RUNS") for reason in audit.blocking_reasons)
    assert "UNASSIGNED_RUNS_PRESENT" in audit.blocking_reasons


def test_independent_run_audit_passes_when_each_class_has_disjoint_splits() -> None:
    records = []
    for class_id in ("F-A", "F-B"):
        for split in (SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST):
            records.append(
                IndependentRunRecord(
                    run_id=f"{class_id}-{split.value}",
                    class_ids=[class_id],
                    split=split,
                    satellite_instance_id="SAT-1",
                    operating_condition_id="OC-1",
                )
            )
    audit = audit_independent_runs(records)
    assert audit.status == CredibilityStatus.PASS
    assert not audit.blocking_reasons


def test_open_set_policy_rejects_low_margin_prediction() -> None:
    prediction = DataDrivenPrediction(
        prediction_id="P-OPEN",
        episode_id="EP-OPEN",
        model_id="MODEL",
        model_version="1",
        classes=[
            {"class_id": "F-A", "probability": 0.40, "rank": 1},
            {"class_id": "F-B", "probability": 0.38, "rank": 2},
            {"class_id": "NORMAL", "probability": 0.22, "rank": 3},
        ],
        feature_schema_id="FEATURE",
        input_data_sha256="3" * 64,
    )
    assessment = assess_open_set(
        prediction,
        OpenSetPolicy(minimum_top_probability=0.35, minimum_top_margin=0.08),
    )
    assert assessment.status == OpenSetStatus.REJECT_UNKNOWN
    assert "TOP_MARGIN_BELOW_THRESHOLD" in assessment.reasons
    assert "CALIBRATED_PROBABILITIES_REQUIRED" in assessment.reasons
    assert assessment.data_branch_eligible_for_fusion is False


def test_open_set_default_accepts_only_receipt_backed_calibrated_probabilities() -> None:
    prediction = DataDrivenPrediction(
        prediction_id="P-OPEN-CALIBRATED",
        episode_id="EP-OPEN-CALIBRATED",
        model_id="MODEL",
        model_version="1",
        classes=[
            {"class_id": "F-A", "probability": 0.80, "rank": 1},
            {"class_id": "F-B", "probability": 0.10, "rank": 2},
            {"class_id": "NORMAL", "probability": 0.10, "rank": 3},
        ],
        feature_schema_id="FEATURE",
        input_data_sha256="5" * 64,
        calibrated_probabilities=True,
        calibration_id="CAL-INDEPENDENT-1",
    )

    assessment = assess_open_set(prediction)

    assert assessment.status == OpenSetStatus.ACCEPT_KNOWN
    assert assessment.reasons == []


def test_prediction_rejects_orphan_calibration_identifier() -> None:
    with pytest.raises(ValueError, match="calibration_id requires calibrated_probabilities=True"):
        DataDrivenPrediction(
            prediction_id="P-ORPHAN-CAL",
            episode_id="EP-ORPHAN-CAL",
            model_id="MODEL",
            model_version="1",
            classes=[
                {"class_id": "F-A", "probability": 0.90, "rank": 1},
                {"class_id": "NORMAL", "probability": 0.10, "rank": 2},
            ],
            feature_schema_id="FEATURE",
            input_data_sha256="6" * 64,
            calibration_id="CAL-WITHOUT-RECEIPT",
        )


def test_prediction_rejects_malformed_evidence_lineage() -> None:
    prediction = DataDrivenPrediction(
        prediction_id="P-BAD-LINEAGE",
        episode_id="EP-BAD-LINEAGE",
        model_id="MODEL",
        model_version="1",
        classes=[
            {"class_id": "F-A", "probability": 0.90, "rank": 1},
            {"class_id": "NORMAL", "probability": 0.10, "rank": 2},
        ],
        feature_schema_id="FEATURE",
        input_data_sha256="7" * 64,
        metadata={"evidence_lineage_by_fault": {"F-A": ["not-an-object"]}},
    )

    with pytest.raises(ValueError, match="entries must be objects"):
        prediction.to_fault_ranking(["F-A"])


def test_online_workflow_can_suppress_rejected_data_branch(tmp_path: Path) -> None:
    event = SimulationRunCompleted(
        event_id="SIM-EVENT-V0515-OPEN",
        run_id="heater_stuck_on",
        run_dir=str(SCENE_ROOT / "heater_stuck_on"),
        nominal_run_dir=str(SCENE_ROOT / "nominal"),
    )
    preview, _ = WholeSpacecraftRuntimeEvidenceBuilder(PROTOCOL_ROOT).build(event)
    registered = sorted({item.target_fault_id for item in preview.evidence_definitions})
    probability = 1.0 / (len(registered) + 1)
    ordered = [("NORMAL", probability), *[(fault_id, probability) for fault_id in registered]]
    ordered.sort(key=lambda item: item[0])
    classes = [
        {"class_id": class_id, "probability": value, "rank": index}
        for index, (class_id, value) in enumerate(ordered, start=1)
    ]
    prediction = DataDrivenPrediction(
        prediction_id=stable_id("PRED", {"episode": preview.episode.episode_id, "v": "v0515-open"}),
        episode_id=preview.episode.episode_id,
        model_id="uncertain-sequence-model",
        model_version="1",
        classes=classes,
        feature_schema_id="FEATURE-V0515",
        input_data_sha256="4" * 64,
    )
    permissions = [
        PermissionScope.READ_DATA,
        PermissionScope.READ_KNOWLEDGE,
        PermissionScope.RUN_DIAGNOSIS,
        PermissionScope.WRITE_AUDIT,
        PermissionScope.WRITE_ARTIFACTS,
    ]
    workflow = WholeSpacecraftOnlineDiagnosisWorkflow()
    envelope = TaskEnvelope(
        task_id="V0515-OPEN-SET-WORKFLOW",
        task_type="simulation.online_diagnosis",
        input_payload={
            "simulation_event": event.model_dump(mode="json"),
            "protocol_root": str(PROTOCOL_ROOT),
            "runtime_evidence_builder_id": "legacy.whole_spacecraft.v4",
            "artifact_output_dir": str(tmp_path / "artifacts"),
            "reasoner_mode": "rule",
            "data_driven_prediction": prediction.model_dump(mode="json"),
            "open_set_policy": {
                "minimum_top_probability": 0.5,
                "minimum_top_margin": 0.1,
                "suppress_data_branch_on_reject": True,
            },
        },
        execution_policy=ExecutionPolicy(deterministic_only=True, allow_llm=False),
        permission_scope=permissions,
    )
    result = workflow.invoke(
        envelope,
        WorkflowContext(
            workflow_id=workflow.workflow_id,
            run_id="V0515-OPEN-SET-WORKFLOW",
            permissions=permissions,
        ),
    )
    assert result.status.value == "SUCCEEDED"
    fusion = result.output_payload["candidate_fusion"]
    assert fusion["birrf_result"]["config"]["data_weight"] == 0.0
    assert "DATA_BRANCH_SUPPRESSED_BY_OPEN_SET_POLICY" in fusion["birrf_result"]["conflict_flags"]
    assert fusion["metadata"]["open_set_assessment"]["status"] == "REJECT_UNKNOWN"
    assert result.output_payload["decision_record"]["disposition"] == "ACCEPT"


def test_manifest_round_trip_records_model_family_not_hgb_only() -> None:
    manifest = _manifest(
        model_id="lstm-fault-model",
        runtime=ModelRuntime.PYTORCH,
        semantics=ModelOutputSemantics.LOGITS,
    )
    restored = DataDrivenModelManifest.model_validate_json(manifest.model_dump_json())
    assert restored.runtime == ModelRuntime.PYTORCH
    assert restored.feeds_birrf_directly is True
    assert "HGB" not in json.dumps(restored.model_dump(mode="json"))


def test_online_workflow_can_invoke_registered_non_hgb_model(tmp_path: Path) -> None:
    event = SimulationRunCompleted(
        event_id="SIM-EVENT-V0515-REGISTRY",
        run_id="heater_stuck_on",
        run_dir=str(SCENE_ROOT / "heater_stuck_on"),
        nominal_run_dir=str(SCENE_ROOT / "nominal"),
    )
    preview, _ = WholeSpacecraftRuntimeEvidenceBuilder(PROTOCOL_ROOT).build(event)
    registered = sorted({item.target_fault_id for item in preview.evidence_definitions})
    manifest = DataDrivenModelManifest(
        model_id="thermal-tcn-model",
        model_version="1.2.0",
        display_name="Thermal TCN fixture",
        task_type=DataDrivenTaskType.MULTICLASS_FAULT_CLASSIFICATION,
        runtime=ModelRuntime.PYTORCH,
        output_semantics=ModelOutputSemantics.LOGITS,
        capabilities=[ModelCapability.FAULT_RANKING, ModelCapability.NORMAL_CLASS, ModelCapability.TEMPORAL_CONTEXT],
        supported_fault_ids=registered,
        feature_schema_ids=["FEATURE-THERMAL-TCN"],
    )

    def predict(request: ModelInferenceRequest) -> RawModelOutput:
        logits = {fault_id: -5.0 for fault_id in registered}
        logits["THERMAL_HEATER_STUCK_ON"] = 4.0
        logits["NORMAL"] = -1.0
        return RawModelOutput(
            episode_id=request.episode_id,
            model_id=manifest.model_id,
            model_version=manifest.model_version,
            output_semantics=manifest.output_semantics,
            class_values=logits,
        )

    registry = DataDrivenModelRegistry.for_research()
    registry.register(CallableRawModelProvider(manifest, predict))
    inference_request = ModelInferenceRequest(
        episode_id=preview.episode.episode_id,
        model_id=manifest.model_id,
        model_version=manifest.model_version,
        feature_schema_id="FEATURE-THERMAL-TCN",
        input_data_sha256="5" * 64,
        registered_fault_ids=registered,
        payload={"sequence": [[0.0, 1.0], [1.0, 1.0]]},
    )
    permissions = [
        PermissionScope.READ_DATA,
        PermissionScope.READ_KNOWLEDGE,
        PermissionScope.RUN_DIAGNOSIS,
        PermissionScope.WRITE_AUDIT,
        PermissionScope.WRITE_ARTIFACTS,
    ]
    workflow = WholeSpacecraftOnlineDiagnosisWorkflow(data_model_registry=registry)
    envelope = TaskEnvelope(
        task_id="V0515-REGISTRY-WORKFLOW",
        task_type="simulation.online_diagnosis",
        input_payload={
            "simulation_event": event.model_dump(mode="json"),
            "protocol_root": str(PROTOCOL_ROOT),
            "runtime_evidence_builder_id": "legacy.whole_spacecraft.v4",
            "artifact_output_dir": str(tmp_path / "artifacts"),
            "reasoner_mode": "rule",
            "data_driven_inference_request": inference_request.model_dump(mode="json"),
        },
        execution_policy=ExecutionPolicy(deterministic_only=True, allow_llm=False),
        permission_scope=permissions,
    )
    result = workflow.invoke(
        envelope,
        WorkflowContext(
            workflow_id=workflow.workflow_id,
            run_id="V0515-REGISTRY-WORKFLOW",
            permissions=permissions,
        ),
    )
    assert result.status.value == "SUCCEEDED"
    assert result.output_payload["data_driven_prediction"]["model_family"] == "PYTORCH"
    assert result.output_payload["candidate_fusion"]["data_ranking"]["candidates"][0]["fault_id"] == "THERMAL_HEATER_STUCK_ON"
    assert result.output_payload["decision_record"]["accepted_fault_id"] == "THERMAL_HEATER_STUCK_ON"
