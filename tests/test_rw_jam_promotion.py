from __future__ import annotations

from astrograph.data_driven import (
    BenchmarkReleaseStatus,
    BenchmarkStatus,
    CalibrationMetrics,
    CredibilityStatus,
    DataDrivenModelManifest,
    DataDrivenTaskType,
    ModelBenchmarkMetrics,
    ModelBenchmarkReport,
    ModelCapability,
    ModelOutputSemantics,
    ModelRuntime,
    PerClassMetric,
    ProbabilityCalibrationReceipt,
    ProbabilityCalibrationQualificationReceipt,
    SplitName,
)
from astrograph.integrations import (
    ALevelDatasetEvidence,
    ALevelRunEvidence,
    ActuatorTelemetryContractEvidence,
    ExpertPromotionEvidence,
    KnowledgePromotionEvidence,
    ModelPromotionEvidence,
    PromotionStatus,
    ReviewMode,
    RWJamPromotionGate,
    RWJamPromotionRequest,
)


def _h(char: str) -> str:
    return char * 64


def test_empty_promotion_request_is_blocked() -> None:
    result = RWJamPromotionGate().assess(RWJamPromotionRequest())
    assert result.status == PromotionStatus.BLOCKED
    assert result.score_semantics == "SCREENING_SCORE"
    assert "A_LEVEL_DATASET_EVIDENCE_MISSING" in result.blockers
    assert "QUALIFIED_MODEL_EVIDENCE_MISSING" in result.blockers
    assert "FORMAL_NEO4J_EVIDENCE_MISSING" in result.blockers


def test_complete_evidence_can_reach_formal_ready() -> None:
    channels = [
        "sensor.adcs.rw.speed_rad_s_0",
        "command.adcs.rw.motor_torque_nm_0",
        "estimate.adcs.rw.actual_torque_nm_0",
        "sensor.adcs.rw.motor_current_a_0",
        "estimate.adcs.pointing_error_deg",
    ]
    runs = []
    for split, count in [(SplitName.TRAIN, 30), (SplitName.VALIDATION, 10), (SplitName.TEST, 10)]:
        for index in range(count):
            runs.append(ALevelRunEvidence(
                run_id=f"{split.value}-{index}", pair_id=f"PAIR-{split.value}-{index}", split=split,
                simulator_version="2.11.0", independence_group_id=f"G-{split.value}-{index}",
                formal_training_qualified=True, named_expert_approved=True, observable_channels=channels,
                content_sha256=_h("a"), configuration_sha256=_h("b"),
            ))
    dataset = ALevelDatasetEvidence(
        dataset_manifest_id="RWJAM-A-V1", dataset_manifest_sha256=_h("c"), runs=runs,
        actuator_contract=ActuatorTelemetryContractEvidence(
            wheel_inertia_kg_m2=0.1, motor_torque_constant_nm_per_a=0.02, current_limit_a=2.0,
            sample_period_s=0.1, command_actual_time_aligned=True, saturation_semantics_frozen=True,
            contract_sha256=_h("d"),
        ), split_frozen=True, leakage_audit_passed=True,
    )
    calibration_receipt = ProbabilityCalibrationQualificationReceipt(
        calibration_id="temp-v1", model_id="rw-jam-tcn", model_version="1.0.0",
        calibration_dataset_id="RWJAM-A-V1", calibration_dataset_sha256=_h("2"),
        model_artifact_sha256=_h("e"), calibration_method="TEMPERATURE_SCALING_ON_LOG_PROBABILITIES",
        fit_receipt_sha256=_h("3"), design_sha256=_h("8"), admission_bundle_sha256=_h("9"), calibration_report_sha256=_h("4"),
        validation_expected_calibration_error=0.04, external_validation_expected_calibration_error=0.05,
        calibration_sample_count=10, calibration_group_count=10, development_group_count=10,
        confirmatory_group_count=10, external_validation_dataset_id="RWJAM-EXT-V1",
        external_validation_dataset_sha256=_h("5"), external_validation_identity_sha256=_h("6"),
        external_validation_group_count=10, selective_threshold_receipt_sha256=_h("7"),
        qualification_status="PASS_FOR_ENGINEERING_CALIBRATION_REVIEW",
    )
    manifest = DataDrivenModelManifest(
        model_id="rw-jam-tcn", model_version="1.0.0", display_name="RW Jam TCN",
        task_type=DataDrivenTaskType.MULTICLASS_FAULT_CLASSIFICATION, runtime=ModelRuntime.PYTORCH,
        output_semantics=ModelOutputSemantics.PROBABILITIES,
        capabilities=[ModelCapability.FAULT_RANKING, ModelCapability.CALIBRATED_PROBABILITY],
        supported_fault_ids=["ADCS_RW_JAM"], feature_schema_ids=["adcs.rw_jam.features.v1"],
        calibration_id="temp-v1",
        calibration_receipt_sha256=calibration_receipt.receipt_sha256,
        training_dataset_ids=["RWJAM-A-V1"],
    )
    report = ModelBenchmarkReport(
        status=BenchmarkStatus.PASS, model_id="rw-jam-tcn", model_version="1.0.0",
        evaluated_split=SplitName.TEST, run_ids=[f"TEST-{i}" for i in range(10)],
        run_audit_status=CredibilityStatus.PASS,
        metrics=ModelBenchmarkMetrics(
            sample_count=10, known_sample_count=10, unknown_sample_count=0, accuracy=0.95,
            balanced_accuracy=0.95, macro_f1=0.95, top_k_recall={"1": 0.95},
            calibration=CalibrationMetrics(sample_count=10, negative_log_likelihood=0.1, multiclass_brier_score=0.05, expected_calibration_error=0.04),
            per_class=[PerClassMetric(class_id="ADCS_RW_JAM", support=10, precision=0.95, recall=0.95, f1=0.95)],
        ),
    )
    model = ModelPromotionEvidence(
        manifest=manifest, benchmark_report=report, benchmark_release_status=BenchmarkReleaseStatus.ACCEPTED,
        benchmark_scope_id="scope-a", validation_expected_calibration_error=0.04,
        calibration_receipt=calibration_receipt,
        artifact_sha256=_h("e"), reproducible_training=True,
    )
    knowledge = KnowledgePromotionEvidence(
        backend="neo4j", knowledge_version="kg-v1", health_status="PASS",
        read_only_query_executed=True, query_sha256=_h("f"), matched_fault_ids=["ADCS_RW_JAM"],
        causal_paths=[{"fault_id": "ADCS_RW_JAM", "edge_count": 3}], formal_release_accepted=True,
    )
    expert = ExpertPromotionEvidence(
        reviewer_id="expert-001", reviewer_role="ADCS PHM reviewer", review_mode=ReviewMode.LLM_ASSISTED_HUMAN,
        approved=True, dataset_manifest_sha256=_h("c"), model_artifact_sha256=_h("e"),
        knowledge_query_sha256=_h("f"), review_record_sha256=_h("1"),
    )
    result = RWJamPromotionGate().assess(RWJamPromotionRequest(dataset=dataset, model=model, knowledge=knowledge, expert=expert))
    assert result.status == PromotionStatus.FORMAL_READY
    assert result.score_semantics == "DIAGNOSTIC_CONFIDENCE"
    assert result.blockers == []

    unbound_metric = model.model_copy(update={"validation_expected_calibration_error": 0.01})
    blocked = RWJamPromotionGate().assess(
        RWJamPromotionRequest(dataset=dataset, model=unbound_metric, knowledge=knowledge, expert=expert)
    )
    assert blocked.status == PromotionStatus.BLOCKED
    assert "MODEL_CALIBRATION_METRIC_NOT_RECEIPT_BOUND" in blocked.blockers

    legacy_receipt = ProbabilityCalibrationReceipt(
        calibration_id="temp-v1", model_id="rw-jam-tcn", model_version="1.0.0",
        calibration_dataset_id="RWJAM-A-V1", calibration_split_sha256=_h("2"),
        calibration_method="TEMPERATURE_SCALING", calibration_report_sha256=_h("3"),
        validation_expected_calibration_error=0.04, calibration_sample_count=10, calibration_group_count=10,
    )
    legacy_manifest = manifest.model_copy(update={"calibration_receipt_sha256": legacy_receipt.receipt_sha256})
    legacy_model = model.model_copy(update={"manifest": legacy_manifest, "calibration_receipt": legacy_receipt})
    legacy = RWJamPromotionGate().assess(RWJamPromotionRequest(dataset=dataset, model=legacy_model, knowledge=knowledge, expert=expert))
    assert legacy.status == PromotionStatus.BLOCKED
    assert "MODEL_EXTERNAL_CALIBRATION_QUALIFICATION_MISSING" in legacy.blockers
