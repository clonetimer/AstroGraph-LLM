from .metrics import HypothesisEvaluation, SetMetric, evaluate_hypothesis_set
from .online_acceptance import (
    COMPOSITE_SCENARIOS,
    OUT_OF_SCOPE_SCENARIOS,
    REGISTERED_EXPECTATIONS,
    REQUIRED_TOOL_IDS,
    ArtifactAudit,
    OnlineAcceptanceReport,
    ScenarioExpectation,
    SimulationSourceAudit,
    StabilityReport,
    all_expected_scenarios,
    audit_artifact_dir,
    audit_simulation_output_root,
    build_stability_report,
    load_and_validate_summary,
    scenario_expectation,
    validate_summary,
)
from .perturbations import EvidencePerturbationSpec, PerturbationAudit, apply_evidence_perturbations
from .kg_gain import (
    ABLATION_ARMS,
    FixedCaseSpec,
    build_failure_boundary_report,
    evaluate_kg_gain_ablation,
    evaluate_neo4j_fallback,
    v0530_fixed_case_specs,
    write_v0530_reports,
)
from .llm_method_ablation import LLMMethodAblationRow, annotate_net_diagnostic_effect, summarize_llm_method_ablation
from .next_state_gate import (
    audit_activity_axis_artifacts,
    audit_frozen_predictions,
    audit_linked_activity_axis_predictions,
    audit_online_kg_fallback_report,
    audit_kg_redundancy_quality_gate,
    build_next_state_gate_report,
    write_next_state_gate_report,
)
from .rw_active_discovery import (
    RWActiveDiscoveryComparisonReport,
    RWActiveDiscoveryComparisonService,
    RWActiveDiscoveryPilotReport,
    RWActiveDiscoveryPilotService,
)

__all__ = [
    "HypothesisEvaluation",
    "SetMetric",
    "evaluate_hypothesis_set",
    "EvidencePerturbationSpec",
    "PerturbationAudit",
    "apply_evidence_perturbations",
    "COMPOSITE_SCENARIOS",
    "OUT_OF_SCOPE_SCENARIOS",
    "REGISTERED_EXPECTATIONS",
    "REQUIRED_TOOL_IDS",
    "ArtifactAudit",
    "OnlineAcceptanceReport",
    "ScenarioExpectation",
    "SimulationSourceAudit",
    "StabilityReport",
    "all_expected_scenarios",
    "audit_artifact_dir",
    "audit_simulation_output_root",
    "build_stability_report",
    "load_and_validate_summary",
    "scenario_expectation",
    "validate_summary",
    "ABLATION_ARMS",
    "FixedCaseSpec",
    "build_failure_boundary_report",
    "evaluate_kg_gain_ablation",
    "LLMMethodAblationRow",
    "annotate_net_diagnostic_effect",
    "summarize_llm_method_ablation",
    "evaluate_neo4j_fallback",
    "v0530_fixed_case_specs",
    "write_v0530_reports",
    "audit_activity_axis_artifacts",
    "audit_frozen_predictions",
    "audit_linked_activity_axis_predictions",
    "audit_online_kg_fallback_report",
    "audit_kg_redundancy_quality_gate",
    "build_next_state_gate_report",
    "write_next_state_gate_report",
    "RWActiveDiscoveryPilotReport",
    "RWActiveDiscoveryPilotService",
    "RWActiveDiscoveryComparisonReport",
    "RWActiveDiscoveryComparisonService",
    "LLM_METHOD_ARM_ORDER",
    "confirmatory_readiness",
    "build_confirmatory_execution_receipt",
    "build_confirmatory_dataset_snapshot_receipt",
    "evaluate_confirmatory_promotion",
    "exact_mcnemar_two_sided",
    "exact_group_direction_sign_two_sided",
    "freeze_confirmatory_case_manifest",
    "load_confirmatory_design",
    "materialize_confirmatory_dataset_snapshot",
    "paired_arm_effect",
    "validate_confirmatory_case_manifest",
    "verify_confirmatory_execution_identity",
    "verify_confirmatory_dataset_snapshot_receipt",
    "acquire_model_runtime_attestation",
    "build_confirmatory_client_identity",
    "build_confirmatory_cyclonedx_sbom",
    "build_confirmatory_inference_runtime_snapshot",
    "build_confirmatory_python_environment_lock",
    "build_confirmatory_runtime_identity_admission",
    "build_model_runtime_attestation",
    "load_confirmatory_python_environment_lock",
    "load_model_runtime_attestation",
    "load_model_runtime_spec",
    "verify_confirmatory_python_environment_lock",
    "verify_confirmatory_runtime_identity_drift",
    "verify_model_runtime_attestation",
]

from .llm_method_campaign import (
    ARM_ORDER as LLM_METHOD_ARM_ORDER,
    build_confirmatory_execution_receipt,
    build_confirmatory_dataset_snapshot_receipt,
    confirmatory_readiness,
    evaluate_confirmatory_promotion,
    exact_group_direction_sign_two_sided,
    exact_mcnemar_two_sided,
    freeze_confirmatory_case_manifest,
    load_confirmatory_design,
    materialize_confirmatory_dataset_snapshot,
    paired_arm_effect,
    validate_confirmatory_case_manifest,
    verify_confirmatory_execution_identity,
    verify_confirmatory_dataset_snapshot_receipt,
)

from .confirmatory_runtime_identity import (
    acquire_model_runtime_attestation,
    build_confirmatory_client_identity,
    build_confirmatory_cyclonedx_sbom,
    build_confirmatory_inference_runtime_snapshot,
    build_confirmatory_python_environment_lock,
    build_confirmatory_runtime_identity_admission,
    build_model_runtime_attestation,
    load_confirmatory_python_environment_lock,
    load_model_runtime_attestation,
    load_model_runtime_spec,
    verify_confirmatory_python_environment_lock,
    verify_confirmatory_runtime_identity_drift,
    verify_model_runtime_attestation,
)


from .calibration_campaign import (
    CalibrationContext,
    evaluate_calibration_transfer_campaign,
    evaluate_fixed_threshold_transfer,
    group_bootstrap_mean_ci,
)
from .calibration_fitting import (
    apply_temperature_scaler_receipt,
    fit_temperature_scaler_receipt,
    qualify_temperature_calibration,
    temperature_scale_probability_map,
    verify_temperature_scaler_fit_receipt,
)
from .conformal_candidates import (
    apply_group_conformal_candidate_receipt,
    fit_group_conformal_candidate_receipt,
    verify_group_conformal_receipt,
)
from .fusion_dependence import (
    BranchPrediction,
    FusionEvaluationCase,
    TruthRecord as FusionTruthRecord,
    audit_birrf_leave_branch_out,
    audit_branch_dependence,
)

__all__ += [
    "CalibrationContext",
    "evaluate_calibration_transfer_campaign",
    "evaluate_fixed_threshold_transfer",
    "group_bootstrap_mean_ci",
    "temperature_scale_probability_map",
    "fit_temperature_scaler_receipt",
    "verify_temperature_scaler_fit_receipt",
    "apply_temperature_scaler_receipt",
    "qualify_temperature_calibration",
    "apply_group_conformal_candidate_receipt",
    "fit_group_conformal_candidate_receipt",
    "verify_group_conformal_receipt",
    "BranchPrediction",
    "FusionEvaluationCase",
    "FusionTruthRecord",
    "audit_birrf_leave_branch_out",
    "audit_branch_dependence",
]

from .replication_r22 import (
    OPS_FAMILY,
    ESA_FAMILY,
    admit_new_holdout_groups,
    build_esa_preprocessed_input_receipt,
    build_new_holdout_intake_receipt,
    build_r22_readiness,
    esa_channel_payload_identity,
    evaluate_esa_transport_replication,
    evaluate_ops_replication,
    generate_esa_transport_predictions,
    generate_ops_replication_predictions,
    load_replication_plan,
    ops_r20_frozen_baseline_prediction,
    ops_r21_frozen_remediation_prediction,
    require_esa_no_superiority_contrast,
    verify_new_holdout_admission,
)

__all__ += [
    "OPS_FAMILY",
    "ESA_FAMILY",
    "admit_new_holdout_groups",
    "build_esa_preprocessed_input_receipt",
    "build_new_holdout_intake_receipt",
    "build_r22_readiness",
    "esa_channel_payload_identity",
    "evaluate_esa_transport_replication",
    "evaluate_ops_replication",
    "generate_esa_transport_predictions",
    "generate_ops_replication_predictions",
    "load_replication_plan",
    "ops_r20_frozen_baseline_prediction",
    "ops_r21_frozen_remediation_prediction",
    "require_esa_no_superiority_contrast",
    "verify_new_holdout_admission",
]

from .dxc_discrete_consistency import (
    DXCDiscreteConsistencyConfig,
    DXCDiscreteConsistencyResult,
    DXCDiscreteFaultEvent,
    diagnose_discrete_state_consistency,
)
from .dxc_hybrid_fusion_r26 import DXCFusionPolicy, DXCR26FusedPrediction, fuse_dxc_r26
from .esa_r26_frozen import (
    build_r26_esa_holdout_binding,
    build_r26_frozen_esa_detector_artifact,
    run_r26_frozen_esa_prediction,
    verify_r26_frozen_esa_detector_artifact,
)

__all__ += [
    "DXCDiscreteConsistencyConfig",
    "DXCDiscreteConsistencyResult",
    "DXCDiscreteFaultEvent",
    "diagnose_discrete_state_consistency",
    "DXCFusionPolicy",
    "DXCR26FusedPrediction",
    "fuse_dxc_r26",
    "build_r26_esa_holdout_binding",
    "build_r26_frozen_esa_detector_artifact",
    "run_r26_frozen_esa_prediction",
    "verify_r26_frozen_esa_detector_artifact",
]

from .dxc_r27_expected_response import DXCR27Prediction, assemble_dxc_r27_prediction, predict_dxc_r27

__all__ += ["DXCR27Prediction", "assemble_dxc_r27_prediction", "predict_dxc_r27"]

from .dxc_r29_reduced_order import DXCR29Prediction, assemble_dxc_r29_prediction, predict_dxc_r29

__all__ += ["DXCR29Prediction", "assemble_dxc_r29_prediction", "predict_dxc_r29"]
from .diagnostic_regression import (
    BenchmarkCase,
    BenchmarkSnapshot,
    DetectionMetrics,
    DiagnosticBenchmarkMetrics,
    DiagnosticRegressionBoard,
    EvidenceClass,
    IdentificationMetrics,
    LocalizationMetrics,
    RegressionBoardRow,
    SnapshotComparison,
    build_regression_board,
    compare_snapshots,
    evaluate_cases,
    evaluate_diagnostic_benchmark,
)

__all__ += [
    "BenchmarkCase",
    "BenchmarkSnapshot",
    "DetectionMetrics",
    "DiagnosticBenchmarkMetrics",
    "DiagnosticRegressionBoard",
    "EvidenceClass",
    "IdentificationMetrics",
    "LocalizationMetrics",
    "RegressionBoardRow",
    "SnapshotComparison",
    "build_regression_board",
    "compare_snapshots",
    "evaluate_cases",
    "evaluate_diagnostic_benchmark",
]
