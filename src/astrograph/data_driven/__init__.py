"""Data-driven diagnosis compatibility namespace.

Runtime and research surfaces are now separated as ``data_driven.runtime`` and
``data_driven.lab``. Historical root-level imports remain available through lazy
attribute loading so importing one runtime contract does not eagerly import training
campaigns and research models.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    # unified platform (Phase 3)
    "DatasetTaskSpec": ".unified_platform",
    "make_unified_platform_model": ".unified_platform",
    # feedback cycle (Phase 2)
    "E2E_MIN_IMPROVEMENT": ".feedback",
    "FeedbackCandidate": ".feedback",
    "FeedbackCycleLedger": ".feedback",
    "FeedbackCycleRecord": ".feedback",
    "FeedbackCycleStatus": ".feedback",
    "SelectionDecision": ".feedback",
    "TASK_MAX_REGRESSION": ".feedback",
    "TaskMetrics": ".feedback",
    "advance_cycle": ".feedback",
    "build_upgrade_proposal": ".feedback",
    "evaluate_candidate": ".feedback",
    # adapters
    "adapt_raw_model_output": ".adapters",
    "ranked_class_ids": ".adapters",
    # contracts
    "DataDrivenModelManifest": ".contracts",
    "DataDrivenTaskType": ".contracts",
    "ModelCapability": ".contracts",
    "ProbabilityCalibrationReceipt": ".contracts",
    "ProbabilityCalibrationQualificationReceipt": ".contracts",
    "ModelInferenceRequest": ".contracts",
    "ObservationScope": ".contracts",
    "ModelOutputSemantics": ".contracts",
    "ModelRuntime": ".contracts",
    "RawModelOutput": ".contracts",
    # credibility / open set
    "CalibrationMetrics": ".credibility",
    "CredibilityStatus": ".credibility",
    "IndependentRunAudit": ".credibility",
    "IndependentRunRecord": ".credibility",
    "RunProvenanceType": ".credibility",
    "OpenSetAssessment": ".credibility",
    "OpenSetPolicy": ".credibility",
    "OpenSetStatus": ".credibility",
    "SplitName": ".credibility",
    "TemperatureScaler": ".credibility",
    "TemperatureScalingRecord": ".credibility",
    "assess_open_set": ".credibility",
    "audit_independent_runs": ".credibility",
    "calibration_metrics": ".credibility",
    "resolve_run_scenario_dir": ".credibility",
    # campaign
    "CampaignExecutionReport": ".campaign",
    "CampaignPlanStatus": ".campaign",
    "CampaignRunPlan": ".campaign",
    "CampaignScenario": ".campaign",
    "CampaignSplitCounts": ".campaign",
    "CampaignVariant": ".campaign",
    "IndependentRunCampaignPlan": ".campaign",
    "IndependentRunCampaignSpec": ".campaign",
    "build_independent_run_campaign_plan": ".campaign",
    "build_inventory_from_campaign_plan": ".campaign",
    "campaign_plan_sha256": ".campaign",
    "execute_independent_run_campaign": ".campaign",
    # activity axes
    "ActivityAxis": ".activity_axes",
    "ActivityAxisGate": ".activity_axes",
    "ActivityAxisGateDecision": ".activity_axes",
    "ActivityAxisGateStatus": ".activity_axes",
    "RuntimeActivityAxisCapability": ".activity_axes",
    "RuntimeActivityAxisCapabilityManifest": ".activity_axes",
    "decide_campaign_activity_axes": ".activity_axes",
    "default_v0522_activity_axis_manifest": ".activity_axes",
    "load_activity_axis_manifest": ".activity_axes",
    "manifest_sha256": ".activity_axes",
    "requested_axes_from_campaign_spec": ".activity_axes",
    "write_activity_axis_manifest": ".activity_axes",
    # benchmark
    "BenchmarkClaimType": ".benchmark",
    "BenchmarkEvidenceScope": ".benchmark",
    "BenchmarkPredictionRecord": ".benchmark",
    "BenchmarkReleaseDecision": ".benchmark",
    "BenchmarkReleaseGate": ".benchmark",
    "BenchmarkReleaseStatus": ".benchmark",
    "BenchmarkStatus": ".benchmark",
    "ModelBenchmarkMetrics": ".benchmark",
    "ModelBenchmarkReport": ".benchmark",
    "PerClassMetric": ".benchmark",
    "evaluate_model_benchmark": ".benchmark",
    # runtime models/providers/registry
    "ClassProbability": ".models",
    "DataDrivenPrediction": ".models",
    "DataDrivenRankingProvider": ".models",
    "StaticDataDrivenRankingProvider": ".models",
    "CallableRawModelProvider": ".providers",
    "ExternalServiceModelProvider": ".providers",
    "RawDataDrivenModelProvider": ".providers",
    "SklearnProbabilityProvider": ".providers",
    "urllib_json_transport": ".providers",
    "DataDrivenModelRegistry": ".registry",
    # selection
    "FaultModelBinding": ".selection",
    "FaultModelBindingRegistry": ".selection",
    "FaultModelBindingSet": ".selection",
    "ModelSelectionCandidate": ".selection",
    "ModelSelectionDecision": ".selection",
    "ModelSelectionRequest": ".selection",
    "ModelSelectionStatus": ".selection",
    # research HGB
    "CASE_TO_FAULT_ID": ".whole_spacecraft_hgb",
    "COMPOSITE_MEMBERS": ".whole_spacecraft_hgb",
    "WholeSpacecraftHGBModel": ".whole_spacecraft_hgb",
    "WholeSpacecraftHGBProvider": ".whole_spacecraft_hgb",
    "predict_scene": ".whole_spacecraft_hgb",
    "train_whole_spacecraft_hgb": ".whole_spacecraft_hgb",
    # catalog
    "FaultCatalogValidationIssue": ".fault_catalog",
    "FaultCatalogValidationReport": ".fault_catalog",
    "FaultModeCatalog": ".fault_catalog",
    "FaultModeDefinition": ".fault_catalog",
    "FaultSubsystem": ".fault_catalog",
    "FaultValidationRequirement": ".fault_catalog",
    "TelemetryChannelRole": ".fault_catalog",
    "TelemetryChannelSpec": ".fault_catalog",
    # benchmark pipeline
    "BenchmarkComparisonPolicy": ".benchmark_pipeline",
    "BenchmarkModelCandidate": ".benchmark_pipeline",
    "FaultModelRecommendation": ".benchmark_pipeline",
    "FaultModelRecommendationStatus": ".benchmark_pipeline",
    "ModelBenchmarkComparisonReport": ".benchmark_pipeline",
    "compare_model_benchmarks": ".benchmark_pipeline",
    # temporal autoencoder
    "TemporalAutoencoderTrainingSummary": ".temporal_autoencoder",
    "TemporalLSTMAutoencoderConfig": ".temporal_autoencoder",
    "TemporalLSTMAutoencoderModel": ".temporal_autoencoder",
    "TemporalLSTMAutoencoderProvider": ".temporal_autoencoder",
    "fit_temporal_lstm_autoencoder": ".temporal_autoencoder",
    "load_temporal_lstm_autoencoder": ".temporal_autoencoder",
    "save_temporal_lstm_autoencoder": ".temporal_autoencoder",
    # cross condition
    "ACTIVITY_AXIS_SOURCES": ".cross_condition",
    "ActivityAxisPredictionRow": ".cross_condition",
    "ActivityAxisSample": ".cross_condition",
    "FrozenModelArtifact": ".cross_condition",
    "FrozenModelBundleManifest": ".cross_condition",
    "discover_activity_axis_samples": ".cross_condition",
    "feature_row_for_run": ".cross_condition",
    "freeze_seed_models": ".cross_condition",
    "load_frozen_models": ".cross_condition",
    "run_activity_axis_inference": ".cross_condition",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})