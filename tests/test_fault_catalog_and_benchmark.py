from __future__ import annotations

from astrograph.data_driven import (
    BenchmarkClaimType,
    BenchmarkComparisonPolicy,
    BenchmarkEvidenceScope,
    BenchmarkStatus,
    CredibilityStatus,
    DataDrivenModelManifest,
    DataDrivenTaskType,
    FaultModeCatalog,
    FaultModeDefinition,
    FaultModelRecommendationStatus,
    FaultSubsystem,
    FaultValidationRequirement,
    ModelBenchmarkMetrics,
    ModelBenchmarkReport,
    ModelCapability,
    ModelOutputSemantics,
    ModelRuntime,
    PerClassMetric,
    SplitName,
    TelemetryChannelSpec,
    compare_model_benchmarks,
)


def _catalog() -> FaultModeCatalog:
    return FaultModeCatalog(
        catalog_id="test-catalog",
        catalog_version="0.1.0",
        fault_modes=[
            FaultModeDefinition(
                fault_id="FAULT_A",
                display_name="Fault A",
                subsystem=FaultSubsystem.ADCS,
                description="Synthetic fault for governed model comparison.",
                telemetry_channels=[TelemetryChannelSpec(channel="sensor.a")],
                validation=FaultValidationRequirement(minimum_independent_runs=1),
            )
        ],
    )


def _report(model_id: str, *, f1: float, recall: float, precision: float) -> ModelBenchmarkReport:
    return ModelBenchmarkReport(
        status=BenchmarkStatus.PASS,
        model_id=model_id,
        model_version="1.0.0",
        evaluated_split=SplitName.TEST,
        run_ids=[f"run-{model_id}"],
        run_audit_status=CredibilityStatus.PASS,
        metrics=ModelBenchmarkMetrics(
            sample_count=10,
            known_sample_count=10,
            unknown_sample_count=0,
            accuracy=f1,
            balanced_accuracy=f1,
            macro_f1=f1,
            top_k_recall={"1": f1, "3": 1.0, "5": 1.0},
            per_class=[
                PerClassMetric(
                    class_id="FAULT_A",
                    support=10,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                )
            ],
        ),
    )


def _scope(model_id: str) -> tuple[str, BenchmarkEvidenceScope]:
    return (
        f"{model_id}@1.0.0",
        BenchmarkEvidenceScope(
            scope_id=f"scope-{model_id}",
            dataset_manifest_id="dataset-v1",
            supported_axes=[],
            seed_only=True,
            frozen_split=True,
            test_used_for_tuning=False,
        ),
    )


def test_fault_catalog_validates_model_manifest() -> None:
    catalog = _catalog()
    manifest = DataDrivenModelManifest(
        model_id="model-a",
        model_version="1.0.0",
        display_name="Model A",
        task_type=DataDrivenTaskType.MULTICLASS_FAULT_CLASSIFICATION,
        runtime=ModelRuntime.SCIKIT_LEARN,
        output_semantics=ModelOutputSemantics.PROBABILITIES,
        capabilities=[ModelCapability.FAULT_RANKING],
        supported_fault_ids=["FAULT_A"],
    )
    assert catalog.validate_manifest(manifest).passed is True
    invalid = manifest.model_copy(update={"supported_fault_ids": ["UNKNOWN_FAULT"]})
    report = catalog.validate_manifest(invalid)
    assert report.passed is False
    assert report.issues[0].code == "MODEL_SUPPORTS_UNKNOWN_FAULT"


def test_benchmark_comparison_recommends_best_model_but_disables_binding() -> None:
    reports = [
        _report("model-a", f1=0.90, recall=0.92, precision=0.88),
        _report("model-b", f1=0.75, recall=0.80, precision=0.72),
    ]
    scopes = dict([_scope("model-a"), _scope("model-b")])
    comparison = compare_model_benchmarks(
        catalog=_catalog(),
        reports=reports,
        scopes=scopes,
        claim_type=BenchmarkClaimType.SEED_REGRESSION_RESULT,
        policy=BenchmarkComparisonPolicy(minimum_score_gap=0.01),
    )
    recommendation = comparison.recommendations[0]
    assert recommendation.status == FaultModelRecommendationStatus.RECOMMENDED
    assert recommendation.selected_model_id == "model-a"
    assert recommendation.requires_human_review is True
    assert len(comparison.proposed_binding_set.bindings) == 1
    assert comparison.proposed_binding_set.bindings[0].enabled is False


def test_benchmark_comparison_blocks_missing_scope() -> None:
    comparison = compare_model_benchmarks(
        catalog=_catalog(),
        reports=[_report("model-a", f1=0.9, recall=0.9, precision=0.9)],
        scopes={},
    )
    recommendation = comparison.recommendations[0]
    assert recommendation.status == FaultModelRecommendationStatus.NO_ELIGIBLE_MODEL
    assert "BENCHMARK_EVIDENCE_SCOPE_MISSING" in recommendation.candidates[0].blocking_reasons
