from __future__ import annotations

from astrograph.data_driven import (
    BenchmarkClaimType,
    BenchmarkEvidenceScope,
    BenchmarkReleaseGate,
    BenchmarkReleaseStatus,
    BenchmarkStatus,
    CredibilityStatus,
    ModelBenchmarkMetrics,
    ModelBenchmarkReport,
    SplitName,
)


HASH_A = "a" * 64


def _report() -> ModelBenchmarkReport:
    return ModelBenchmarkReport(
        status=BenchmarkStatus.PASS,
        model_id="MODEL-RF",
        model_version="0.5.17",
        evaluated_split=SplitName.TEST,
        run_ids=["RUN-1", "RUN-2"],
        run_audit_status=CredibilityStatus.PASS,
        metrics=ModelBenchmarkMetrics(
            sample_count=2,
            known_sample_count=2,
            unknown_sample_count=0,
            accuracy=1.0,
            balanced_accuracy=1.0,
            macro_f1=1.0,
            top_k_recall={"1": 1.0},
        ),
    )


def _seed_only_scope() -> BenchmarkEvidenceScope:
    return BenchmarkEvidenceScope(
        scope_id="SCOPE-SEED",
        dataset_manifest_id="DATASET-V0517",
        dataset_manifest_sha256=HASH_A,
        supported_axes=["seed"],
        unsupported_axes=[
            "severity",
            "initial_state",
            "operating_condition",
            "fault_timing",
            "noise",
            "dropout",
        ],
        seed_only=True,
        frozen_split=True,
        test_used_for_tuning=False,
        flight_data_present=False,
        satellite_instance_count=1,
    )


def test_release_gate_accepts_seed_regression_claim_for_seed_only_scope() -> None:
    decision = BenchmarkReleaseGate().decide(
        _report(),
        _seed_only_scope(),
        claim_type=BenchmarkClaimType.SEED_REGRESSION_RESULT,
    )

    assert decision.status == BenchmarkReleaseStatus.ACCEPTED
    assert decision.blocking_reasons == []
    assert "Seed-level simulator regression" in decision.allowed_claim_boundary


def test_release_gate_blocks_cross_condition_claim_for_seed_only_scope() -> None:
    decision = BenchmarkReleaseGate().decide(
        _report(),
        _seed_only_scope(),
        claim_type=BenchmarkClaimType.CROSS_CONDITION_GENERALIZATION,
    )

    assert decision.status == BenchmarkReleaseStatus.BLOCKED
    assert "MISSING_CROSS_CONDITION_AXES" in decision.blocking_reasons
    assert "SEED_ONLY_SCOPE_CANNOT_SUPPORT_GENERALIZATION" in decision.blocking_reasons
    assert any(item.startswith("missing_axes=") for item in decision.warnings)


def test_release_gate_blocks_cross_satellite_and_flight_claims_without_evidence() -> None:
    gate = BenchmarkReleaseGate()
    cross_sat = gate.decide(
        _report(),
        _seed_only_scope(),
        claim_type=BenchmarkClaimType.CROSS_SATELLITE_GENERALIZATION,
    )
    flight = gate.decide(
        _report(),
        _seed_only_scope(),
        claim_type=BenchmarkClaimType.FLIGHT_VALIDATION,
    )

    assert cross_sat.status == BenchmarkReleaseStatus.BLOCKED
    assert "SINGLE_SATELLITE_SCOPE_CANNOT_SUPPORT_CROSS_SATELLITE" in cross_sat.blocking_reasons
    assert flight.status == BenchmarkReleaseStatus.BLOCKED
    assert flight.blocking_reasons == ["NO_FLIGHT_DATA_PRESENT"]


def test_release_gate_blocks_if_test_split_was_used_for_tuning() -> None:
    scope = _seed_only_scope().model_copy(update={"test_used_for_tuning": True})
    decision = BenchmarkReleaseGate().decide(
        _report(),
        scope,
        claim_type=BenchmarkClaimType.SEED_REGRESSION_RESULT,
    )

    assert decision.status == BenchmarkReleaseStatus.BLOCKED
    assert decision.blocking_reasons == ["TEST_USED_FOR_TUNING"]

