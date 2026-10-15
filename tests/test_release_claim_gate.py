from __future__ import annotations

from astrograph.data_driven import BenchmarkClaimType, BenchmarkReleaseDecision, BenchmarkReleaseStatus
from astrograph.release_gate import (
    ReleaseClaim,
    ReleaseClaimDisposition,
    ReleaseClaimGate,
    ReleaseClaimManifest,
    ReleaseClaimStatus,
)


def _benchmark_decisions() -> list[BenchmarkReleaseDecision]:
    return [
        BenchmarkReleaseDecision(
            status=BenchmarkReleaseStatus.ACCEPTED,
            claim_type=BenchmarkClaimType.SEED_REGRESSION_RESULT,
            report_model_id="MODEL-A",
            report_model_version="0.5.19",
            allowed_claim_boundary="seed only",
        ),
        BenchmarkReleaseDecision(
            status=BenchmarkReleaseStatus.BLOCKED,
            claim_type=BenchmarkClaimType.CROSS_CONDITION_GENERALIZATION,
            report_model_id="MODEL-A",
            report_model_version="0.5.19",
            blocking_reasons=["SEED_ONLY_SCOPE_CANNOT_SUPPORT_GENERALIZATION"],
            allowed_claim_boundary="cross condition only with represented axes",
        ),
    ]


def test_release_claim_gate_accepts_claimed_seed_and_not_claimed_broad_capability() -> None:
    manifest = ReleaseClaimManifest(
        release_id="REL",
        claims=[
            ReleaseClaim(
                claim_id="seed",
                claim_type=BenchmarkClaimType.SEED_REGRESSION_RESULT,
                statement="seed regression claim",
                required_model_ids=["MODEL-A"],
            ),
            ReleaseClaim(
                claim_id="cross-condition",
                claim_type=BenchmarkClaimType.CROSS_CONDITION_GENERALIZATION,
                disposition=ReleaseClaimDisposition.NOT_CLAIMED,
                statement="cross-condition generalization is not claimed",
            ),
        ],
    )

    result = ReleaseClaimGate().evaluate(manifest, _benchmark_decisions())

    assert result.status == ReleaseClaimStatus.ACCEPTED
    assert result.blocking_reasons == []
    assert [item.status for item in result.decisions] == [ReleaseClaimStatus.ACCEPTED, ReleaseClaimStatus.ACCEPTED]


def test_release_claim_gate_blocks_claimed_cross_condition_without_release_evidence() -> None:
    manifest = ReleaseClaimManifest(
        release_id="REL",
        claims=[
            ReleaseClaim(
                claim_id="cross-condition",
                claim_type=BenchmarkClaimType.CROSS_CONDITION_GENERALIZATION,
                statement="cross-condition generalization claim",
                required_model_ids=["MODEL-A"],
            ),
        ],
    )

    result = ReleaseClaimGate().evaluate(manifest, _benchmark_decisions())

    assert result.status == ReleaseClaimStatus.BLOCKED
    decision = result.decisions[0]
    assert decision.status == ReleaseClaimStatus.BLOCKED
    assert "CLAIM_NOT_SUPPORTED_BY_BENCHMARK_RELEASE_GATE" in decision.blocking_reasons
    assert "MODEL-A:SEED_ONLY_SCOPE_CANNOT_SUPPORT_GENERALIZATION" in decision.blocking_reasons


def test_release_claim_gate_blocks_missing_required_model_decision() -> None:
    manifest = ReleaseClaimManifest(
        release_id="REL",
        claims=[
            ReleaseClaim(
                claim_id="seed",
                claim_type=BenchmarkClaimType.SEED_REGRESSION_RESULT,
                statement="seed regression claim",
                required_model_ids=["MODEL-A", "MODEL-B"],
            ),
        ],
    )

    result = ReleaseClaimGate().evaluate(manifest, _benchmark_decisions())

    assert result.status == ReleaseClaimStatus.BLOCKED
    assert "seed:REQUIRED_MODEL_RELEASE_DECISION_MISSING:MODEL-B" in result.blocking_reasons
