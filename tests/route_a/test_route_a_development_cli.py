from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from astrograph.assurance.analysis_plan import PrimaryAnalysisPlanV2, RouteANonInferiorityMargins
from astrograph.experiments.route_a_protocol import (
    FrozenPolicyPredictionBundle,
    FrozenProbabilityPrediction,
    OfflineTruthRow,
    OfflineTruthTable,
)
from tools.run_route_a_development_selection import run_manifest

H = "1" * 64
SEED = "2" * 64


def _write(path: Path, payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, indent=2).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _prediction(sample: str, group: str, truth: str) -> FrozenProbabilityPrediction:
    probabilities = {"A": 0.9, "B": 0.1} if truth == "A" else {"A": 0.1, "B": 0.9}
    row = FrozenProbabilityPrediction(
        sample_id=sample,
        group_id=group,
        probabilities=probabilities,
        predicted_id=truth,
        runtime_trace_sha256=H,
    )
    return row.model_copy(update={"row_sha256": row.compute_sha256()})


def _bundle(candidate: str, fusion_hash: str) -> FrozenPolicyPredictionBundle:
    item = FrozenPolicyPredictionBundle(
        candidate_id=candidate,
        base_policy_id=candidate,
        confidence_threshold=0.0,
        source_tree_sha256=H,
        model_package_sha256=H,
        temporal_policy_sha256=H,
        evidence_package_sha256=fusion_hash,
        fusion_policy_sha256=fusion_hash,
        development_split_sha256=H,
        prediction_generation_code_sha256=H,
        predictions=(
            _prediction("a", "3", "A"),
            _prediction("b", "8", "B"),
        ),
    )
    return item.model_copy(update={"bundle_sha256": item.compute_sha256()})


def test_hashed_manifest_runs_and_tampering_fails(tmp_path: Path) -> None:
    plan = PrimaryAnalysisPlanV2(
        plan_id="cli",
        status="FROZEN",
        independent_unit="MONTE_CARLO_GROUP",
        development_groups=("3", "8"),
        candidate_ids=("A0", "A7"),
        candidate_thresholds={"A0": 0.0, "A7": 0.0},
        candidate_fusion_policy_sha256s={"A0": H, "A7": "3" * 64},
        candidate_evidence_package_sha256s={"A0": H, "A7": "3" * 64},
        source_tree_sha256=H,
        model_package_sha256=H,
        temporal_policy_sha256=H,
        development_split_sha256=H,
        prediction_generation_code_sha256=H,
        noninferiority_margins=RouteANonInferiorityMargins(
            max_accuracy_drop=0.0,
            max_coverage_drop=0.0,
            max_selective_risk_increase=0.0,
            max_nll_increase=0.0,
            max_brier_increase=0.0,
            max_ece_increase=0.0,
            max_aurc_increase=0.0,
            max_worst_group_accuracy_drop=0.0,
        ),
        bootstrap_replicates=1000,
        bootstrap_seed_sha256=SEED,
    )
    a0 = _bundle("A0", H)
    a7 = _bundle("A7", "3" * 64)
    truth = OfflineTruthTable(
        dataset_id="cli",
        development_split_sha256=H,
        eligible_prediction_bundle_sha256s=(a0.compute_sha256(), a7.compute_sha256()),
        rows=(
            OfflineTruthRow(sample_id="a", group_id="3", true_id="A"),
            OfflineTruthRow(sample_id="b", group_id="8", true_id="B"),
        ),
    )

    refs = []
    for name, payload in (
        ("plan.json", plan.model_dump(mode="json")),
        ("truth.json", truth.model_dump(mode="json")),
        ("a0.json", a0.model_dump(mode="json")),
        ("a7.json", a7.model_dump(mode="json")),
    ):
        digest = _write(tmp_path / name, payload)
        refs.append((name, digest))
    manifest = {
        "schema_version": "astrograph-route-a-development-run-manifest.v1",
        "analysis_plan": {"path": refs[0][0], "sha256": refs[0][1]},
        "truth_table": {"path": refs[1][0], "sha256": refs[1][1]},
        "prediction_bundles": [
            {"path": refs[2][0], "sha256": refs[2][1]},
            {"path": refs[3][0], "sha256": refs[3][1]},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = run_manifest(manifest_path)
    assert result["selected_candidate_id"] in {"A0", "A7"}
    assert result["result_status"] == "DEVELOPMENT_SELECTED_NOT_CONFIRMATORY"

    (tmp_path / "a7.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_manifest(manifest_path)
