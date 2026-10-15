from __future__ import annotations

import json
from pathlib import Path

from astrograph.governance.spacecraft_convergence import (
    ConvergenceDisposition,
    evaluate_spacecraft_scope_convergence,
    load_spacecraft_scope_convergence,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/release_claims/spacecraft_scope_convergence_r32.v1.json"


def test_r32_spacecraft_feature_freeze_passes_without_overclaiming_external_qualification() -> None:
    result = load_spacecraft_scope_convergence(CONFIG)
    assert result.software_feature_freeze == ConvergenceDisposition.PASS
    assert result.stable_engineering_candidate == ConvergenceDisposition.PASS
    assert result.diagnostic_external_qualification == ConvergenceDisposition.BLOCKED
    assert result.next_phase == "ASTROGRAPH_BENCH_UI"
    assert result.core_feature_expansion_frozen is True
    assert result.core_functional_blockers == ()
    assert any(item.startswith("ESA_MISSION1:") for item in result.external_qualification_blockers)
    assert any(item.startswith("DXC_ADAPT:") for item in result.external_qualification_blockers)
    assert any(item.startswith("OPS_SAT:") for item in result.external_qualification_blockers)


def test_r32_required_spacecraft_algorithms_are_agent_independent_and_profile_is_explicit() -> None:
    profile = json.loads((ROOT / "configs/spacecraft/spacecraft_engineering_profile_r32.v1.json").read_text(encoding="utf-8"))
    assert profile["activation"] == "EXPLICIT_ENGINEERING_OR_BENCH_ONLY_NOT_AUTOMATIC_PRODUCTION_PROMOTION"
    assert profile["agent_policy"]["core_prediction_requires_agent"] is False
    assert profile["agent_policy"]["agent_tools_are_optional_review"] is True
    assert profile["agent_policy"]["agent_release_authority"] is False
    assert set(profile["algorithm_bindings"]) == {"ADCS_V5_2_2", "PHM_V6_0", "OPS_SAT", "ESA_MISSION1", "DXC_ADAPT"}


def test_r32_missing_core_capability_blocks_stable_candidate() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["required_capabilities"]["model_based_diagnosis"] = "MISSING"
    result = evaluate_spacecraft_scope_convergence(payload)
    assert result.software_feature_freeze == ConvergenceDisposition.BLOCKED
    assert result.stable_engineering_candidate == ConvergenceDisposition.BLOCKED
    assert result.next_phase == "SPACECRAFT_CORE_REMEDIATION"


def test_r32_evidence_snapshot_retains_r24_negative_and_positive_boundaries() -> None:
    snapshot = json.loads((ROOT / "governance/evidence/research/r32_spacecraft_convergence/R24_BENCHMARK_SNAPSHOT.json").read_text(encoding="utf-8"))
    assert snapshot["datasets"]["ADCS_V5_2_2"]["astrograph_current_frozen"]["end_to_end_15_class"]["macro_f1"] > 0.9
    assert snapshot["datasets"]["PHM_V6_0"]["astrograph_current_frozen"]["end_to_end_18_class"]["macro_f1"] > 0.95
    assert snapshot["datasets"]["ESA_MISSION1_LIGHTWEIGHT"]["astrograph_frozen"]["event_recall"] == 0.0
    assert snapshot["datasets"]["DXC09"]["frozen_external"]["competition_exact_full_set_accuracy"] == 0.0


def test_r32_release_claim_does_not_claim_external_or_flight_qualification() -> None:
    manifest = json.loads((ROOT / "configs/release_claims/spacecraft_engineering_stable_r32.json").read_text(encoding="utf-8"))
    claims = {item["claim_id"]: item for item in manifest["claims"]}
    assert claims["r32-cross-condition-generalization"]["disposition"] == "NOT_CLAIMED"
    assert claims["r32-cross-satellite-generalization"]["disposition"] == "NOT_CLAIMED"
    assert claims["r32-flight-validation"]["disposition"] == "NOT_CLAIMED"
