from __future__ import annotations

from pathlib import Path

from astrograph.agentization.capabilities import CapabilityEnvironment, build_registry


def registry(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    env = CapabilityEnvironment(
        project_root=root,
        allowed_roots=(root, tmp_path),
        artifact_dir=tmp_path / "artifacts",
        workflow_root=root / "configs/agentization/workflows",
        external_audit_roots=(root,),
    )
    return build_registry(env)


def test_model_independent_route_is_visible_but_bounded(tmp_path):
    result = registry(tmp_path).invoke("temporal.model_independent.describe", "DEVELOPMENT", {})
    assert result["status"] == "DEVELOPMENT_EVIDENCE"
    assert result["selection_groups"] == ["3", "8"]
    assert result["excluded_groups"] == ["4", "9"]
    assert result["runtime_ground_truth_access"] is False
    assert result["historical_p2_required"] is False
    assert result["new_confirmatory_evidence"] == "NONE"
    assert result["thesis_assessment"]["standalone_sufficient_for_excellent_thesis"] is False


def test_trace_index_verifier_rejects_truth_access(tmp_path):
    import hashlib, json

    path = tmp_path / "index.json"
    payload = {
        "development_groups": [3, 8],
        "excluded_groups": [4, 9],
        "truth_access_during_generation": True,
        "index_sha256": "0" * 64,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        registry(tmp_path).invoke(
            "temporal.model_independent.verify_trace_index",
            "DEVELOPMENT",
            {"path": str(path), "expected_sha256": digest},
        )
    except Exception as exc:
        assert getattr(exc, "code", None) == "LABEL_ACCESS_FORBIDDEN"
    else:
        raise AssertionError("truth-access trace index was accepted")


def test_mismatch_tolerant_route_is_visible_but_cross_reference_claim_is_frozen(tmp_path):
    result = registry(tmp_path).invoke(
        "temporal.mismatch_tolerant.describe", "DEVELOPMENT", {}
    )
    assert result["status"] == "DEVELOPMENT_EVIDENCE"
    assert result["selection_groups"] == ["3", "8"]
    assert result["excluded_groups"] == ["4", "9"]
    assert result["groups_4_9_read"] is False
    assert result["runtime_ground_truth_access"] is False
    assert result["arbitrary_cross_reference_robustness_established"] is False
    assert result["new_confirmatory_evidence"] == "NONE"
    assert result["selected_policy"]["selected_policy"]["alignment_fit_max"] == 0.08


def test_mismatch_trace_index_verifier_accepts_frozen_source_receipt(tmp_path):
    import hashlib

    root = Path(__file__).resolve().parents[2]
    path = root / "research/mismatch_tolerant_temporal_r001/trace_index_receipt.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = registry(tmp_path).invoke(
        "temporal.mismatch_tolerant.verify_trace_index",
        "DEVELOPMENT",
        {"path": str(path), "expected_sha256": digest},
    )
    assert result["status"] == "VERIFIED"
    assert result["truth_access_during_generation"] is False
    assert result["full_trajectory_alignment_used"] is False


def test_mismatch_trace_index_verifier_rejects_full_trajectory_alignment(tmp_path):
    import hashlib
    import json

    path = tmp_path / "mismatch-index.json"
    payload = {
        "development_groups": [3, 8],
        "excluded_groups": [4, 9],
        "truth_access_during_generation": False,
        "outcome_data_read_during_generation": False,
        "full_trajectory_alignment_used": True,
        "padded_edge_action": "INVALIDATE",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        registry(tmp_path).invoke(
            "temporal.mismatch_tolerant.verify_trace_index",
            "DEVELOPMENT",
            {"path": str(path), "expected_sha256": digest},
        )
    except Exception as exc:
        assert getattr(exc, "code", None) == "NONCAUSAL_ALIGNMENT_FORBIDDEN"
    else:
        raise AssertionError("full-trajectory alignment was accepted")
