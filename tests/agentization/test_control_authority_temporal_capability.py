from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from astrograph.agentization.capabilities import CapabilityEnvironment, build_registry
from astrograph.agentization.protocol import CapabilityProtocolError


def registry(root: Path):
    return build_registry(
        CapabilityEnvironment(root, (root,), root / "artifacts", root / "workflows", (root,))
    )


def test_control_authority_description_preserves_boundary(tmp_path: Path):
    result = registry(tmp_path).invoke(
        "temporal.control_authority.describe", "DEVELOPMENT", {}
    )
    assert result["groups_4_9_read"] is False
    assert result["runtime_ground_truth_access"] is False
    assert result["fault_identification_established"] is False
    assert result["new_confirmatory_evidence"] == "NONE"


def test_control_authority_trace_verifier_rejects_truth_access(tmp_path: Path):
    payload = {
        "development_groups": [3, 8],
        "excluded_groups": [4, 9],
        "calibration_groups": [0, 1, 2, 5, 6, 7],
        "groups_4_9_read": False,
        "runtime_ground_truth_access": False,
        "outcome_data_read_during_generation": False,
        "candidate_selections_frozen_before_truth_join": True,
        "candidate_count": 60,
        "index_sha256": "x",
    }
    path = tmp_path / "index.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert (
        registry(tmp_path)
        .invoke(
            "temporal.control_authority.verify_trace_index",
            "DEVELOPMENT",
            {"path": str(path), "expected_sha256": digest},
        )["status"]
        == "VERIFIED"
    )

    payload["runtime_ground_truth_access"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(CapabilityProtocolError):
        registry(tmp_path).invoke(
            "temporal.control_authority.verify_trace_index",
            "DEVELOPMENT",
            {"path": str(path), "expected_sha256": digest},
        )


def test_temporal_stack_descriptor_is_available(tmp_path: Path):
    config_dir = tmp_path / "configs" / "research"
    config_dir.mkdir(parents=True)
    payload = {
        "status": "DEVELOPMENT_TEMPORAL_STACK_FROZEN_FOR_SEALED_CAMPAIGN_DESIGN",
        "groups_4_9_read": False,
        "new_confirmatory_evidence": "NONE",
    }
    (config_dir / "adcs_temporal_stack_freeze_r001.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    result = registry(tmp_path).invoke(
        "temporal.stack.describe", "CONFIRMATORY_SEALED", {}
    )
    assert result["groups_4_9_read"] is False
    assert result["new_confirmatory_evidence"] == "NONE"
