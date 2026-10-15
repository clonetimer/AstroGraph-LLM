from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from astrograph.agentization.capabilities import CapabilityEnvironment, build_registry
from astrograph.agentization.protocol import CapabilityProtocolError


def registry(root: Path):
    environment = CapabilityEnvironment(root, (root,), root / "artifacts", root / "workflows", (root,))
    return build_registry(environment)


def test_so3_description_preserves_research_boundary(tmp_path: Path):
    result = registry(tmp_path).invoke("temporal.so3_star_gyro.describe", "DEVELOPMENT", {})
    assert result["status"] == "DEVELOPMENT_EVIDENCE"
    assert result["groups_4_9_read"] is False
    assert result["runtime_ground_truth_access"] is False
    assert result["fault_identification_established"] is False
    assert result["new_confirmatory_evidence"] == "NONE"


def test_trace_index_verifier_enforces_label_and_group_isolation(tmp_path: Path):
    payload = {
        "development_groups": [3, 8],
        "excluded_groups": [4, 9],
        "calibration_groups": [0, 1, 2, 5, 6, 7],
        "groups_4_9_read": False,
        "runtime_ground_truth_access": False,
        "outcome_data_read_during_generation": False,
        "candidate_selections_frozen_before_truth_join": True,
        "candidate_count": 224,
        "index_sha256": "x",
    }
    path = tmp_path / "index.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = registry(tmp_path).invoke(
        "temporal.so3_star_gyro.verify_trace_index",
        "DEVELOPMENT",
        {"path": str(path), "expected_sha256": digest},
    )
    assert result["status"] == "VERIFIED"

    payload["groups_4_9_read"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(CapabilityProtocolError):
        registry(tmp_path).invoke(
            "temporal.so3_star_gyro.verify_trace_index",
            "DEVELOPMENT",
            {"path": str(path), "expected_sha256": digest},
        )
