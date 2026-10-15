from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from astrograph.agentization.capabilities import CapabilityEnvironment, build_registry
from astrograph.agentization.protocol import CapabilityProtocolError


def registry(root: Path):
    env = CapabilityEnvironment(
        project_root=root,
        allowed_roots=(root,),
        artifact_dir=root / "artifacts",
        workflow_root=root / "workflows",
        external_audit_roots=(root,),
    )
    return build_registry(env)


def test_reference_set_description_is_bounded(tmp_path: Path) -> None:
    result = registry(tmp_path).invoke("temporal.reference_set.describe", "DEVELOPMENT", {})
    assert result["status"] == "DEVELOPMENT_EVIDENCE"
    assert result["groups_4_9_read"] is False
    assert result["probabilistic_or_conformal_guarantee"] is False
    assert result["new_confirmatory_evidence"] == "NONE"


def test_reference_set_trace_index_verifier_requires_provenance_exclusion(tmp_path: Path) -> None:
    payload = {
        "development_groups": [3, 8],
        "excluded_groups": [4, 9],
        "groups_4_9_read": False,
        "truth_access_during_generation": False,
        "outcome_data_read_during_generation": False,
        "full_trajectory_alignment_used": False,
        "target_provenance_group_exclusion_enforced": True,
        "prefix_template_identity_verified": True,
        "index_sha256": "x",
    }
    path = tmp_path / "index.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    result = registry(tmp_path).invoke(
        "temporal.reference_set.verify_trace_index",
        "DEVELOPMENT",
        {"path": str(path), "expected_sha256": sha},
    )
    assert result["status"] == "VERIFIED"
    payload["target_provenance_group_exclusion_enforced"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(CapabilityProtocolError, match="target provenance group"):
        registry(tmp_path).invoke(
            "temporal.reference_set.verify_trace_index",
            "DEVELOPMENT",
            {"path": str(path), "expected_sha256": sha},
        )


def test_reference_set_verifier_is_not_available_to_diagnosis_profile(tmp_path: Path) -> None:
    with pytest.raises(CapabilityProtocolError):
        registry(tmp_path).invoke(
            "temporal.reference_set.verify_trace_index",
            "DIAGNOSIS",
            {"path": "x", "expected_sha256": "0" * 64},
        )
