from __future__ import annotations

import json
from pathlib import Path

from astrograph.evaluation.esa_r26_frozen import verify_r26_frozen_esa_detector_artifact
from astrograph.evaluation.external_dataset_contract import file_identity, verify_seal

ROOT = Path(__file__).resolve().parents[1]


def test_r26_protocol_is_content_bound_to_final_sources_and_receipts():
    plan = json.loads((ROOT / "configs/evaluation/r26_remediation_protocol.v1.json").read_text(encoding="utf-8"))
    verify_seal(plan)
    assert plan["patch_id"] == "PATCH-R26"
    assert plan["base_artifact_sha256"] == "8c948f2aff971542ccd63eb2bb912a27c42f41f06e60a010b80de48ff0645612"
    assert plan["dxc09_sample"]["selected_policy"] == "DISCRETE_FIRST_SINGLE_CONTINUOUS"
    for relative, expected in plan["source_bindings"].items():
        assert file_identity(ROOT / relative) == expected

    dxc = json.loads((ROOT / "governance/evidence/research/r26_remediation/dxc_discrete_consistency_development.json").read_text(encoding="utf-8"))
    verify_seal(dxc)
    assert dxc["receipt_sha256"] == plan["dxc09_sample"]["development_receipt_sha256"]
    assert dxc["seen_competition_or_rematch_regenerated"] is False

    esa = json.loads((ROOT / "governance/evidence/research/r26_remediation/esa_frozen_detector.json").read_text(encoding="utf-8"))
    verify_r26_frozen_esa_detector_artifact(esa)
    assert esa["receipt_sha256"] == plan["esa"]["detector_receipt_sha256"]
    assert plan["esa"]["new_holdout_supplied"] is False


def test_r26_esa_synthetic_smoke_is_not_external_validation():
    receipt = json.loads((ROOT / "governance/evidence/research/r26_remediation/esa_frozen_synthetic_prediction_receipt.json").read_text(encoding="utf-8"))
    verify_seal(receipt)
    assert receipt["truth_joined"] is False
    assert receipt["formal_evidence_generated"] is False
    assert receipt["status"] == "PREDICTION_FROZEN_PRE_TRUTH"
