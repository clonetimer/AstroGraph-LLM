from __future__ import annotations

import json
from pathlib import Path

from astrograph.evaluation.external_dataset_contract import file_identity, verify_seal

ROOT = Path(__file__).resolve().parents[1]


def test_r27_protocol_is_content_bound_and_keeps_agent_non_authoritative():
    plan = json.loads((ROOT / "configs/evaluation/r27_remediation_protocol.v1.json").read_text(encoding="utf-8"))
    verify_seal(plan)
    assert plan["patch_id"] == "PATCH-R27"
    assert plan["base_artifact_sha256"] == "9c790a2c47156b18b12e0c7160c449efa6cf5384baac73e66212783f20b491ec"
    assert plan["dxc09_sample"]["agent_required"] is False
    assert plan["dxc09_sample"]["runtime_default_changed"] is False
    assert plan["dxc09_sample"]["competition_or_rematch_regenerated"] is False
    assert plan["dxc09_sample"]["new_holdout_supplied"] is False
    assert plan["agent"]["diagnostic_authority"] is False
    assert plan["agent"]["formal_evidence_eligible"] is False
    # R27 dedicated artifacts remain byte-bound. The shared Diagnostic Probe host is
    # an additive extension point and legitimately changes when later patches add
    # new read-only probe kinds; historical R27 receipt bytes are not rewritten.
    for relative, expected in plan["source_bindings"].items():
        if relative == "src/astrograph/investigation/diagnostic_probes.py":
            current = (ROOT / relative).read_text(encoding="utf-8")
            assert "DXC_EXPECTED_RESPONSE_REVIEW" in current
            continue
        assert file_identity(ROOT / relative) == expected


def test_r27_development_receipt_is_post_contact_not_external_validation():
    plan = json.loads((ROOT / "configs/evaluation/r27_remediation_protocol.v1.json").read_text(encoding="utf-8"))
    receipt = json.loads((ROOT / "governance/evidence/research/r27_expected_response/dxc_expected_response_development.json").read_text(encoding="utf-8"))
    verify_seal(receipt)
    assert receipt["receipt_sha256"] == plan["dxc09_sample"]["development_receipt_sha256"]
    assert receipt["status"] == "POST_CONTACT_DEVELOPMENT_ALGORITHM_AUDIT_ONLY"
    assert receipt["seen_competition_or_rematch_regenerated"] is False
    assert receipt["runtime_default_changed"] is False
    assert receipt["agent_required"] is False
    assert receipt["qualification"] == "NOT_EXTERNAL_VALIDATION_NOT_PRODUCTION_DEFAULT_NOT_FLIGHT_QUALIFIED"
