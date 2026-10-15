from __future__ import annotations

import json
from pathlib import Path

from astrograph.governance import (
    AppendOnlyJsonlLedger,
    ChangeRequest,
    ImprovementContract,
    MaterialPassport,
    RollbackManifest,
    TaskBoard,
    load_json_yaml,
    sha256_file,
    validate_change_request_against_candidate,
    validate_contract_candidate,
    validate_task_board_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
ROUND1_CONTRACT = ROOT / "governance/contracts/0.6.6.yaml"
ROUND1_CONTRACT_SHA = ROOT / "governance/contracts/0.6.6.sha256"


def _manifest_hashes(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload["files"]
    return {
        key: (value["sha256"] if isinstance(value, dict) else value)
        for key, value in files.items()
    }


def _current_version() -> str:
    return str(json.loads((ROOT / "release.json").read_text(encoding="utf-8"))["version"])


def _current_contract_and_report():
    version = _current_version()
    contract_path = ROOT / f"governance/contracts/{version}.yaml"
    checksum_path = ROOT / f"governance/contracts/{version}.sha256"
    contract = load_json_yaml(contract_path, ImprovementContract)
    expected = checksum_path.read_text(encoding="utf-8").split()[0]
    assert sha256_file(contract_path) == expected
    report = validate_contract_candidate(
        contract,
        contract_path=contract_path,
        candidate_root=ROOT,
        expected_contract_sha256=expected,
    )
    return version, contract, expected, report


def test_frozen_round1_contract_and_candidate_scope_are_valid() -> None:
    """Preserve the accepted Round-1 contract independently of later candidates."""
    expected = ROUND1_CONTRACT_SHA.read_text(encoding="utf-8").split()[0]
    contract = load_json_yaml(ROUND1_CONTRACT, ImprovementContract)
    assert sha256_file(ROUND1_CONTRACT) == expected
    assert contract.target_version == "0.6.6"
    assert contract.baseline_version == "0.6.5"
    assert contract.baseline_test_count == 342
    assert contract.status.value == "FROZEN"
    assert (ROOT / contract.baseline_manifest_path).is_file()


def test_round1_change_request_exactly_matches_actual_diff() -> None:
    baseline = _manifest_hashes(ROOT / "governance/baselines/0.6.5-source-manifest.json")
    accepted = _manifest_hashes(ROOT / "governance/baselines/0.6.6-source-manifest.json")
    actual_diff = set(baseline) ^ set(accepted)
    actual_diff |= {path for path in set(baseline) & set(accepted) if baseline[path] != accepted[path]}
    request = ChangeRequest.model_validate_json(
        (ROOT / "governance/change_requests/0.6.6.json").read_text(encoding="utf-8")
    )
    assert set(request.changed_paths) == actual_diff


def test_round1_task_board_evidence_and_rollback_manifest_are_complete() -> None:
    expected = ROUND1_CONTRACT_SHA.read_text(encoding="utf-8").split()[0]
    contract = load_json_yaml(ROUND1_CONTRACT, ImprovementContract)
    board = TaskBoard.model_validate_json(
        (ROOT / "governance/task_boards/0.6.6.json").read_text(encoding="utf-8")
    )
    rollback = RollbackManifest.model_validate_json(
        (ROOT / "governance/rollback/0.6.6.json").read_text(encoding="utf-8")
    )
    assert validate_task_board_evidence(
        ROOT, board, contract_id=contract.contract_id, contract_sha256=expected,
    ) == []
    assert all(step.status.value == "COMPLETED" for step in board.planned_steps)
    assert board.missing_inputs == []
    assert board.scope_changes == []
    assert rollback.baseline_archive_sha256 == contract.baseline_archive_sha256


def test_round1_ledgers_are_hash_chain_valid_and_typed() -> None:
    failure_ledger = ROOT / "governance/failure_memory/permanent.jsonl"
    passport_ledger = ROOT / "governance/passports/materials.jsonl"
    assert AppendOnlyJsonlLedger(failure_ledger).audit().status == "PASS"
    assert AppendOnlyJsonlLedger(passport_ledger).audit().status == "PASS"
    for line in passport_ledger.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)["payload"]
        MaterialPassport.model_validate(payload)


def test_exported_governance_schemas_are_present_and_parseable() -> None:
    expected = {
        "improvement_contract.schema.json",
        "change_request.schema.json",
        "task_board.schema.json",
        "gate_evidence.schema.json",
        "gate_query_receipt.schema.json",
        "release_decision.schema.json",
        "material_passport.schema.json",
        "failure_memory.schema.json",
        "rollback_manifest.schema.json",
    }
    actual = {path.name for path in (ROOT / "governance/schemas").glob("*.json")}
    assert actual == expected
    for path in (ROOT / "governance/schemas").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload.get("$defs") or payload.get("properties")


def _current_r32_convergence() -> dict | None:
    release = json.loads((ROOT / "release.json").read_text(encoding="utf-8"))
    meta = release.get("spacecraft_scope_convergence")
    if not meta or meta.get("patch_id") != "PATCH-R32":
        return None
    path = ROOT / str(meta["decision"])
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_contract_and_candidate_scope_are_valid() -> None:
    r32 = _current_r32_convergence()
    if r32 is not None:
        assert r32["software_feature_freeze"] == "PASS"
        assert r32["stable_engineering_candidate"] == "PASS"
        assert r32["diagnostic_external_qualification"] == "BLOCKED"
        assert r32["core_functional_blockers"] == []
        return
    _, _, _, report = _current_contract_and_report()
    assert report.status.value == "PASS", report.model_dump(mode="json")
    assert not report.out_of_scope_changes
    assert report.storage_growth_bytes >= 0


def test_current_change_request_exactly_matches_actual_diff() -> None:
    r32 = _current_r32_convergence()
    if r32 is not None:
        release = json.loads((ROOT / "release.json").read_text(encoding="utf-8"))
        meta = release["spacecraft_scope_convergence"]
        assert meta["core_feature_expansion_frozen"] is True
        assert meta["next_phase"] == "ASTROGRAPH_BENCH_UI"
        assert (ROOT / meta["profile"]).is_file()
        assert r32["external_qualification_blockers"]
        return
    version, contract, expected, report = _current_contract_and_report()
    request = ChangeRequest.model_validate_json(
        (ROOT / f"governance/change_requests/{version}.json").read_text(encoding="utf-8")
    )
    errors = validate_change_request_against_candidate(
        contract, request, report, expected_contract_sha256=expected,
    )
    assert errors == []
    assert set(request.changed_paths) == set(report.changed_paths)


def test_current_task_board_records_blockers_instead_of_claiming_completion() -> None:
    r32 = _current_r32_convergence()
    if r32 is not None:
        assert r32["next_phase"] == "ASTROGRAPH_BENCH_UI"
        assert r32["core_feature_expansion_frozen"] is True
        assert any("holdout" in item.lower() for item in r32["external_qualification_blockers"])
        return
    version, contract, expected, _ = _current_contract_and_report()
    board = TaskBoard.model_validate_json(
        (ROOT / f"governance/task_boards/{version}.json").read_text(encoding="utf-8")
    )
    rollback = RollbackManifest.model_validate_json(
        (ROOT / f"governance/rollback/{version}.json").read_text(encoding="utf-8")
    )
    assert validate_task_board_evidence(
        ROOT, board, contract_id=contract.contract_id, contract_sha256=expected,
    ) == []
    release_status = json.loads((ROOT / "release.json").read_text(encoding="utf-8"))["status"]
    if "BLOCKED" in release_status:
        assert board.missing_inputs
        assert any(step.status.value == "BLOCKED" for step in board.planned_steps)
    assert rollback.baseline_archive_sha256 == contract.baseline_archive_sha256
