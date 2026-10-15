from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from astrograph.governance import (
    AcceptanceCriterion,
    AppendOnlyJsonlLedger,
    ChangeRequest,
    CheckStatus,
    FrozenArtifact,
    GateCheck,
    GateEvidence,
    GateSeverity,
    HiddenTruthAccessStatus,
    ImprovementContract,
    ResourceBudget,
    TaskBoard,
    TaskStep,
    TaskStepStatus,
    decide_release,
    dump_json_yaml,
    reserve_gate_query,
    sha256_file,
    tree_sha256,
    validate_change_request_against_candidate,
    validate_change_request_scope,
    validate_contract_candidate,
    validate_task_board_evidence,
)
from astrograph.governance.models import ReleaseDecisionStatus

HASH = "a" * 64


def _build_candidate(tmp_path: Path) -> tuple[Path, Path, ImprovementContract, Path]:
    root = tmp_path / "candidate"
    (root / "governance").mkdir(parents=True)
    (root / "src/astrograph").mkdir(parents=True)
    (root / "tests").mkdir(parents=True)
    (root / "release.json").write_text('{"version":"1.1.0"}', encoding="utf-8")
    (root / "pyproject.toml").write_text('[project]\nname="x"\nversion="1.1.0"\n', encoding="utf-8")
    (root / "src/astrograph/__init__.py").write_text('__version__ = "1.1.0"\n', encoding="utf-8")
    (root / "tests/test_existing.py").write_text("def test_existing(): assert True\n", encoding="utf-8")
    baseline = {
        "release.json": sha256_file(root / "release.json"),
        "pyproject.toml": sha256_file(root / "pyproject.toml"),
        "src/astrograph/__init__.py": sha256_file(root / "src/astrograph/__init__.py"),
        "tests/test_existing.py": sha256_file(root / "tests/test_existing.py"),
    }
    manifest = root / "governance/baseline.json"
    manifest.write_text(json.dumps({"files": baseline}), encoding="utf-8")
    inventory = root / "governance/test-nodeids.json"
    inventory.write_text(json.dumps({"nodeids": ["tests/test_existing.py::test_existing"]}), encoding="utf-8")
    frozen = root / "governance/frozen.txt"
    frozen.write_text("fixed", encoding="utf-8")
    runner = root / "governance/runner.py"
    runner.write_text("runner = True\n", encoding="utf-8")
    evaluator = root / "governance/evaluator.py"
    evaluator.write_text("evaluator = True\n", encoding="utf-8")
    contract = ImprovementContract(
        contract_id="contract-test",
        baseline_version="1.0.0",
        target_version="1.1.0",
        baseline_archive_sha256=HASH,
        baseline_manifest_path="governance/baseline.json",
        baseline_manifest_sha256=sha256_file(manifest),
        baseline_test_count=1,
        baseline_test_inventory_path="governance/test-nodeids.json",
        baseline_test_inventory_sha256=sha256_file(inventory),
        baseline_tree_size_bytes=sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
        objective="test governance",
        in_scope=["governance/", "release.json", "pyproject.toml", "src/astrograph/__init__.py"],
        out_of_scope=["src/astrograph/diagnosis/", "tests/test_existing.py"],
        required_deliverables=["governance/frozen.txt"],
        prohibited_actions=["do not alter diagnosis"],
        protected_paths=["governance/frozen.txt"],
        frozen_artifacts=[
            FrozenArtifact(path="governance/frozen.txt", sha256=sha256_file(frozen), role="frozen"),
            FrozenArtifact(path="governance/runner.py", sha256=sha256_file(runner), role="formal-gate-entrypoint"),
            FrozenArtifact(path="governance/evaluator.py", sha256=sha256_file(evaluator), role="isolated-evaluator"),
        ],
        acceptance_criteria=[
            AcceptanceCriterion(
                criterion_id="FULL_TEST_SUITE",
                description="tests",
                severity=GateSeverity.HARD,
                expected_status=CheckStatus.PASS,
                evaluator="pytest",
                claim_boundary="tests only",
            )
        ],
        resource_budget=ResourceBudget(max_formal_gate_queries=1, max_storage_growth_bytes=10_000),
        stop_conditions=["stop on failure"],
        claim_boundary="governance only",
    )
    contract_path = root / "governance/contract.json"
    dump_json_yaml(contract_path, contract)
    return root, contract_path, contract, frozen


def _evidence(contract: ImprovementContract, *, status: CheckStatus = CheckStatus.PASS) -> GateEvidence:
    return GateEvidence(
        gate_run_id="gate-test",
        contract_id=contract.contract_id,
        contract_sha256=HASH,
        candidate_id="candidate-test",
        candidate_tree_sha256_before=HASH,
        candidate_tree_sha256_after=HASH,
        evaluator_id="evaluator",
        evaluator_sha256=next(item.sha256 for item in contract.frozen_artifacts if item.role == "formal-gate-entrypoint"),
        evaluator_module_sha256=next(item.sha256 for item in contract.frozen_artifacts if item.role == "isolated-evaluator"),
        gate_query_index=1,
        gate_query_receipt_path="gate_query_receipt.json",
        gate_query_receipt_sha256=HASH,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        checks=[GateCheck(criterion_id="FULL_TEST_SUITE", status=status, severity=GateSeverity.HARD)],
        tests_collected=1,
        tests_passed=1 if status == CheckStatus.PASS else 0,
        tests_failed=0 if status == CheckStatus.PASS else 1,
        baseline_test_nodeids_total=1,
        hidden_truth_access_status=HiddenTruthAccessStatus.NOT_APPLICABLE,
        source_tree_clean=True,
    )


def _decide(contract: ImprovementContract, evidence: GateEvidence, **kwargs):
    return decide_release(
        contract,
        evidence,
        expected_contract_sha256=HASH,
        expected_evaluator_sha256=next(item.sha256 for item in contract.frozen_artifacts if item.role == "formal-gate-entrypoint"),
        expected_evaluator_module_sha256=next(item.sha256 for item in contract.frozen_artifacts if item.role == "isolated-evaluator"),
        evidence_artifact_errors=kwargs.get("evidence_artifact_errors", []),
        query_receipt_errors=kwargs.get("query_receipt_errors", []),
    )


def test_contract_validation_requires_external_contract_pin(tmp_path: Path) -> None:
    root, contract_path, contract, _ = _build_candidate(tmp_path)
    report = validate_contract_candidate(contract, contract_path=contract_path, candidate_root=root)
    assert report.status == CheckStatus.FAIL
    assert "EXTERNAL_CONTRACT_PIN_REQUIRED" in report.blocking_reasons


def test_contract_candidate_accepts_in_scope_governance_changes(tmp_path: Path) -> None:
    root, contract_path, contract, _ = _build_candidate(tmp_path)
    (root / "governance/new.txt").write_text("new", encoding="utf-8")
    report = validate_contract_candidate(
        contract, contract_path=contract_path, candidate_root=root,
        expected_contract_sha256=sha256_file(contract_path),
    )
    assert report.status == CheckStatus.PASS
    assert "governance/new.txt" in report.changed_paths


def test_contract_candidate_blocks_out_of_scope_diagnosis_change(tmp_path: Path) -> None:
    root, contract_path, contract, _ = _build_candidate(tmp_path)
    diagnosis = root / "src/astrograph/diagnosis/rules.py"
    diagnosis.parent.mkdir(parents=True)
    diagnosis.write_text("changed = True\n", encoding="utf-8")
    report = validate_contract_candidate(
        contract, contract_path=contract_path, candidate_root=root,
        expected_contract_sha256=sha256_file(contract_path),
    )
    assert report.status == CheckStatus.FAIL
    assert "OUT_OF_SCOPE_CHANGE_DETECTED" in report.blocking_reasons


def test_contract_candidate_blocks_contract_and_frozen_artifact_tampering(tmp_path: Path) -> None:
    root, contract_path, contract, frozen = _build_candidate(tmp_path)
    frozen.write_text("tampered", encoding="utf-8")
    report = validate_contract_candidate(
        contract, contract_path=contract_path, candidate_root=root,
        expected_contract_sha256="d" * 64,
    )
    assert report.status == CheckStatus.FAIL
    assert "CONTRACT_HASH_MISMATCH" in report.blocking_reasons
    assert "FROZEN_ARTIFACT_INTEGRITY_FAILURE" in report.blocking_reasons


def test_change_request_must_exactly_cover_actual_diff(tmp_path: Path) -> None:
    root, contract_path, contract, _ = _build_candidate(tmp_path)
    (root / "governance/new.txt").write_text("new", encoding="utf-8")
    report = validate_contract_candidate(
        contract, contract_path=contract_path, candidate_root=root,
        expected_contract_sha256=sha256_file(contract_path),
    )
    request = ChangeRequest(
        request_id="change", contract_id=contract.contract_id,
        contract_sha256=sha256_file(contract_path), candidate_id="candidate",
        summary="test", rationale="test exact diff", changed_paths=["governance/contract.json"],
        rollback_plan="restore baseline",
    )
    errors = validate_change_request_against_candidate(
        contract, request, report, expected_contract_sha256=sha256_file(contract_path),
    )
    assert "UNDECLARED_ACTUAL_CHANGE:governance/new.txt" in errors


def test_protected_path_may_be_declared_only_when_frozen(tmp_path: Path) -> None:
    _, contract_path, contract, _ = _build_candidate(tmp_path)
    request = ChangeRequest(
        request_id="change", contract_id=contract.contract_id,
        contract_sha256=sha256_file(contract_path), candidate_id="candidate",
        summary="test", rationale="protected frozen path", changed_paths=["governance/frozen.txt"],
        rollback_plan="restore baseline",
    )
    assert validate_change_request_scope(contract, request) == []


def test_task_board_completed_evidence_must_exist(tmp_path: Path) -> None:
    root, contract_path, contract, _ = _build_candidate(tmp_path)
    board = TaskBoard(
        board_id="board", contract_id=contract.contract_id,
        contract_sha256=sha256_file(contract_path), goal="test",
        planned_steps=[TaskStep(step_id="x", description="done", status=TaskStepStatus.COMPLETED, evidence=["missing.txt"])],
        deliverability="READY",
    )
    errors = validate_task_board_evidence(
        root, board, contract_id=contract.contract_id, contract_sha256=sha256_file(contract_path),
    )
    assert "TASK_EVIDENCE_MISSING:x:missing.txt" in errors


def test_gate_query_budget_is_persistently_enforced(tmp_path: Path) -> None:
    _, _, contract, _ = _build_candidate(tmp_path)
    receipt, path = reserve_gate_query(contract, contract_sha256=HASH, candidate_id="c1", state_dir=tmp_path / "state")
    assert receipt.query_index == 1 and path.is_file()
    with pytest.raises(RuntimeError, match="FORMAL_GATE_QUERY_BUDGET_EXCEEDED"):
        reserve_gate_query(contract, contract_sha256=HASH, candidate_id="c1", state_dir=tmp_path / "state")


def test_release_decision_accepts_only_complete_verified_evidence(tmp_path: Path) -> None:
    _, _, contract, _ = _build_candidate(tmp_path)
    decision = _decide(contract, _evidence(contract))
    assert decision.status == ReleaseDecisionStatus.ACCEPT


def test_forged_evaluator_hash_is_blocked(tmp_path: Path) -> None:
    _, _, contract, _ = _build_candidate(tmp_path)
    evidence = _evidence(contract)
    evidence.evaluator_sha256 = "d" * 64
    decision = _decide(contract, evidence)
    assert decision.status == ReleaseDecisionStatus.BLOCKED
    assert "EVALUATOR_ENTRYPOINT_HASH_MISMATCH" in decision.blockers


def test_unverified_hidden_truth_access_escalates(tmp_path: Path) -> None:
    _, _, contract, _ = _build_candidate(tmp_path)
    evidence = _evidence(contract)
    evidence.hidden_truth_access_status = HiddenTruthAccessStatus.UNVERIFIED
    decision = _decide(contract, evidence)
    assert decision.status == ReleaseDecisionStatus.ESCALATE
    assert "HIDDEN_TRUTH_ACCESS_UNVERIFIED" in decision.blockers


def test_hidden_truth_violation_blocks(tmp_path: Path) -> None:
    _, _, contract, _ = _build_candidate(tmp_path)
    evidence = _evidence(contract)
    evidence.hidden_truth_access_status = HiddenTruthAccessStatus.VIOLATION
    evidence.hidden_truth_accessed = True
    decision = _decide(contract, evidence)
    assert decision.status == ReleaseDecisionStatus.BLOCKED


def test_evidence_artifact_failure_blocks(tmp_path: Path) -> None:
    _, _, contract, _ = _build_candidate(tmp_path)
    decision = _decide(contract, _evidence(contract), evidence_artifact_errors=["EVIDENCE_ARTIFACT_MISSING:x"])
    assert decision.status == ReleaseDecisionStatus.BLOCKED


def test_baseline_test_identity_regression_is_rejected(tmp_path: Path) -> None:
    _, _, contract, _ = _build_candidate(tmp_path)
    evidence = _evidence(contract)
    evidence.baseline_test_nodeids_missing = ["tests/test_existing.py::test_existing"]
    decision = _decide(contract, evidence)
    assert decision.status == ReleaseDecisionStatus.REJECT
    assert "BASELINE_TEST_IDENTITY_REGRESSION" in decision.blockers


@pytest.mark.parametrize("status", [CheckStatus.NOT_RUN, CheckStatus.TIMEOUT, CheckStatus.ERROR])
def test_incomplete_hard_gate_can_never_be_accepted(tmp_path: Path, status: CheckStatus) -> None:
    _, _, contract, _ = _build_candidate(tmp_path)
    evidence = _evidence(contract, status=status)
    if status == CheckStatus.TIMEOUT:
        evidence.tests_timed_out = True
    decision = _decide(contract, evidence)
    assert decision.status == ReleaseDecisionStatus.ESCALATE


def test_gate_severity_mismatch_blocks(tmp_path: Path) -> None:
    _, _, contract, _ = _build_candidate(tmp_path)
    evidence = _evidence(contract)
    evidence.checks[0].severity = GateSeverity.SOFT
    decision = _decide(contract, evidence)
    assert decision.status == ReleaseDecisionStatus.BLOCKED


def test_completed_task_step_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        TaskStep(step_id="x", description="done", status=TaskStepStatus.COMPLETED)


def test_gate_evidence_rejects_inconsistent_test_arithmetic() -> None:
    with pytest.raises(ValidationError):
        GateEvidence(
            gate_run_id="g", contract_id="c", contract_sha256=HASH, candidate_id="x",
            candidate_tree_sha256_before=HASH, candidate_tree_sha256_after=HASH,
            evaluator_id="e", evaluator_sha256="b" * 64, evaluator_module_sha256="c" * 64,
            gate_query_index=1, gate_query_receipt_path="r", gate_query_receipt_sha256=HASH,
            started_at="2026-01-01T00:00:00+00:00", finished_at="2026-01-01T00:00:01+00:00",
            checks=[GateCheck(criterion_id="x", status=CheckStatus.PASS)],
            tests_collected=2, tests_passed=1, source_tree_clean=True,
            hidden_truth_access_status=HiddenTruthAccessStatus.NOT_APPLICABLE,
        )


def test_append_only_ledger_detects_payload_tampering(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = AppendOnlyJsonlLedger(ledger_path)
    ledger.append("failure", {"error_code": "X"})
    ledger.append("failure", {"error_code": "Y"})
    assert ledger.audit().status == "PASS"
    text = ledger_path.read_text(encoding="utf-8").replace('"error_code":"X"', '"error_code":"Z"')
    ledger_path.write_text(text, encoding="utf-8")
    result = ledger.audit()
    assert result.status == "FAIL"
    assert any(error.startswith("RECORD_HASH_MISMATCH") for error in result.errors)


def test_tree_hash_ignores_python_and_pytest_caches(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.txt").write_text("a", encoding="utf-8")
    before = tree_sha256(root)
    (root / "__pycache__").mkdir()
    (root / "__pycache__/x.pyc").write_bytes(b"cache")
    (root / ".pytest_cache").mkdir()
    (root / ".pytest_cache/state").write_text("cache", encoding="utf-8")
    assert tree_sha256(root) == before
