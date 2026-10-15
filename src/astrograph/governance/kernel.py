from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import os
import subprocess
import tempfile
import time
import tomllib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, TypeVar

from pydantic import BaseModel

from astrograph.audit.store import sha256_json
from astrograph.governance.models import (
    ChangeRequest,
    CheckStatus,
    ContractValidationReport,
    GateEvidence,
    GateQueryReceipt,
    GateSeverity,
    HiddenTruthAccessStatus,
    ImprovementContract,
    ReleaseDecision,
    ReleaseDecisionStatus,
    TaskBoard,
)

T = TypeVar("T", bound=BaseModel)

_IGNORED_TREE_NAMES = {
    "__pycache__", ".pytest_cache", ".git", ".claude", "build", "dist", "htmlcov", "worktrees",
}
_IGNORED_TREE_SUFFIXES = (".pyc", ".pyo")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_yaml(path: str | Path, model: type[T]) -> T:
    """Load the governance JSON subset of YAML without a runtime YAML dependency."""
    return model.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def dump_json_yaml(path: str | Path, value: BaseModel | dict[str, Any]) -> None:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _iter_tree_files(root: Path) -> Iterable[Path]:
    candidates: Iterable[Path]
    if (root / ".git").exists():
        visible = subprocess.run(
            [
                "git", "-C", str(root), "ls-files", "--cached", "--others",
                "--exclude-standard", "-z",
            ],
            capture_output=True,
            check=False,
        )
        candidates = (
            [root / item.decode("utf-8", errors="strict") for item in visible.stdout.split(b"\0") if item]
            if visible.returncode == 0 else root.rglob("*")
        )
    else:
        candidates = root.rglob("*")
    for path in sorted(candidates):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in _IGNORED_TREE_NAMES for part in relative.parts):
            continue
        if path.suffix in _IGNORED_TREE_SUFFIXES:
            continue
        yield path


def build_tree_manifest(root: str | Path) -> dict[str, str]:
    root = Path(root).resolve()
    return {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in _iter_tree_files(root)
    }


def tree_size_bytes(root: str | Path) -> int:
    return sum(path.stat().st_size for path in _iter_tree_files(Path(root).resolve()))


def tree_sha256(root: str | Path) -> str:
    return sha256_json(build_tree_manifest(root))


def load_manifest(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    files = payload.get("files", payload)
    if not isinstance(files, dict):
        raise ValueError("baseline manifest must contain a files object")
    result = {str(key): str(value) for key, value in files.items()}
    for relative, digest in result.items():
        if len(digest) != 64:
            raise ValueError(f"invalid sha256 for {relative}")
    return result


def diff_manifest(baseline: dict[str, str], current: dict[str, str]) -> list[str]:
    return sorted(
        relative for relative in set(baseline) | set(current)
        if baseline.get(relative) != current.get(relative)
    )


def _matches_scope(path: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        normalized = pattern.replace("\\", "/")
        if normalized.endswith("/") and path.startswith(normalized):
            return True
        if fnmatch.fnmatchcase(path, normalized) or path == normalized:
            return True
    return False


def _read_candidate_version(root: Path) -> tuple[str, list[str]]:
    versions: list[str] = []
    errors: list[str] = []
    try:
        versions.append(str(json.loads((root / "release.json").read_text(encoding="utf-8"))["version"]))
    except Exception:
        errors.append("RELEASE_VERSION_UNREADABLE")
    try:
        versions.append(str(tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]))
    except Exception:
        errors.append("PYPROJECT_VERSION_UNREADABLE")
    try:
        runtime_init = root / "src/astrograph/__init__.py"
        tree = ast.parse(runtime_init.read_text(encoding="utf-8"), filename=str(runtime_init))
        runtime_version = ""
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if (
                any(isinstance(target, ast.Name) and target.id == "__version__" for target in targets)
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                runtime_version = value.value
                break
        if not runtime_version:
            raise ValueError("literal __version__ assignment not found")
        versions.append(runtime_version)
    except Exception:
        errors.append("RUNTIME_VERSION_UNREADABLE")
    distinct = sorted(set(versions))
    if len(distinct) > 1:
        errors.append("VERSION_METADATA_MISMATCH:" + ",".join(distinct))
    return (distinct[0] if len(distinct) == 1 else ""), errors


def frozen_artifact_hash(contract: ImprovementContract, *, role: str = "", path: str = "") -> str:
    matches = [
        item for item in contract.frozen_artifacts
        if (role and item.role == role) or (path and item.path == path)
    ]
    if len(matches) != 1:
        return ""
    return matches[0].sha256


def validate_contract_candidate(
    contract: ImprovementContract,
    *,
    contract_path: str | Path,
    candidate_root: str | Path,
    expected_contract_sha256: str = "",
) -> ContractValidationReport:
    root = Path(candidate_root).resolve()
    contract_path = Path(contract_path).resolve()
    contract_sha = sha256_file(contract_path)
    blockers: list[str] = []
    if not expected_contract_sha256:
        blockers.append("EXTERNAL_CONTRACT_PIN_REQUIRED")
    elif contract_sha != expected_contract_sha256:
        blockers.append("CONTRACT_HASH_MISMATCH")
    if contract.status.value != "FROZEN":
        blockers.append("CONTRACT_NOT_FROZEN")

    baseline_manifest_path = root / contract.baseline_manifest_path
    if not baseline_manifest_path.is_file():
        blockers.append("BASELINE_MANIFEST_MISSING")
        baseline: dict[str, str] = {}
    elif sha256_file(baseline_manifest_path) != contract.baseline_manifest_sha256:
        blockers.append("BASELINE_MANIFEST_HASH_MISMATCH")
        baseline = {}
    else:
        baseline = load_manifest(baseline_manifest_path)

    current = build_tree_manifest(root)
    changed = diff_manifest(baseline, current) if baseline else []
    out_of_scope = [
        path for path in changed
        if not _matches_scope(path, contract.in_scope) or _matches_scope(path, contract.out_of_scope)
    ]
    if out_of_scope:
        blockers.append("OUT_OF_SCOPE_CHANGE_DETECTED")

    missing_deliverables = [path for path in contract.required_deliverables if not (root / path).exists()]
    if missing_deliverables:
        blockers.append("REQUIRED_DELIVERABLE_MISSING")

    frozen_errors: list[str] = []
    for artifact in contract.frozen_artifacts:
        path = root / artifact.path
        if not path.is_file():
            frozen_errors.append(f"MISSING:{artifact.path}")
        elif sha256_file(path) != artifact.sha256:
            frozen_errors.append(f"HASH_MISMATCH:{artifact.path}")
    if frozen_errors:
        blockers.append("FROZEN_ARTIFACT_INTEGRITY_FAILURE")

    if contract.baseline_test_inventory_path:
        inventory = root / contract.baseline_test_inventory_path
        if not inventory.is_file():
            blockers.append("BASELINE_TEST_INVENTORY_MISSING")
        elif sha256_file(inventory) != contract.baseline_test_inventory_sha256:
            blockers.append("BASELINE_TEST_INVENTORY_HASH_MISMATCH")

    actual_size = tree_size_bytes(root)
    growth = actual_size - contract.baseline_tree_size_bytes if contract.baseline_tree_size_bytes else 0
    if contract.baseline_tree_size_bytes and growth > contract.resource_budget.max_storage_growth_bytes:
        blockers.append("STORAGE_GROWTH_BUDGET_EXCEEDED")

    version, version_errors = _read_candidate_version(root)
    blockers.extend(version_errors)
    if version and version != contract.target_version:
        blockers.append(f"TARGET_VERSION_MISMATCH:{version}!={contract.target_version}")

    blockers = list(dict.fromkeys(blockers))
    return ContractValidationReport(
        status=CheckStatus.PASS if not blockers else CheckStatus.FAIL,
        contract_id=contract.contract_id,
        contract_sha256=contract_sha,
        candidate_root=str(root),
        candidate_version=version,
        changed_paths=changed,
        out_of_scope_changes=out_of_scope,
        missing_deliverables=missing_deliverables,
        frozen_artifact_errors=frozen_errors,
        actual_tree_size_bytes=actual_size,
        storage_growth_bytes=growth,
        blocking_reasons=blockers,
    )


def validate_change_request_scope(contract: ImprovementContract, request: ChangeRequest) -> list[str]:
    blockers: list[str] = []
    if request.contract_id != contract.contract_id:
        blockers.append("CHANGE_REQUEST_CONTRACT_ID_MISMATCH")
    for path in request.changed_paths:
        if not _matches_scope(path, contract.in_scope):
            blockers.append(f"CHANGE_PATH_NOT_IN_SCOPE:{path}")
        if _matches_scope(path, contract.out_of_scope):
            blockers.append(f"CHANGE_PATH_EXPLICITLY_OUT_OF_SCOPE:{path}")
        if _matches_scope(path, contract.protected_paths):
            artifact = next((item for item in contract.frozen_artifacts if item.path == path), None)
            if artifact is None:
                blockers.append(f"PROTECTED_PATH_NOT_FROZEN:{path}")
    return list(dict.fromkeys(blockers))


def validate_change_request_against_candidate(
    contract: ImprovementContract,
    request: ChangeRequest,
    report: ContractValidationReport,
    *,
    expected_contract_sha256: str,
) -> list[str]:
    blockers = validate_change_request_scope(contract, request)
    if request.contract_sha256 != expected_contract_sha256:
        blockers.append("CHANGE_REQUEST_CONTRACT_HASH_MISMATCH")
    actual = set(report.changed_paths)
    declared = set(request.changed_paths)
    for path in sorted(actual - declared):
        blockers.append(f"UNDECLARED_ACTUAL_CHANGE:{path}")
    for path in sorted(declared - actual):
        blockers.append(f"DECLARED_CHANGE_NOT_PRESENT:{path}")
    return list(dict.fromkeys(blockers))


def validate_task_board_evidence(root: str | Path, board: TaskBoard, *, contract_id: str, contract_sha256: str) -> list[str]:
    root = Path(root).resolve()
    errors: list[str] = []
    if board.contract_id != contract_id:
        errors.append("TASK_BOARD_CONTRACT_ID_MISMATCH")
    if board.contract_sha256 != contract_sha256:
        errors.append("TASK_BOARD_CONTRACT_HASH_MISMATCH")
    for step in board.planned_steps:
        if step.status.value == "COMPLETED":
            for relative in step.evidence:
                path = Path(relative)
                if path.is_absolute() or ".." in path.parts:
                    errors.append(f"TASK_EVIDENCE_PATH_UNSAFE:{step.step_id}:{relative}")
                elif not (root / path).exists():
                    errors.append(f"TASK_EVIDENCE_MISSING:{step.step_id}:{relative}")
    return errors


def _acquire_lock(lock_path: Path, timeout_seconds: float = 10.0) -> int:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError("gate query state lock timeout")
            time.sleep(0.05)


def reserve_gate_query(
    contract: ImprovementContract,
    *,
    contract_sha256: str,
    candidate_id: str,
    state_dir: str | Path,
) -> tuple[GateQueryReceipt, Path]:
    state_root = Path(state_dir).resolve()
    contract_root = state_root / contract_sha256
    contract_root.mkdir(parents=True, exist_ok=True)
    lock_path = contract_root / ".reserve.lock"
    lock_fd = _acquire_lock(lock_path)
    try:
        receipts = sorted(contract_root.glob("query-*.json"))
        if len(receipts) >= contract.resource_budget.max_formal_gate_queries:
            raise RuntimeError("FORMAL_GATE_QUERY_BUDGET_EXCEEDED")
        query_index = len(receipts) + 1
        receipt = GateQueryReceipt(
            receipt_id=f"receipt-{uuid.uuid4().hex}",
            contract_id=contract.contract_id,
            contract_sha256=contract_sha256,
            candidate_id=candidate_id,
            query_index=query_index,
            state_id=contract_sha256,
        )
        final_path = contract_root / f"query-{query_index:04d}.json"
        fd, temp_name = tempfile.mkstemp(prefix=".query-", suffix=".tmp", dir=contract_root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(receipt.model_dump_json(indent=2) + "\n")
            os.replace(temp_name, final_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return receipt, final_path
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


def verify_gate_query_receipt(
    evidence: GateEvidence,
    *,
    receipt_path: str | Path,
    contract: ImprovementContract,
    expected_contract_sha256: str,
) -> list[str]:
    errors: list[str] = []
    path = Path(receipt_path)
    if not path.is_file():
        return ["GATE_QUERY_RECEIPT_MISSING"]
    if sha256_file(path) != evidence.gate_query_receipt_sha256:
        errors.append("GATE_QUERY_RECEIPT_HASH_MISMATCH")
    receipt = GateQueryReceipt.model_validate_json(path.read_text(encoding="utf-8"))
    if receipt.contract_id != contract.contract_id or receipt.contract_sha256 != expected_contract_sha256:
        errors.append("GATE_QUERY_RECEIPT_CONTRACT_MISMATCH")
    if receipt.candidate_id != evidence.candidate_id or receipt.query_index != evidence.gate_query_index:
        errors.append("GATE_QUERY_RECEIPT_CANDIDATE_OR_INDEX_MISMATCH")
    return errors


def verify_evidence_artifacts(evidence: GateEvidence, evidence_root: str | Path) -> list[str]:
    root = Path(evidence_root).resolve()
    errors: list[str] = []
    for relative, expected in evidence.evidence_artifacts.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            errors.append(f"EVIDENCE_PATH_UNSAFE:{relative}")
            continue
        target = root / path
        if not target.is_file():
            errors.append(f"EVIDENCE_ARTIFACT_MISSING:{relative}")
        elif sha256_file(target) != expected:
            errors.append(f"EVIDENCE_ARTIFACT_HASH_MISMATCH:{relative}")
    return errors


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def decide_release(
    contract: ImprovementContract,
    evidence: GateEvidence,
    *,
    expected_contract_sha256: str,
    expected_evaluator_sha256: str,
    expected_evaluator_module_sha256: str,
    evidence_artifact_errors: list[str] | None = None,
    query_receipt_errors: list[str] | None = None,
) -> ReleaseDecision:
    blockers: list[str] = []
    escalations: list[str] = []
    rejections: list[str] = []

    if evidence.contract_id != contract.contract_id:
        blockers.append("GATE_CONTRACT_ID_MISMATCH")
    if evidence.contract_sha256 != expected_contract_sha256:
        blockers.append("GATE_CONTRACT_HASH_MISMATCH")
    if evidence.evaluator_sha256 != expected_evaluator_sha256:
        blockers.append("EVALUATOR_ENTRYPOINT_HASH_MISMATCH")
    if evidence.evaluator_module_sha256 != expected_evaluator_module_sha256:
        blockers.append("EVALUATOR_MODULE_HASH_MISMATCH")
    if evidence.gate_query_index > contract.resource_budget.max_formal_gate_queries:
        blockers.append("FORMAL_GATE_QUERY_BUDGET_EXCEEDED")
    blockers.extend(evidence_artifact_errors or [])
    blockers.extend(query_receipt_errors or [])
    if evidence.modified_evaluator_paths:
        blockers.append("EVALUATOR_BOUNDARY_MODIFIED")
    if evidence.hidden_truth_access_status == HiddenTruthAccessStatus.VIOLATION or evidence.hidden_truth_accessed is True:
        blockers.append("HIDDEN_TRUTH_ACCESSED")
    if evidence.hidden_truth_access_status == HiddenTruthAccessStatus.UNVERIFIED:
        escalations.append("HIDDEN_TRUTH_ACCESS_UNVERIFIED")
    if evidence.candidate_tree_sha256_before != evidence.candidate_tree_sha256_after:
        blockers.append("CANDIDATE_TREE_MUTATED_DURING_GATE")
    if not evidence.source_tree_clean:
        blockers.append("SOURCE_TREE_NOT_CLEAN_AFTER_GATE")
    if evidence.tests_collected < contract.baseline_test_count:
        rejections.append("TEST_COUNT_REGRESSION")
    if evidence.baseline_test_nodeids_missing:
        rejections.append("BASELINE_TEST_IDENTITY_REGRESSION")
    if evidence.tests_timed_out:
        escalations.append("TEST_EXECUTION_TIMEOUT")
    if evidence.tests_failed or evidence.tests_errors:
        rejections.append("TEST_FAILURES_PRESENT")
    if evidence.tests_passed + evidence.tests_failed + evidence.tests_errors + evidence.tests_skipped != evidence.tests_collected:
        blockers.append("TEST_COUNT_ARITHMETIC_INVALID")
    started, finished = _parse_iso(evidence.started_at), _parse_iso(evidence.finished_at)
    if started is None or finished is None or finished < started:
        blockers.append("GATE_TIME_RANGE_INVALID")

    by_id = {item.criterion_id: item for item in evidence.checks}
    for criterion in contract.acceptance_criteria:
        check = by_id.get(criterion.criterion_id)
        if check is None:
            escalations.append(f"REQUIRED_GATE_CHECK_MISSING:{criterion.criterion_id}")
            continue
        if check.severity != criterion.severity:
            blockers.append(f"GATE_SEVERITY_MISMATCH:{criterion.criterion_id}")
        if criterion.severity == GateSeverity.HARD and check.status != criterion.expected_status:
            if check.status in {CheckStatus.NOT_RUN, CheckStatus.TIMEOUT, CheckStatus.ERROR}:
                escalations.append(f"HARD_GATE_INCOMPLETE:{criterion.criterion_id}:{check.status.value}")
            else:
                rejections.append(f"HARD_GATE_FAILED:{criterion.criterion_id}")

    if blockers:
        status = ReleaseDecisionStatus.BLOCKED
        reasons = ["Candidate violated a frozen integrity, isolation, provenance, or budget guardrail."]
    elif escalations:
        status = ReleaseDecisionStatus.ESCALATE
        reasons = ["Formal evidence is incomplete or not independently verifiable; acceptance is prohibited."]
    elif rejections:
        status = ReleaseDecisionStatus.REJECT
        reasons = ["Candidate completed evaluation but failed one or more frozen acceptance criteria."]
    else:
        status = ReleaseDecisionStatus.ACCEPT
        reasons = ["All frozen hard gates passed with externally pinned contract and evaluator evidence."]

    all_blockers = list(dict.fromkeys(blockers + escalations + rejections))
    return ReleaseDecision(
        decision_id=f"decision-{evidence.gate_run_id}",
        contract_id=contract.contract_id,
        contract_sha256=evidence.contract_sha256,
        candidate_id=evidence.candidate_id,
        gate_run_id=evidence.gate_run_id,
        gate_evidence_sha256=sha256_json(evidence.model_dump(mode="json")),
        status=status,
        reasons=reasons,
        blockers=all_blockers,
        requires_human_review=status in {ReleaseDecisionStatus.ESCALATE, ReleaseDecisionStatus.BLOCKED},
        claim_boundary=(
            "This decision covers the frozen engineering contract only. It does not establish improved "
            "diagnostic accuracy, Basilisk data validity, causal validity, flight qualification, or safety qualification."
        ),
    )
