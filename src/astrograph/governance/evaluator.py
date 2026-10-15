from __future__ import annotations

import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from astrograph.governance.kernel import (
    load_json_yaml,
    sha256_file,
    tree_sha256,
    validate_change_request_against_candidate,
    validate_contract_candidate,
    validate_task_board_evidence,
)
from astrograph.governance.ledger import AppendOnlyJsonlLedger
from astrograph.governance.models import (
    ChangeRequest,
    CheckStatus,
    GateCheck,
    GateEvidence,
    GateQueryReceipt,
    GateSeverity,
    HiddenTruthAccessStatus,
    ImprovementContract,
    TaskBoard,
    utc_now,
)


class IsolatedGateEvaluator:
    evaluator_id = "astrograph-isolated-gate-evaluator.v2"

    def __init__(self, evaluator_path: str | Path):
        self.evaluator_path = Path(evaluator_path).resolve()
        self.evaluator_module_path = Path(inspect.getfile(IsolatedGateEvaluator)).resolve()

    @staticmethod
    def _run(command: list[str], *, cwd: Path, timeout_seconds: int, env: dict[str, str]) -> tuple[CheckStatus, subprocess.CompletedProcess[str] | None, float, str]:
        started = time.monotonic()
        try:
            result = subprocess.run(
                command, cwd=cwd, env=env, capture_output=True, text=True,
                timeout=timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired:
            return CheckStatus.TIMEOUT, None, time.monotonic() - started, "COMMAND_TIMEOUT"
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return CheckStatus.ERROR, None, time.monotonic() - started, exc.__class__.__name__
        status = CheckStatus.PASS if result.returncode == 0 else CheckStatus.FAIL
        return status, result, time.monotonic() - started, "" if status == CheckStatus.PASS else f"EXIT_{result.returncode}"

    @staticmethod
    def _run_to_files(
        command: list[str], *, cwd: Path, timeout_seconds: int, env: dict[str, str],
        stdout_path: Path, stderr_path: Path,
    ) -> tuple[CheckStatus, float, str, int | None]:
        """Run a command with file-backed output, avoiding inherited pipe deadlocks."""
        started = time.monotonic()
        try:
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            with stdout_path.open("w", encoding="utf-8") as stdout_stream, stderr_path.open("w", encoding="utf-8") as stderr_stream:
                result = subprocess.run(
                    command, cwd=cwd, env=env, stdout=stdout_stream, stderr=stderr_stream,
                    text=True, timeout=timeout_seconds, check=False,
                )
        except subprocess.TimeoutExpired:
            return CheckStatus.TIMEOUT, time.monotonic() - started, "COMMAND_TIMEOUT", None
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return CheckStatus.ERROR, time.monotonic() - started, exc.__class__.__name__, None
        status = CheckStatus.PASS if result.returncode == 0 else CheckStatus.FAIL
        return status, time.monotonic() - started, "" if status == CheckStatus.PASS else f"EXIT_{result.returncode}", result.returncode

    @staticmethod
    def _junit_counts(path: Path) -> tuple[int, int, int, int]:
        root = ET.parse(path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        # Parent <testsuites> nodes can duplicate child totals. Prefer leaf suites.
        leaf_suites = [item for item in suites if not list(item.findall("testsuite"))]
        suites = leaf_suites or suites
        collected = sum(int(item.attrib.get("tests", "0")) for item in suites)
        failures = sum(int(item.attrib.get("failures", "0")) for item in suites)
        errors = sum(int(item.attrib.get("errors", "0")) for item in suites)
        skipped = sum(int(item.attrib.get("skipped", "0")) for item in suites)
        return collected, failures, errors, skipped

    @staticmethod
    def _nodeids(
        text: str, *, candidate_root: Path | None = None, collect_targets: list[str] | None = None
    ) -> list[str]:
        nodeids: set[str] = set()
        explicit_files: list[str] = []
        if candidate_root is not None:
            explicit_files = [
                target for target in (collect_targets or []) if (candidate_root / target).is_file()
            ]
        for line in text.splitlines():
            value = line.strip()
            if "::" not in value or value.startswith("="):
                continue
            relative, suffix = value.split("::", 1)
            if not relative and len(explicit_files) == 1:
                relative = explicit_files[0]
            elif relative and "/" not in relative:
                explicit_name_matches = [item for item in explicit_files if Path(item).name == relative]
                if len(explicit_name_matches) == 1:
                    relative = explicit_name_matches[0]
            # Pytest strips the explicit ``tests/`` argument from displayed nodeids
            # when the candidate is collected from that directory.  Frozen inventories
            # retain repository-relative paths, so canonicalize only genuine top-level
            # test files and preserve explicit paths such as ``examples/...``.
            if candidate_root is not None and "/" not in relative and (candidate_root / "tests" / relative).is_file():
                relative = f"tests/{relative}"
            nodeids.add(f"{relative}::{suffix}")
        return sorted(nodeids)

    @staticmethod
    def _collect_targets(baseline_nodeids: list[str]) -> list[str]:
        targets = {"tests"}
        for nodeid in baseline_nodeids:
            relative = nodeid.split("::", 1)[0]
            if "/" in relative and not relative.startswith("tests/"):
                targets.add(relative)
        return sorted(targets)

    @staticmethod
    def _partition_test_targets(nodeids: list[str], *, max_partitions: int = 6) -> list[list[str]]:
        """Split tests into deterministic process-isolation partitions.

        Most files are count-balanced.  Multiprocessing SQLite nodeids are isolated
        individually because combining them with a large pytest session can leave a
        resource-tracker/queue cleanup wait after the assertions have completed.  The
        frozen nodeids are not skipped or altered; each is still executed exactly once.
        """
        isolate_files = {"tests/test_sqlite_idempotency.py"}
        isolated: list[list[str]] = []
        counts: dict[str, int] = {}
        for nodeid in nodeids:
            relative = nodeid.split("::", 1)[0]
            if "/" not in relative:
                relative = f"tests/{relative}"
            canonical = nodeid if nodeid.startswith(relative + "::") else f"{relative}::{nodeid.split('::', 1)[1]}"
            if relative in isolate_files:
                isolated.append([canonical])
            else:
                counts[relative] = counts.get(relative, 0) + 1
        if not counts:
            return sorted(isolated)
        partition_count = min(max(1, max_partitions), len(counts))
        partitions: list[list[str]] = [[] for _ in range(partition_count)]
        loads = [0] * partition_count
        for relative, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            index = min(range(partition_count), key=lambda item: (loads[item], item))
            partitions[index].append(relative)
            loads[index] += count
        return sorted(isolated) + [sorted(items) for items in partitions if items]

    @classmethod
    def _merge_junit_files(cls, paths: list[Path], output: Path) -> tuple[int, int, int, int]:
        aggregate = ET.Element("testsuites")
        total_tests = total_failures = total_errors = total_skipped = 0
        for path in paths:
            root = ET.parse(path).getroot()
            suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
            for suite in suites:
                aggregate.append(suite)
            tests, failures, errors, skipped = cls._junit_counts(path)
            total_tests += tests
            total_failures += failures
            total_errors += errors
            total_skipped += skipped
        aggregate.attrib.update({
            "tests": str(total_tests),
            "failures": str(total_failures),
            "errors": str(total_errors),
            "skipped": str(total_skipped),
        })
        ET.ElementTree(aggregate).write(output, encoding="utf-8", xml_declaration=True)
        return total_tests, total_failures, total_errors, total_skipped

    @staticmethod
    def _relative(path: Path, root: Path) -> str:
        return path.resolve().relative_to(root.resolve()).as_posix()

    @classmethod
    def _evaluate_unified_runtime_external_gate(
        cls,
        *,
        candidate_root: Path,
        output_dir: Path,
        env: dict[str, str],
        timeout_seconds: int,
    ) -> GateCheck:
        """Execute the frozen unified-runtime gate inside the formal evidence run.

        A pre-generated PASS file is not sufficient: the formal evaluator executes the
        pinned gate program against the frozen trusted-data prerequisite and the frozen
        runtime-evaluation artifact, writes its own result outside the candidate tree,
        and records the process/output status as a hard gate check.
        """
        evaluator = candidate_root / "governance/evaluators/unified_runtime_gate.py"
        evaluation = candidate_root / "governance/evidence/0.7.0/unified_runtime_evaluation.json"
        trusted_gate = candidate_root / "governance/evidence/0.6.7/trusted_data_unlock_gate_decision.json"
        result_path = output_dir / "unified_runtime_external_gate.json"
        stdout_path = output_dir / "unified_runtime_external_gate.stdout.log"
        stderr_path = output_dir / "unified_runtime_external_gate.stderr.log"
        missing = [
            cls._relative(path, candidate_root)
            for path in (evaluator, evaluation, trusted_gate)
            if not path.is_file()
        ]
        if missing:
            return GateCheck(
                criterion_id="UNIFIED_RUNTIME_EXTERNAL_GATE",
                status=CheckStatus.FAIL,
                severity=GateSeverity.HARD,
                observed="missing=" + ",".join(missing),
                expected="frozen external gate, evaluation, and trusted prerequisite exist",
                error_code="UNIFIED_RUNTIME_EXTERNAL_GATE_INPUT_MISSING",
            )
        status, duration, error, returncode = cls._run_to_files(
            [
                sys.executable, str(evaluator),
                "--evaluation", str(evaluation),
                "--trusted-gate", str(trusted_gate),
                "--output", str(result_path),
            ],
            cwd=candidate_root,
            timeout_seconds=timeout_seconds,
            env=env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        payload_status = "MISSING"
        parse_error = ""
        if result_path.is_file():
            try:
                payload_status = str(json.loads(result_path.read_text(encoding="utf-8")).get("status", "MISSING"))
            except (OSError, UnicodeError, json.JSONDecodeError, AttributeError, TypeError) as exc:
                parse_error = exc.__class__.__name__
        passed = status == CheckStatus.PASS and returncode == 0 and payload_status == "PASS" and not parse_error
        evidence_paths = [stdout_path.name, stderr_path.name]
        if result_path.is_file():
            evidence_paths.append(result_path.name)
        return GateCheck(
            criterion_id="UNIFIED_RUNTIME_EXTERNAL_GATE",
            status=CheckStatus.PASS if passed else (status if status != CheckStatus.PASS else CheckStatus.FAIL),
            severity=GateSeverity.HARD,
            observed=f"process={status.value};returncode={returncode};payload={payload_status}",
            expected="process=PASS;returncode=0;payload=PASS",
            evidence_paths=evidence_paths,
            duration_seconds=duration,
            error_code="" if passed else (parse_error or error or "UNIFIED_RUNTIME_EXTERNAL_GATE_FAILED"),
        )

    @classmethod
    def _evaluate_trusted_single_fault_external_gate(
        cls,
        *,
        candidate_root: Path,
        output_dir: Path,
        env: dict[str, str],
        timeout_seconds: int,
    ) -> GateCheck:
        evaluator = candidate_root / "governance/evaluators/trusted_single_fault_gate.py"
        evaluation = candidate_root / "governance/evidence/0.7.1/trusted_single_fault_evaluation.json"
        result_path = output_dir / "trusted_single_fault_external_gate.json"
        stdout_path = output_dir / "trusted_single_fault_external_gate.stdout.log"
        stderr_path = output_dir / "trusted_single_fault_external_gate.stderr.log"
        missing = [
            cls._relative(path, candidate_root)
            for path in (evaluator, evaluation)
            if not path.is_file()
        ]
        if missing:
            return GateCheck(
                criterion_id="TRUSTED_SINGLE_FAULT_EXTERNAL_GATE",
                status=CheckStatus.FAIL,
                severity=GateSeverity.HARD,
                observed="missing=" + ",".join(missing),
                expected="frozen external gate and evaluation exist",
                error_code="TRUSTED_SINGLE_FAULT_EXTERNAL_GATE_INPUT_MISSING",
            )
        status, duration, error, returncode = cls._run_to_files(
            [
                sys.executable, str(evaluator),
                "--evaluation", str(evaluation),
                "--output", str(result_path),
            ],
            cwd=candidate_root, timeout_seconds=timeout_seconds, env=env,
            stdout_path=stdout_path, stderr_path=stderr_path,
        )
        payload_status = "MISSING"
        parse_error = ""
        if result_path.is_file():
            try:
                payload_status = str(json.loads(result_path.read_text(encoding="utf-8")).get("status", "MISSING"))
            except (OSError, UnicodeError, json.JSONDecodeError, AttributeError, TypeError) as exc:
                parse_error = exc.__class__.__name__
        passed = status == CheckStatus.PASS and returncode == 0 and payload_status == "PASS" and not parse_error
        evidence_paths = [stdout_path.name, stderr_path.name]
        if result_path.is_file():
            evidence_paths.append(result_path.name)
        return GateCheck(
            criterion_id="TRUSTED_SINGLE_FAULT_EXTERNAL_GATE",
            status=CheckStatus.PASS if passed else (status if status != CheckStatus.PASS else CheckStatus.FAIL),
            severity=GateSeverity.HARD,
            observed=f"process={status.value};returncode={returncode};payload={payload_status}",
            expected="process=PASS;returncode=0;payload=PASS",
            evidence_paths=evidence_paths,
            duration_seconds=duration,
            error_code="" if passed else (parse_error or error or "TRUSTED_SINGLE_FAULT_EXTERNAL_GATE_FAILED"),
        )

    @classmethod
    def _evaluate_expanded_single_fault_external_gate(
        cls,
        *,
        candidate_root: Path,
        output_dir: Path,
        env: dict[str, str],
        timeout_seconds: int,
    ) -> GateCheck:
        """Execute the frozen 4C expanded single-fault gate against external data.

        The candidate contains only a frozen data-identity manifest.  Restricted
        truth and public telemetry are supplied by the trusted host through
        environment variables and must remain outside the candidate source tree.
        """
        evaluator = candidate_root / "governance/evaluators/expanded_single_fault_gate.py"
        identity = candidate_root / "governance/evidence/0.7.1-extension/expanded_single_fault_dataset_identity.json"
        truth_value = env.get("ASTROGRAPH_4C_TRUTH", "")
        formal_value = env.get("ASTROGRAPH_4C_FORMAL_PUBLIC", "")
        research_value = env.get("ASTROGRAPH_4C_RESEARCH_PUBLIC", "")
        truth = Path(truth_value).resolve() if truth_value else Path()
        formal_public = Path(formal_value).resolve() if formal_value else Path()
        research_public = Path(research_value).resolve() if research_value else Path()
        result_path = output_dir / "expanded_single_fault_external_gate.json"
        stdout_path = output_dir / "expanded_single_fault_external_gate.stdout.log"
        stderr_path = output_dir / "expanded_single_fault_external_gate.stderr.log"
        missing: list[str] = []
        for path in (evaluator, identity):
            if not path.is_file():
                missing.append(cls._relative(path, candidate_root))
        external_inputs = {
            "ASTROGRAPH_4C_TRUTH": truth,
            "ASTROGRAPH_4C_FORMAL_PUBLIC": formal_public,
            "ASTROGRAPH_4C_RESEARCH_PUBLIC": research_public,
        }
        for name, path in external_inputs.items():
            if not str(path) or not path.exists():
                missing.append(name)
                continue
            try:
                path.relative_to(candidate_root)
            except ValueError:
                pass
            else:
                missing.append(f"{name}:MUST_BE_EXTERNAL")
        if missing:
            return GateCheck(
                criterion_id="EXPANDED_SINGLE_FAULT_EXTERNAL_GATE",
                status=CheckStatus.FAIL,
                severity=GateSeverity.HARD,
                observed="missing_or_invalid=" + ",".join(missing),
                expected="frozen gate/identity and external truth/public inputs exist",
                error_code="EXPANDED_SINGLE_FAULT_EXTERNAL_GATE_INPUT_MISSING",
            )
        status, duration, error, returncode = cls._run_to_files(
            [
                sys.executable, str(evaluator),
                "--project-root", str(candidate_root),
                "--truth", str(truth),
                "--public-root", str(formal_public),
                "--public-root", str(research_public),
                "--identity", str(identity),
                "--output", str(result_path),
            ],
            cwd=candidate_root,
            timeout_seconds=timeout_seconds,
            env=env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        payload_status = "MISSING"
        checks_pass = False
        parse_error = ""
        if result_path.is_file():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                payload_status = str(payload.get("status", "MISSING"))
                checks_pass = bool(payload.get("checks")) and all(payload["checks"].values())
            except (OSError, UnicodeError, json.JSONDecodeError, AttributeError, TypeError) as exc:
                parse_error = exc.__class__.__name__
        passed = (
            status == CheckStatus.PASS
            and returncode == 0
            and payload_status == "PASS"
            and checks_pass
            and not parse_error
        )
        evidence_paths = [stdout_path.name, stderr_path.name]
        if result_path.is_file():
            evidence_paths.append(result_path.name)
        return GateCheck(
            criterion_id="EXPANDED_SINGLE_FAULT_EXTERNAL_GATE",
            status=CheckStatus.PASS if passed else (status if status != CheckStatus.PASS else CheckStatus.FAIL),
            severity=GateSeverity.HARD,
            observed=(
                f"process={status.value};returncode={returncode};"
                f"payload={payload_status};checks={checks_pass}"
            ),
            expected="process=PASS;returncode=0;payload=PASS;checks=true",
            evidence_paths=evidence_paths,
            duration_seconds=duration,
            error_code="" if passed else (parse_error or error or "EXPANDED_SINGLE_FAULT_EXTERNAL_GATE_FAILED"),
        )

    def run(
        self,
        *,
        contract_path: str | Path,
        candidate_root: str | Path,
        expected_contract_sha256: str,
        candidate_id: str,
        output_dir: str | Path,
        gate_query_receipt: GateQueryReceipt,
        gate_query_receipt_source: str | Path,
        hidden_truth_access_status: HiddenTruthAccessStatus = HiddenTruthAccessStatus.NOT_APPLICABLE,
    ) -> GateEvidence:
        candidate_root = Path(candidate_root).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        contract = load_json_yaml(contract_path, ImprovementContract)
        started_at = utc_now()
        before = tree_sha256(candidate_root)
        evaluator_sha = sha256_file(self.evaluator_path)
        evaluator_module_sha = sha256_file(self.evaluator_module_path)
        checks: list[GateCheck] = []

        receipt_copy = output_dir / "gate_query_receipt.json"
        shutil.copy2(gate_query_receipt_source, receipt_copy)

        validation = validate_contract_candidate(
            contract,
            contract_path=contract_path,
            candidate_root=candidate_root,
            expected_contract_sha256=expected_contract_sha256,
        )
        validation_path = output_dir / "contract_validation.json"
        validation_path.write_text(validation.model_dump_json(indent=2) + "\n", encoding="utf-8")
        checks.append(GateCheck(
            criterion_id="CONTRACT_AND_SCOPE_INTEGRITY",
            status=validation.status,
            severity=GateSeverity.HARD,
            observed=";".join(validation.blocking_reasons) or "PASS",
            evidence_paths=[validation_path.name],
            error_code=validation.blocking_reasons[0] if validation.blocking_reasons else "",
        ))

        change_request_path = candidate_root / f"governance/change_requests/{contract.target_version}.json"
        change_errors: list[str] = []
        if not change_request_path.is_file():
            change_errors = ["CHANGE_REQUEST_MISSING"]
        else:
            request = ChangeRequest.model_validate_json(change_request_path.read_text(encoding="utf-8"))
            change_errors = validate_change_request_against_candidate(
                contract, request, validation, expected_contract_sha256=expected_contract_sha256,
            )
        change_report = output_dir / "change_request_validation.json"
        change_report.write_text(json.dumps({"status": "PASS" if not change_errors else "FAIL", "errors": change_errors}, indent=2) + "\n", encoding="utf-8")
        checks.append(GateCheck(
            criterion_id="CHANGE_REQUEST_COMPLETENESS",
            status=CheckStatus.PASS if not change_errors else CheckStatus.FAIL,
            severity=GateSeverity.HARD,
            observed="PASS" if not change_errors else ";".join(change_errors),
            evidence_paths=[change_report.name],
            error_code=change_errors[0] if change_errors else "",
        ))

        board_path = candidate_root / f"governance/task_boards/{contract.target_version}.json"
        board_errors: list[str] = []
        if not board_path.is_file():
            board_errors = ["TASK_BOARD_MISSING"]
        else:
            board = TaskBoard.model_validate_json(board_path.read_text(encoding="utf-8"))
            board_errors = validate_task_board_evidence(
                candidate_root, board, contract_id=contract.contract_id, contract_sha256=expected_contract_sha256,
            )
        board_report = output_dir / "task_board_validation.json"
        board_report.write_text(json.dumps({"status": "PASS" if not board_errors else "FAIL", "errors": board_errors}, indent=2) + "\n", encoding="utf-8")
        checks.append(GateCheck(
            criterion_id="TASK_BOARD_EVIDENCE",
            status=CheckStatus.PASS if not board_errors else CheckStatus.FAIL,
            severity=GateSeverity.HARD,
            observed="PASS" if not board_errors else ";".join(board_errors),
            evidence_paths=[board_report.name],
            error_code=board_errors[0] if board_errors else "",
        ))

        ledger_results: dict[str, dict] = {}
        ledger_errors: list[str] = []
        for relative in ("governance/passports/materials.jsonl", "governance/failure_memory/permanent.jsonl"):
            result = AppendOnlyJsonlLedger(candidate_root / relative).audit()
            ledger_results[relative] = result.model_dump(mode="json")
            if result.status != "PASS":
                ledger_errors.extend(f"{relative}:{item}" for item in result.errors)
        ledger_report = output_dir / "ledger_audit.json"
        ledger_report.write_text(json.dumps({"status": "PASS" if not ledger_errors else "FAIL", "ledgers": ledger_results, "errors": ledger_errors}, indent=2) + "\n", encoding="utf-8")
        checks.append(GateCheck(
            criterion_id="LEDGER_INTEGRITY",
            status=CheckStatus.PASS if not ledger_errors else CheckStatus.FAIL,
            severity=GateSeverity.HARD,
            observed="PASS" if not ledger_errors else ";".join(ledger_errors),
            evidence_paths=[ledger_report.name],
            error_code=ledger_errors[0] if ledger_errors else "",
        ))

        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONHASHSEED"] = "0"
        env["PYTEST_ADDOPTS"] = ""
        candidate_src = str(candidate_root / "src")
        env["PYTHONPATH"] = candidate_src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        tests_collected = tests_failed = tests_errors = tests_skipped = tests_passed = 0
        tests_timed_out = False
        baseline_missing: list[str] = []
        current_nodeids: list[str] = []

        baseline_nodeids: list[str] = []
        if contract.baseline_test_inventory_path:
            inventory_payload = json.loads((candidate_root / contract.baseline_test_inventory_path).read_text(encoding="utf-8"))
            baseline_nodeids = list(inventory_payload.get("nodeids", []))
        collect_targets = self._collect_targets(baseline_nodeids)
        collect_status, collect_result, collect_duration, collect_error = self._run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "-c", os.devnull, *collect_targets],
            cwd=candidate_root,
            timeout_seconds=min(contract.resource_budget.max_runtime_seconds, 120),
            env=env,
        )
        collect_stdout = output_dir / "pytest.collect.stdout.log"
        collect_stderr = output_dir / "pytest.collect.stderr.log"
        collect_stdout.write_text(collect_result.stdout if collect_result else "", encoding="utf-8")
        collect_stderr.write_text(collect_result.stderr if collect_result else collect_error, encoding="utf-8")
        if collect_result:
            current_nodeids = self._nodeids(
                collect_result.stdout, candidate_root=candidate_root, collect_targets=collect_targets
            )
        if contract.baseline_test_inventory_path:
            baseline_missing = sorted(set(baseline_nodeids) - set(current_nodeids))
        identity_status = CheckStatus.PASS if collect_status == CheckStatus.PASS and not baseline_missing else (collect_status if collect_status != CheckStatus.PASS else CheckStatus.FAIL)
        identity_report = output_dir / "test_identity.json"
        identity_report.write_text(json.dumps({
            "status": identity_status.value,
            "current_count": len(current_nodeids),
            "baseline_count": len(baseline_nodeids) if contract.baseline_test_inventory_path else 0,
            "missing_baseline_nodeids": baseline_missing,
        }, indent=2) + "\n", encoding="utf-8")
        checks.append(GateCheck(
            criterion_id="BASELINE_TEST_IDENTITY",
            status=identity_status,
            severity=GateSeverity.HARD,
            observed=f"current={len(current_nodeids)};missing={len(baseline_missing)}",
            expected="all frozen baseline nodeids remain collectable",
            evidence_paths=[collect_stdout.name, collect_stderr.name, identity_report.name],
            duration_seconds=collect_duration,
            error_code=collect_error or ("BASELINE_TEST_IDENTITY_REGRESSION" if baseline_missing else ""),
        ))

        with tempfile.TemporaryDirectory(prefix="astrograph-guardrail-gate-") as temp_dir:
            temp_root = Path(temp_dir)
            partition_targets = self._partition_test_targets(current_nodeids)
            partition_reports: list[dict[str, object]] = []
            partition_junits: list[Path] = []
            partition_evidence_paths: list[str] = []
            suite_started = time.monotonic()
            aggregate_status = CheckStatus.PASS
            pytest_error = ""

            for index, targets in enumerate(partition_targets, start=1):
                elapsed = time.monotonic() - suite_started
                remaining = max(1, int(contract.resource_budget.max_runtime_seconds - elapsed))
                junit_temp = temp_root / f"pytest.partition-{index:02d}.xml"
                stdout_name = f"pytest.partition-{index:02d}.stdout.log"
                stderr_name = f"pytest.partition-{index:02d}.stderr.log"
                status, duration, error, returncode = self._run_to_files(
                    [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-c", os.devnull, *targets, f"--junitxml={junit_temp}"],
                    cwd=candidate_root,
                    timeout_seconds=remaining,
                    env=env,
                    stdout_path=output_dir / stdout_name,
                    stderr_path=output_dir / stderr_name,
                )
                partition_evidence_paths.extend([stdout_name, stderr_name])
                counts = (0, 0, 0, 0)
                junit_name = ""
                if junit_temp.is_file():
                    junit_name = f"pytest.partition-{index:02d}.xml"
                    shutil.copy2(junit_temp, output_dir / junit_name)
                    partition_junits.append(junit_temp)
                    partition_evidence_paths.append(junit_name)
                    counts = self._junit_counts(junit_temp)
                partition_reports.append({
                    "partition": index,
                    "targets": targets,
                    "status": status.value,
                    "returncode": returncode,
                    "duration_seconds": duration,
                    "error_code": error,
                    "tests": counts[0],
                    "failures": counts[1],
                    "errors": counts[2],
                    "skipped": counts[3],
                    "junit": junit_name,
                })
                if status == CheckStatus.TIMEOUT:
                    aggregate_status = CheckStatus.TIMEOUT
                    tests_timed_out = True
                    pytest_error = error or "COMMAND_TIMEOUT"
                    break
                if status != CheckStatus.PASS and aggregate_status == CheckStatus.PASS:
                    aggregate_status = status
                    pytest_error = error

            partition_report_path = output_dir / "pytest.partitions.json"
            partition_report_path.write_text(json.dumps({
                "partition_count": len(partition_reports),
                "collected_identity_count": len(current_nodeids),
                "partitions": partition_reports,
            }, indent=2) + "\n", encoding="utf-8")
            partition_evidence_paths.insert(0, partition_report_path.name)

            junit_path = output_dir / "pytest.xml"
            if partition_junits:
                tests_collected, tests_failed, tests_errors, tests_skipped = self._merge_junit_files(partition_junits, junit_path)
                tests_passed = tests_collected - tests_failed - tests_errors - tests_skipped
                partition_evidence_paths.append(junit_path.name)
            duration = time.monotonic() - suite_started
            if tests_collected != len(current_nodeids) and aggregate_status == CheckStatus.PASS:
                aggregate_status = CheckStatus.FAIL
                pytest_error = "PARTITION_TEST_IDENTITY_MISMATCH"
            checks.append(GateCheck(
                criterion_id="FULL_TEST_SUITE",
                status=aggregate_status,
                severity=GateSeverity.HARD,
                observed=f"collected={tests_collected};passed={tests_passed};failed={tests_failed};errors={tests_errors};skipped={tests_skipped};partitions={len(partition_reports)}",
                expected=f"collected={len(current_nodeids)};collected>={contract.baseline_test_count};failed=0;errors=0;timeout=false",
                evidence_paths=partition_evidence_paths,
                duration_seconds=duration,
                error_code=pytest_error,
            ))
            count_status = CheckStatus.PASS if tests_collected >= contract.baseline_test_count and tests_collected == len(current_nodeids) and not tests_timed_out else CheckStatus.FAIL
            checks.append(GateCheck(
                criterion_id="TEST_COUNT_NON_REGRESSION",
                status=count_status,
                severity=GateSeverity.HARD,
                observed=str(tests_collected),
                expected=f"={len(current_nodeids)} and >={contract.baseline_test_count}",
                evidence_paths=[junit_path.name] if junit_path.is_file() else [partition_report_path.name],
                error_code="TEST_COUNT_REGRESSION" if count_status != CheckStatus.PASS else "",
            ))

            if any(item.criterion_id == "UNIFIED_RUNTIME_EXTERNAL_GATE" for item in contract.acceptance_criteria):
                checks.append(self._evaluate_unified_runtime_external_gate(
                    candidate_root=candidate_root,
                    output_dir=output_dir,
                    env=env,
                    timeout_seconds=min(contract.resource_budget.max_runtime_seconds, 120),
                ))

            if any(item.criterion_id == "TRUSTED_SINGLE_FAULT_EXTERNAL_GATE" for item in contract.acceptance_criteria):
                checks.append(self._evaluate_trusted_single_fault_external_gate(
                    candidate_root=candidate_root,
                    output_dir=output_dir,
                    env=env,
                    timeout_seconds=min(contract.resource_budget.max_runtime_seconds, 120),
                ))

            if any(item.criterion_id == "EXPANDED_SINGLE_FAULT_EXTERNAL_GATE" for item in contract.acceptance_criteria):
                checks.append(self._evaluate_expanded_single_fault_external_gate(
                    candidate_root=candidate_root,
                    output_dir=output_dir,
                    env=env,
                    timeout_seconds=min(contract.resource_budget.max_runtime_seconds, 240),
                ))

            distribution_path = output_dir / "distribution_audit.json"
            dist_status, dist_result, dist_duration, dist_error = self._run(
                [sys.executable, "scripts/data_driven/validate_distribution.py", "--root", str(candidate_root), "--output", str(distribution_path)],
                cwd=candidate_root,
                timeout_seconds=min(contract.resource_budget.max_runtime_seconds, 120),
                env=env,
            )
            dist_stdout = output_dir / "distribution.stdout.log"
            dist_stderr = output_dir / "distribution.stderr.log"
            dist_stdout.write_text(dist_result.stdout if dist_result else "", encoding="utf-8")
            dist_stderr.write_text(dist_result.stderr if dist_result else dist_error, encoding="utf-8")
            checks.append(GateCheck(
                criterion_id="DISTRIBUTION_INTEGRITY",
                status=dist_status,
                severity=GateSeverity.HARD,
                observed="PASS" if dist_status == CheckStatus.PASS else dist_error,
                evidence_paths=[distribution_path.name, dist_stdout.name, dist_stderr.name],
                duration_seconds=dist_duration,
                error_code=dist_error,
            ))

        budget_ok = validation.storage_growth_bytes <= contract.resource_budget.max_storage_growth_bytes
        checks.append(GateCheck(
            criterion_id="RESOURCE_BUDGET",
            status=CheckStatus.PASS if budget_ok else CheckStatus.FAIL,
            severity=GateSeverity.HARD,
            observed=f"storage_growth_bytes={validation.storage_growth_bytes};query_index={gate_query_receipt.query_index}",
            expected=f"storage_growth<={contract.resource_budget.max_storage_growth_bytes};query_index<={contract.resource_budget.max_formal_gate_queries}",
            evidence_paths=[receipt_copy.name, validation_path.name],
            error_code="RESOURCE_BUDGET_EXCEEDED" if not budget_ok else "",
        ))

        after = tree_sha256(candidate_root)
        clean = before == after
        checks.append(GateCheck(
            criterion_id="SOURCE_TREE_STABILITY",
            status=CheckStatus.PASS if clean else CheckStatus.FAIL,
            severity=GateSeverity.HARD,
            observed=f"before={before};after={after}",
            expected="tree hash unchanged",
            error_code="SOURCE_TREE_MUTATED_DURING_GATE" if not clean else "",
        ))

        expected_entry = next((item.sha256 for item in contract.frozen_artifacts if item.role == "formal-gate-entrypoint"), "")
        expected_module = next((item.sha256 for item in contract.frozen_artifacts if item.role == "isolated-evaluator"), "")
        evaluator_ok = bool(expected_entry and expected_module and evaluator_sha == expected_entry and evaluator_module_sha == expected_module)
        checks.append(GateCheck(
            criterion_id="EVALUATOR_INTEGRITY",
            status=CheckStatus.PASS if evaluator_ok else CheckStatus.FAIL,
            severity=GateSeverity.HARD,
            observed=f"entrypoint={evaluator_sha};module={evaluator_module_sha}",
            expected=f"entrypoint={expected_entry};module={expected_module}",
            evidence_paths=[],
            error_code="EVALUATOR_HASH_MISMATCH" if not evaluator_ok else "",
        ))

        artifacts: dict[str, str] = {}
        for path in sorted(output_dir.iterdir()):
            if path.is_file() and path.name not in {"gate_evidence.json", "release_decision.json"}:
                artifacts[path.name] = sha256_file(path)

        modified_evaluator_paths = [
            item.split(":", 1)[1] for item in validation.frozen_artifact_errors
            if any(token in item for token in ("run_isolated_gate.py", "governance/evaluator.py", "governance/kernel.py", "governance/models.py"))
        ]
        evidence = GateEvidence(
            gate_run_id=f"gate-{uuid.uuid4().hex}",
            contract_id=contract.contract_id,
            contract_sha256=sha256_file(contract_path),
            candidate_id=candidate_id,
            candidate_tree_sha256_before=before,
            candidate_tree_sha256_after=after,
            evaluator_id=self.evaluator_id,
            evaluator_sha256=evaluator_sha,
            evaluator_module_sha256=evaluator_module_sha,
            gate_query_index=gate_query_receipt.query_index,
            gate_query_receipt_path=receipt_copy.name,
            gate_query_receipt_sha256=sha256_file(receipt_copy),
            started_at=started_at,
            finished_at=utc_now(),
            checks=checks,
            evidence_artifacts=artifacts,
            tests_collected=tests_collected,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_skipped=tests_skipped,
            tests_errors=tests_errors,
            tests_timed_out=tests_timed_out,
            baseline_test_nodeids_total=len(current_nodeids),
            baseline_test_nodeids_missing=baseline_missing,
            hidden_truth_access_status=hidden_truth_access_status,
            hidden_truth_accessed=None,
            modified_evaluator_paths=modified_evaluator_paths,
            source_tree_clean=clean,
            candidate_storage_growth_bytes=validation.storage_growth_bytes,
            environment={
                "python": platform.python_version(),
                "platform": platform.platform(),
                "executable": sys.executable,
                "pytest_config": os.devnull,
                "pytest_target": "tests",
            },
        )
        (output_dir / "gate_evidence.json").write_text(evidence.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return evidence
