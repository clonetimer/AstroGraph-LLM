from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from astrograph.governance.evaluator import IsolatedGateEvaluator


def _write_junit(path: Path, *, tests: int, failures: int = 0, errors: int = 0, skipped: int = 0) -> None:
    suite = ET.Element(
        "testsuite",
        tests=str(tests),
        failures=str(failures),
        errors=str(errors),
        skipped=str(skipped),
    )
    for index in range(tests):
        ET.SubElement(suite, "testcase", classname="example", name=f"case_{index}")
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def test_partition_targets_cover_each_collected_file_once() -> None:
    nodeids = [
        "test_a.py::test_one",
        "test_a.py::test_two",
        "test_b.py::test_one",
        "tests/test_c.py::test_one",
        "tests/test_d.py::test_one",
    ]
    partitions = IsolatedGateEvaluator._partition_test_targets(nodeids, max_partitions=3)
    flattened = [target for partition in partitions for target in partition]
    assert sorted(flattened) == [
        "tests/test_a.py",
        "tests/test_b.py",
        "tests/test_c.py",
        "tests/test_d.py",
    ]
    assert len(flattened) == len(set(flattened))


def test_partition_targets_are_deterministic_and_count_balanced() -> None:
    nodeids = [*(f"test_heavy.py::test_{index}" for index in range(9))]
    nodeids += [*(f"test_medium.py::test_{index}" for index in range(5))]
    nodeids += ["test_small_a.py::test_one", "test_small_b.py::test_one"]
    first = IsolatedGateEvaluator._partition_test_targets(nodeids, max_partitions=2)
    second = IsolatedGateEvaluator._partition_test_targets(list(reversed(nodeids)), max_partitions=2)
    assert first == second
    counts = {nodeid.split("::", 1)[0]: 0 for nodeid in nodeids}
    for nodeid in nodeids:
        counts[nodeid.split("::", 1)[0]] += 1
    loads = []
    for partition in first:
        loads.append(sum(counts[target.removeprefix("tests/")] for target in partition))
    assert max(loads) - min(loads) <= 3


def test_merge_junit_files_preserves_complete_counts(tmp_path: Path) -> None:
    first = tmp_path / "first.xml"
    second = tmp_path / "second.xml"
    output = tmp_path / "aggregate.xml"
    _write_junit(first, tests=3)
    _write_junit(second, tests=2, skipped=1)
    counts = IsolatedGateEvaluator._merge_junit_files([first, second], output)
    assert counts == (5, 0, 0, 1)
    assert IsolatedGateEvaluator._junit_counts(output) == counts


def test_explicit_file_isolation_can_isolate_each_test_file() -> None:
    nodeids = [
        "test_a.py::test_one",
        "test_b.py::test_one",
        "test_c.py::test_one",
    ]
    partitions = IsolatedGateEvaluator._partition_test_targets(nodeids, max_partitions=128)
    assert len(partitions) == 3
    assert all(len(partition) == 1 for partition in partitions)


def test_file_backed_runner_does_not_wait_for_inherited_output_pipe(tmp_path: Path) -> None:
    import os
    import sys
    import time

    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    command = [
        sys.executable,
        "-c",
        "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(1.5)']); print('parent-finished')",
    ]
    started = time.monotonic()
    status, _, error, returncode = IsolatedGateEvaluator._run_to_files(
        command,
        cwd=tmp_path,
        timeout_seconds=5,
        env=os.environ.copy(),
        stdout_path=stdout,
        stderr_path=stderr,
    )
    assert status.value == "PASS"
    assert returncode == 0
    assert error == ""
    assert time.monotonic() - started < 1.0
    assert "parent-finished" in stdout.read_text(encoding="utf-8")



def test_collected_nodeids_are_canonicalized_against_repository_root(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_local.py").write_text("", encoding="utf-8")
    text = "test_local.py::test_one\nexamples/generated/test_contract.py::test_two\n"
    assert IsolatedGateEvaluator._nodeids(text, candidate_root=tmp_path) == [
        "examples/generated/test_contract.py::test_two",
        "tests/test_local.py::test_one",
    ]



def test_empty_explicit_file_nodeid_is_resolved_from_collect_target(tmp_path: Path) -> None:
    external = tmp_path / "examples/generated/test_contract.py"
    external.parent.mkdir(parents=True)
    external.write_text("", encoding="utf-8")
    nodeids = IsolatedGateEvaluator._nodeids(
        "::test_contract\n",
        candidate_root=tmp_path,
        collect_targets=["examples/generated/test_contract.py", "tests"],
    )
    assert nodeids == ["examples/generated/test_contract.py::test_contract"]

def test_collect_targets_include_frozen_tests_outside_tests_directory() -> None:
    targets = IsolatedGateEvaluator._collect_targets([
        "tests/test_a.py::test_one",
        "examples/generated/test_contract.py::test_two",
    ])
    assert targets == ["examples/generated/test_contract.py", "tests"]


def test_partition_targets_preserve_explicit_repository_paths() -> None:
    nodeids = [
        "tests/test_a.py::test_one",
        "examples/generated/test_contract.py::test_two",
    ]
    flattened = [item for part in IsolatedGateEvaluator._partition_test_targets(nodeids, max_partitions=2) for item in part]
    assert sorted(flattened) == ["examples/generated/test_contract.py", "tests/test_a.py"]

def test_formal_evaluator_executes_unified_runtime_external_gate(tmp_path: Path) -> None:
    import json
    import os

    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(root / "src")
    check = IsolatedGateEvaluator._evaluate_unified_runtime_external_gate(
        candidate_root=root,
        output_dir=tmp_path,
        env=env,
        timeout_seconds=30,
    )
    assert check.status.value == "PASS"
    assert check.criterion_id == "UNIFIED_RUNTIME_EXTERNAL_GATE"
    payload = json.loads((tmp_path / "unified_runtime_external_gate.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert "unified_runtime_external_gate.json" in check.evidence_paths


def test_trusted_single_fault_external_gate_executes_frozen_program(tmp_path: Path) -> None:
    from astrograph.governance.evaluator import IsolatedGateEvaluator

    root = tmp_path / "candidate"
    evaluator = root / "governance/evaluators/trusted_single_fault_gate.py"
    evaluation = root / "governance/evidence/0.7.1/trusted_single_fault_evaluation.json"
    evaluator.parent.mkdir(parents=True)
    evaluation.parent.mkdir(parents=True)
    source_evaluator = Path(__file__).parents[1] / "governance/evaluators/trusted_single_fault_gate.py"
    evaluator.write_text(source_evaluator.read_text(encoding="utf-8"), encoding="utf-8")
    payload = {
        "campaign_status": "PASS", "acceptance_valid_count": 48,
        "acceptance_valid_pair_count": 24, "public_trace_leakage_count": 0,
        "exact_diagnosis_count": 24, "false_diagnosis_count": 0,
        "holdout_case_count": 6, "holdout_exact_count": 6, "macro_f1": 1.0,
        "subsystem_exact_count": 24, "component_exact_count": 24,
        "onset_max_error_s": 0.0, "severity_max_error": 0.02,
        "label_or_truth_use_count": 0, "deterministic_replay_mismatch_count": 0,
        "case_id_permutation_mismatch_count": 0,
    }
    evaluation.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()
    check = IsolatedGateEvaluator._evaluate_trusted_single_fault_external_gate(
        candidate_root=root, output_dir=output, env=os.environ.copy(), timeout_seconds=30
    )
    assert check.status.value == "PASS"
    assert (output / "trusted_single_fault_external_gate.json").is_file()



def test_expanded_single_fault_external_gate_executes_frozen_program(tmp_path: Path) -> None:
    import shutil

    external_paths = {
        "ASTROGRAPH_4C_TRUTH": os.environ.get("ASTROGRAPH_4C_TRUTH", ""),
        "ASTROGRAPH_4C_FORMAL_PUBLIC": os.environ.get("ASTROGRAPH_4C_FORMAL_PUBLIC", ""),
        "ASTROGRAPH_4C_RESEARCH_PUBLIC": os.environ.get("ASTROGRAPH_4C_RESEARCH_PUBLIC", ""),
    }
    if not all(value and Path(value).exists() for value in external_paths.values()):
        pytest.skip("legacy 4C external fixture is not configured; it is not an AstroGraph G2 data source")

    source_root = Path(__file__).resolve().parents[1]
    root = tmp_path / "candidate"
    for relative in (
        "governance/evaluators/expanded_single_fault_gate.py",
        "governance/evidence/0.7.1-extension/expanded_single_fault_dataset_identity.json",
        "scripts/legacy/evaluate_expanded_single_fault.py",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((source_root / relative).read_text(encoding="utf-8"), encoding="utf-8")
    shutil.copytree(source_root / "src", root / "src")
    output = tmp_path / "out"
    output.mkdir()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(external_paths)
    check = IsolatedGateEvaluator._evaluate_expanded_single_fault_external_gate(
        candidate_root=root,
        output_dir=output,
        env=env,
        timeout_seconds=120,
    )
    assert check.status.value == "PASS"
    payload = json.loads((output / "expanded_single_fault_external_gate.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert all(payload["checks"].values())


def test_expanded_gate_rejects_truth_inside_candidate(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    evaluator = root / "governance/evaluators/expanded_single_fault_gate.py"
    identity = root / "governance/evidence/0.7.1-extension/expanded_single_fault_dataset_identity.json"
    evaluator.parent.mkdir(parents=True)
    identity.parent.mkdir(parents=True)
    evaluator.write_text("print('not executed')", encoding="utf-8")
    identity.write_text("{}", encoding="utf-8")
    truth = root / "truth.json"
    formal = root / "formal"
    research = root / "research"
    truth.write_text("{}", encoding="utf-8")
    formal.mkdir()
    research.mkdir()
    env = os.environ.copy()
    env["ASTROGRAPH_4C_TRUTH"] = str(truth)
    env["ASTROGRAPH_4C_FORMAL_PUBLIC"] = str(formal)
    env["ASTROGRAPH_4C_RESEARCH_PUBLIC"] = str(research)
    output = tmp_path / "out"
    output.mkdir()
    check = IsolatedGateEvaluator._evaluate_expanded_single_fault_external_gate(
        candidate_root=root, output_dir=output, env=env, timeout_seconds=30
    )
    assert check.status.value == "FAIL"
    assert check.error_code == "EXPANDED_SINGLE_FAULT_EXTERNAL_GATE_INPUT_MISSING"


def test_sqlite_multiprocessing_nodeids_are_isolated_without_skipping() -> None:
    nodeids = [
        "tests/test_a.py::test_one",
        "tests/test_sqlite_idempotency.py::test_one",
        "tests/test_sqlite_idempotency.py::test_two",
        "tests/test_b.py::test_one",
    ]
    partitions = IsolatedGateEvaluator._partition_test_targets(nodeids, max_partitions=2)
    isolated = [part for part in partitions if part and part[0].startswith("tests/test_sqlite_idempotency.py::")]
    assert sorted(item[0] for item in isolated) == sorted(nodeids[1:3])
    flattened = [target for part in partitions for target in part]
    assert "tests/test_a.py" in flattened
    assert "tests/test_b.py" in flattened
