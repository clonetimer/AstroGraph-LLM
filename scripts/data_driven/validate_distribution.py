#!/usr/bin/env python3
"""Validate the portable AstroGraph source distribution and released model artifacts."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import tomllib
from pathlib import Path

# scripts/ 顶层含 _bootstrap.py；本脚本位于 scripts/data_driven/ 子目录，
# 运行时 sys.path[0] 是子目录，需先把 scripts/ 顶层加入 sys.path。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

ROOT = bootstrap()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def distribution_files(root: Path) -> list[Path]:
    if (root / ".git").exists():
        result = subprocess.run(
            [
                "git", "-C", str(root), "ls-files", "--cached", "--others",
                "--exclude-standard", "-z",
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return [
                root / item.decode("utf-8", errors="strict")
                for item in result.stdout.split(b"\0")
                if item
            ]
    return list(root.rglob("*"))


def add_check(checks: list[dict[str, object]], errors: list[dict[str, str]], *, name: str, kind: str, value: str, passed: bool, code: str) -> None:
    checks.append({"name": name, "kind": kind, "value": value, "pass": passed})
    if not passed:
        errors.append({"code": code, "path": value})


def validate(root: Path) -> dict[str, object]:
    release_path = root / "release.json"
    errors: list[dict[str, str]] = []
    checks: list[dict[str, object]] = []
    if not release_path.is_file():
        return {"status": "FAIL", "errors": [{"code": "RELEASE_MANIFEST_MISSING", "path": str(release_path)}]}

    release = json.loads(release_path.read_text(encoding="utf-8"))
    version = str(release.get("version", ""))
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    package_version = str(pyproject["project"]["version"])
    add_check(checks, errors, name="release_version", kind="metadata", value=f"release={version};package={package_version}", passed=version == package_version, code="RELEASE_VERSION_MISMATCH")
    try:
        runtime_init = root / "src" / "astrograph" / "__init__.py"
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
    except Exception:
        runtime_version = ""
    add_check(
        checks,
        errors,
        name="runtime_version",
        kind="metadata",
        value=f"release={version};runtime={runtime_version}",
        passed=version == runtime_version,
        code="RUNTIME_VERSION_MISMATCH",
    )

    for name, value in release.get("production_paths", {}).items():
        if value.startswith("astrograph."):
            passed = module_exists(value)
            kind = "module"
        else:
            passed = (root / value).is_file()
            kind = "file"
        add_check(checks, errors, name=name, kind=kind, value=value, passed=passed, code="PRODUCTION_PATH_MISSING")

    is_stable_release = str(release.get("status", "")).startswith("STABLE_SOFTWARE_RELEASE")
    claims_name = "stable_release.json" if is_stable_release else "engineering_candidate.json"
    claims_path = root / "configs" / "release_claims" / claims_name
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    release_suffix = "stable" if is_stable_release else "engineering-candidate"
    expected_release_id = f"AstroGraph-v{version}-{release_suffix}"
    add_check(checks, errors, name="release_claim_id", kind="metadata", value=str(claims.get("release_id")), passed=claims.get("release_id") == expected_release_id, code="RELEASE_CLAIM_VERSION_MISMATCH")
    for claim in claims.get("claims", []):
        for source in claim.get("source_artifacts", []):
            add_check(checks, errors, name=str(claim.get("claim_id")), kind="claim_source", value=source, passed=(root / source).is_file(), code="CLAIM_SOURCE_MISSING")

    for manifest_path in sorted((root / "reference" / "models").glob("*/model_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_path = manifest_path.parent / manifest["model_file"]
        actual = sha256(model_path) if model_path.is_file() else None
        expected = manifest.get("model_sha256")
        add_check(checks, errors, name=manifest_path.parent.name, kind="model_sha256", value=str(model_path.relative_to(root)), passed=actual == expected, code="MODEL_HASH_MISMATCH")

    # `data/` is an explicitly governed in-scope source-snapshot directory in
    # the current frozen contract.  Runtime/result directories remain banned.
    forbidden_root_dirs = ["reports", "outputs", "results", "local_acceptance_data", "external_validation", "memory", "simulation", "history", "archive"]
    for name in forbidden_root_dirs:
        path = root / name
        if path.exists():
            errors.append({"code": "RUNTIME_OR_HISTORY_DIRECTORY_PRESENT", "path": name})
    # Local interpreter/test caches are ephemeral workspace state and are
    # excluded from both manifests below. Their presence must not invalidate a
    # source distribution assembled from the same working tree.

    bad_name = re.compile(r"(?:round\d+|phase\d+|claimfix|\(\d+\)|(?:^|[._-])(?:tmp|bak|copy|old)(?:[._-]|$))", re.IGNORECASE)
    historical_filename_exemptions: set[str] = set()
    # Apply source naming rules to the same distributable files as the manifests,
    # not Git-ignored human-review workbooks or other local runtime state.
    for path in distribution_files(root):
        relative = path.relative_to(root).as_posix()
        if "worktrees" in path.relative_to(root).parts:
            continue
        if relative in historical_filename_exemptions:
            continue
        if path.is_file() and bad_name.search(path.name):
            errors.append({"code": "NON_SEMANTIC_FILENAME", "path": relative})

    semantic_rules = root / "configs" / "reaction_wheel_diagnosis_rules.json"
    obsolete_rules = root / "configs" / "d3_rw_diagnosis_rules.json"
    add_check(checks, errors, name="reaction_wheel_rules", kind="semantic_path", value=str(semantic_rules.relative_to(root)), passed=semantic_rules.is_file() and not obsolete_rules.exists(), code="REACTION_WHEEL_RULES_RENAME_INCOMPLETE")

    registry_path = root / "benchmarks" / "knowledge_graph" / "core" / "source_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for item in registry.get("sources", []):
        source_ref = str(item["source_ref"])
        add_check(checks, errors, name=str(item["source_id"]), kind="source_ref", value=source_ref, passed=(root / source_ref).is_file(), code="SOURCE_REFERENCE_MISSING")

    anchors_path = root / "benchmarks" / "knowledge_graph" / "core" / "source_anchors.json"
    anchors = json.loads(anchors_path.read_text(encoding="utf-8"))
    for source_ref, expected_hash in anchors.get("documents", {}).items():
        source_path = root / source_ref
        passed = source_path.is_file() and sha256(source_path) == expected_hash
        add_check(checks, errors, name="source_anchor", kind="sha256", value=source_ref, passed=passed, code="SOURCE_ANCHOR_INVALID")

    contract_path = root / "governance" / "contracts" / f"{version}.yaml"
    contract_checksum_path = root / "governance" / "contracts" / f"{version}.sha256"
    contract = {}
    if contract_path.is_file() and contract_checksum_path.is_file():
        expected_contract_hash = contract_checksum_path.read_text(encoding="utf-8").split()[0]
        actual_contract_hash = sha256(contract_path)
        add_check(
            checks,
            errors,
            name="improvement_contract_hash",
            kind="governance",
            value=str(contract_path.relative_to(root)),
            passed=expected_contract_hash == actual_contract_hash,
            code="IMPROVEMENT_CONTRACT_HASH_MISMATCH",
        )
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
        except Exception:
            errors.append({"code": "IMPROVEMENT_CONTRACT_INVALID", "path": str(contract_path.relative_to(root))})
        else:
            add_check(
                checks,
                errors,
                name="improvement_contract_version",
                kind="governance",
                value=str(contract.get("target_version", "")),
                passed=contract.get("target_version") == version and contract.get("status") == "FROZEN",
                code="IMPROVEMENT_CONTRACT_VERSION_MISMATCH",
            )
    else:
        errors.append({"code": "IMPROVEMENT_CONTRACT_MISSING", "path": f"governance/contracts/{version}.yaml"})

    governance_required = [
        "scripts/governance/run_isolated_gate.py",
        "scripts/governance/export_schemas.py",
        f"governance/change_requests/{version}.json",
        f"governance/task_boards/{version}.json",
        f"governance/rollback/{version}.json",
        "governance/failure_memory/permanent.jsonl",
        "governance/passports/materials.jsonl",
    ]
    for relative in governance_required:
        add_check(
            checks,
            errors,
            name="governance_asset",
            kind="governance",
            value=relative,
            passed=(root / relative).is_file(),
            code="GOVERNANCE_ASSET_MISSING",
        )
    schema_files = sorted((root / "governance" / "schemas").glob("*.schema.json"))
    add_check(
        checks,
        errors,
        name="governance_schemas",
        kind="governance",
        value=str(len(schema_files)),
        passed=len(schema_files) == 9,
        code="GOVERNANCE_SCHEMA_SET_INCOMPLETE",
    )
    if contract:
        baseline_path_value = str(contract.get("baseline_manifest_path", ""))
        baseline_path = root / baseline_path_value
        expected_baseline_hash = str(contract.get("baseline_manifest_sha256", ""))
        add_check(
            checks,
            errors,
            name="baseline_manifest",
            kind="governance",
            value=baseline_path_value,
            passed=baseline_path.is_file() and sha256(baseline_path) == expected_baseline_hash,
            code="BASELINE_MANIFEST_INVALID",
        )
        inventory_value = str(contract.get("baseline_test_inventory_path", ""))
        inventory_path = root / inventory_value
        inventory_hash = str(contract.get("baseline_test_inventory_sha256", ""))
        if inventory_value:
            add_check(
                checks,
                errors,
                name="baseline_test_inventory",
                kind="governance",
                value=inventory_value,
                passed=inventory_path.is_file() and sha256(inventory_path) == inventory_hash,
                code="BASELINE_TEST_INVENTORY_INVALID",
            )

    filelist_path = root / "FILELIST.txt"
    checksum_path = root / "SHA256SUMS.txt"
    manifest_path = root / "SOURCE_MANIFEST.json"
    if filelist_path.is_file() and checksum_path.is_file() and manifest_path.is_file():
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        content_excluded = {"FILELIST.txt", "SHA256SUMS.txt", "SOURCE_MANIFEST.json"}
        inventory = distribution_files(root)
        actual_content = sorted(
            str(p.relative_to(root)).replace("\\", "/")
            for p in inventory
            if p.is_file()
            and not {".git", ".claude", ".pytest_cache", "__pycache__", ".tmp_tests", "worktrees", ".venv", "astrograph_core.egg-info"}.intersection(p.relative_to(root).parts)
            and p.suffix not in {".pyc", ".pyo"}
            and not str(p.relative_to(root)).replace("\\", "/").startswith(("scripts/integration/out/", "runs/"))
            and str(p.relative_to(root)).replace("\\", "/") not in content_excluded
        )
        listed = [line for line in filelist_path.read_text(encoding="utf-8").splitlines() if line]
        add_check(checks, errors, name="filelist", kind="manifest", value="FILELIST.txt", passed=listed == actual_content and source_manifest.get("content_file_count") == len(actual_content), code="FILELIST_OR_COUNT_MISMATCH")
        checksum_rows = [line.split("  ", 1) for line in checksum_path.read_text(encoding="utf-8").splitlines() if line]
        expected_checksum_files = sorted(
            str(p.relative_to(root)).replace("\\", "/")
            for p in inventory
            if p.is_file()
            and not {".git", ".claude", ".pytest_cache", "__pycache__", ".tmp_tests", "worktrees", ".venv", "astrograph_core.egg-info"}.intersection(p.relative_to(root).parts)
            and p.suffix not in {".pyc", ".pyo"}
            and not str(p.relative_to(root)).replace("\\", "/").startswith(("scripts/integration/out/", "runs/"))
            and p.name != "SHA256SUMS.txt"
        )
        checksum_names = sorted(row[1] for row in checksum_rows if len(row) == 2)
        hashes_ok = all(len(row) == 2 and (root / row[1]).is_file() and sha256(root / row[1]) == row[0] for row in checksum_rows)
        add_check(checks, errors, name="checksums", kind="manifest", value="SHA256SUMS.txt", passed=checksum_names == expected_checksum_files and hashes_ok, code="CHECKSUM_MANIFEST_INVALID")

    return {
        "schema_version": "astrograph-distribution-audit.v2",
        "version": version,
        "checks": checks,
        "error_count": len(errors),
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
        "claim_boundary": "Validates distribution integrity and released artifacts; it does not establish flight or safety qualification.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.root.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
