"""Contract checks for the bundled synthetic acceptance files (no server needed)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


CASES = Path(__file__).resolve().parents[1] / "examples" / "platform_acceptance_cases_v1"


def test_case_inventory_and_expected_output_contract() -> None:
    dirs = sorted(p for p in CASES.glob("TC*") if p.is_dir())
    assert len(dirs) == 9
    assert len([p for p in dirs if not p.name.startswith("TC09")]) == 8
    for folder in dirs:
        expectation = json.loads((folder / "expected_output.json").read_text(encoding="utf-8"))
        assert expectation["case_id"] == folder.name
        if folder.name.startswith("TC09"):
            assert (folder / "graph_reference.md").is_file()
            continue
        manifest = json.loads((folder / "analysis_input.json").read_text(encoding="utf-8"))
        assert manifest["equipment_scope"] == expectation["equipment_scope"]
        uploaded = expectation["upload_files"]
        assert uploaded and len(uploaded) == len(set(uploaded))
        assert all((folder / name).is_file() for name in uploaded)
        declared = set(manifest["diagnosis_results"] + manifest["log_reports"] + manifest["references"])
        assert set(uploaded).issubset(declared)
        if expectation["preflight_can_start"]:
            assert set(uploaded) == declared
        else:
            # TC07 deliberately references a missing document in the manifest.
            assert declared - set(uploaded) == {"reference.md"}
            assert not (folder / "reference.md").exists()


def test_case_payload_sha256_integrity() -> None:
    checksums = (CASES / "CASES_SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    expected = {str(p.relative_to(CASES)).replace("\\", "/") for p in CASES.rglob("*") if p.is_file() and p.name != "CASES_SHA256SUMS.txt"}
    assert len(checksums) == len(expected)
    for line in checksums:
        digest, name = line.split("  ", 1)
        path = (CASES / name).resolve()
        assert path.is_relative_to(CASES.resolve())
        assert name in expected
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_material_test_script_has_project_local_defaults() -> None:
    program = (CASES / "validate_cases.py").read_text(encoding="utf-8")
    assert "Path(__file__).resolve().parents[2]" in program
    assert "'runs' / 'acceptance_cases'" in program
    assert (CASES.parents[1] / "run_acceptance_cases.ps1").is_file()
    assert (CASES.parents[1] / "run_platform_demo.ps1").is_file()
