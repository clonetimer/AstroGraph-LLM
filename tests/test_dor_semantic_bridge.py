from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from astrograph.diagnosis import DiagnosisPolicyGate
from astrograph.integrations.dor_semantic import (
    DORSemanticAuditRequest,
    DORSemanticAuditService,
    DORSemanticAuditStatus,
    semantic_release_blockers,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "configs/datasets/adcs_v5_2_2/dor_semantic_package.json"
EVIDENCE = ROOT / "configs/datasets/adcs_v5_2_2/runtime_evidence.json"
DOR = Path(os.environ.get("ASTROGRAPH_DOR_EXECUTABLE", "__optional_dor_not_found__"))
DATASET = Path(os.environ.get(
    "ASTROGRAPH_ADCS_V522_DATA_ROOT",
    str(ROOT / "data" / "external" / "whole_spacecraft_adcs_paired_dataset_v5_2_2"),
))
requires_external_dataset = pytest.mark.skipif(
    not (DATASET / "sample_index.csv").is_file(),
    reason="optional ADCS v5.2.2 external dataset is not configured",
)


def _bundle(tmp_path: Path):
    from astrograph.evidence import ConfiguredRuntimeEvidenceBuilder
    from astrograph.simulation import SimulationRunCompleted

    dataset = DATASET
    rows = (dataset / "sample_index.csv").read_text(encoding="utf-8").splitlines()
    import csv

    parsed = list(csv.DictReader(rows))
    sample = next(row for row in parsed if row["diagnostic_class"] == "adcs_rw_jamming" and row["sample_index"] == "4")
    nominal = next(
        row
        for row in parsed
        if row["diagnostic_class"] == "nominal"
        and row["trajectory_pair_id"] == sample["trajectory_pair_id"]
    )
    run_dir = tmp_path / "run"
    nominal_dir = tmp_path / "nominal"
    run_dir.mkdir()
    nominal_dir.mkdir()
    import shutil

    shutil.copy2(dataset / sample["telemetry_file"], run_dir / "telemetry.csv")
    shutil.copy2(dataset / nominal["telemetry_file"], nominal_dir / "telemetry.csv")
    event = SimulationRunCompleted(
        event_id="DOR-BRIDGE-TEST",
        run_id="DOR-BRIDGE-TEST",
        run_dir=str(run_dir),
        nominal_run_dir=str(nominal_dir),
    )
    return ConfiguredRuntimeEvidenceBuilder(EVIDENCE).build(event)[0]


@requires_external_dataset
def test_dor_bridge_compiles_without_labels_or_root_cause(tmp_path: Path):
    bundle = _bundle(tmp_path)
    package, _ = DORSemanticAuditService.load_package(PACKAGE)
    ontology, mechanism_map, evidence_csv = DORSemanticAuditService.compile_inputs(bundle, package)
    serialized = json.dumps({"ontology": ontology, "mechanism_map": mechanism_map}) + evidence_csv
    assert "label.fault_active" not in serialized
    assert "diagnostic_class" not in evidence_csv
    assert len(ontology["faults"]) == 4
    assert len(mechanism_map["bindings"]) == len(bundle.evidence_definitions)
    assert "root_cause" not in serialized


@pytest.mark.skipif(not DOR.is_file(), reason="optional external DOR runtime is not installed")
@requires_external_dataset
def test_real_dor_localization_supports_rw_jam_and_preserves_claim_boundary(tmp_path: Path):
    audit = DORSemanticAuditService().audit(
        _bundle(tmp_path),
        DORSemanticAuditRequest(
            semantic_package_path=str(PACKAGE),
            executable_path=str(DOR),
        ),
    )
    assert audit.status == DORSemanticAuditStatus.SUPPORTED
    assert audit.supported_fault_ids == ["ADCS_RW_JAM"]
    assert audit.eligible_for_formal_fault_release
    assert audit.safety["root_cause_claimed"] is False
    assert audit.safety["propagation_inferred"] is False
    assert audit.dor_stack_info is not None
    assert audit.dor_stack_info["DOR"]["version"]
    assert audit.executable_sha256 is not None
    assert audit.executable == str(DOR.resolve())
    assert audit.runtime is not None
    assert audit.runtime.distribution_source_sha256 is not None
    assert semantic_release_blockers(audit, "ADCS_RW_JAM") == []
    assert semantic_release_blockers(audit, "ADCS_STAR_TRACKER_DROPOUT")


@requires_external_dataset
def test_missing_dor_fails_closed_without_crashing_workflow_contract(tmp_path: Path):
    audit = DORSemanticAuditService().audit(
        _bundle(tmp_path),
        DORSemanticAuditRequest(
            semantic_package_path=str(PACKAGE),
            executable_path=str(tmp_path / "missing-dor"),
        ),
    )
    assert audit.status == DORSemanticAuditStatus.ERROR
    assert not audit.eligible_for_formal_fault_release
    assert semantic_release_blockers(audit, "ADCS_RW_JAM") == ["DOR_SEMANTIC_AUDIT_ERROR"]


def test_policy_gate_has_distinct_semantic_blocker_channel():
    import inspect

    assert "semantic_gate_blockers" in inspect.signature(DiagnosisPolicyGate.decide).parameters
