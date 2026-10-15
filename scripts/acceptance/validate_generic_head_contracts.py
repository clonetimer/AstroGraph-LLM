#!/usr/bin/env python3
"""Validate the generic four-head interface without claiming calibrated diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from astrograph.architecture import load_boundary_policy, scan_architecture_boundaries
from astrograph.diagnosis.generic import (
    AnomalyHead,
    GenericHeadManifest,
    LocalizationHead,
    MechanismHead,
    ReasoningHead,
)


GENERIC_HEAD_SOURCES = (
    "src/astrograph/diagnosis/generic/__init__.py",
    "src/astrograph/diagnosis/generic/contracts.py",
    "src/astrograph/diagnosis/generic/anomaly.py",
    "src/astrograph/diagnosis/generic/calibration.py",
    "src/astrograph/diagnosis/generic/mechanism.py",
    "src/astrograph/diagnosis/generic/localization.py",
    "src/astrograph/diagnosis/generic/reasoning.py",
    "src/astrograph/diagnosis/generic/governor.py",
    "src/astrograph/diagnosis/generic/pipeline.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root: Path) -> dict[str, Any]:
    manifest_path = root / "configs/diagnosis/generic_heads.v1.json"
    manifest = GenericHeadManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    expected = [
        (AnomalyHead.head_id, AnomalyHead.head_version),
        (MechanismHead.head_id, MechanismHead.head_version),
        (LocalizationHead.head_id, LocalizationHead.head_version),
        (ReasoningHead.head_id, ReasoningHead.head_version),
    ]
    declared = [(item.head_id, item.head_version) for item in manifest.bindings]
    boundary = scan_architecture_boundaries(
        root,
        load_boundary_policy(root / "configs/architecture/module_boundaries.json"),
    )
    checks = {
        "four_head_sequence_matches_runtime": declared == expected,
        "generic_boundary_scan_pass": boundary.status == "PASS",
        "generic_boundary_specialization_hits_zero": boundary.generic_match_count == 0,
        "labels_used_false": manifest.labels_used is False,
        "truth_used_false": manifest.truth_used is False,
        "anomaly_head_calibrated": manifest.bindings[0].calibration_status.value
        == "CALIBRATED",
        "mechanism_calibration_disclosed_incomplete": (
            manifest.known_mechanism_classification_calibrated is False
            and manifest.bindings[1].formal_output_allowed is False
        ),
        "root_cause_calibration_disclosed_incomplete": (
            manifest.root_cause_localization_calibrated is False
            and manifest.bindings[2].formal_output_allowed is False
        ),
        "open_set_ood_disclosed_incomplete": manifest.open_set_ood_calibrated is False,
    }
    interface_pass = all(checks.values())
    return {
        "schema_version": "astrograph-generic-head-contract-evidence.v1",
        "pipeline_id": manifest.pipeline_id,
        "pipeline_version": manifest.pipeline_version,
        "formal_hidden_truth_used": False,
        "labels_used": False,
        "checks": checks,
        "generic_source_sha256": {
            relative: _sha256(root / relative) for relative in GENERIC_HEAD_SOURCES
        },
        "manifest_sha256": _sha256(manifest_path),
        "boundary_scanned_file_count": boundary.scanned_file_count,
        "boundary_generic_match_count": boundary.generic_match_count,
        "t310_interface_status": "PASS" if interface_pass else "FAIL",
        "t310_calibration_status": "INCOMPLETE",
        "t311_status": "INCOMPLETE",
        "claim_boundary": (
            "Four-head contracts, ordering, no-label receipts and fail-closed disclosure only. "
            "Known-mechanism calibration, root-cause localization calibration, open-set OOD, "
            "independent-source validation, T-310 completion, T-311, G3 and G4 are not established."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = validate(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0 if evidence["t310_interface_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
