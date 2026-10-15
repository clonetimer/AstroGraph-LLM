from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import joblib

from astrograph.integrations.sat_sim_eligibility import build_eligibility_matrix


def _release(tmp_path: Path, name: str, payload: dict, manifest_schema: str) -> Path:
    root = tmp_path / name
    root.mkdir()
    artifact = root / "model.joblib"
    joblib.dump(payload, artifact)
    manifest = root / "model_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": manifest_schema,
        "model_file": artifact.name,
        "model_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    return manifest


def test_eligibility_matrix_routes_only_from_public_field_contract(tmp_path: Path) -> None:
    public = tmp_path / "public"
    case = public / "answer-bearing-case-name"
    case.mkdir(parents=True)
    telemetry = case / "telemetry.csv"
    fields = ["time_s", "eps.signal", "eps.context", "eps.margin"]
    with telemetry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {"time_s": 0, "eps.signal": 1.0, "eps.context": 2.0, "eps.margin": 3.0},
            {"time_s": 1, "eps.signal": 0.9, "eps.context": 2.0, "eps.margin": 3.0},
        ])
    manifest = {
        "schema_version": "sat-sim.astrograph-public-case.v1",
        "case_id": "must-not-drive-routing",
        "telemetry": {
            "file": telemetry.name,
            "sha256": hashlib.sha256(telemetry.read_bytes()).hexdigest(),
            "rows": 2,
            "fields": fields,
        },
    }
    (case / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    base = {
        "schema_version": "astrograph-sat-sim-catalog-model.v1",
        "backend_models": {
            "unified": {
                "feature_names": ["eps.signal__mean"],
                "minimum_observed_field_ratio": 1.0,
            }
        },
    }
    catalog = _release(tmp_path, "catalog", base, "astrograph-sat-sim-catalog-release.v1")
    open_set = _release(
        tmp_path,
        "open",
        {
            "schema_version": "astrograph-sat-sim-open-set-model.v1",
            "base_artifact": base,
            "backend_calibration": {"unified": {}},
        },
        "astrograph-sat-sim-open-set-model-manifest.v1",
    )

    report = build_eligibility_matrix(public, catalog_model_path=catalog, open_set_model_path=open_set)
    assert report["truth_root_accessed"] is False
    assert report["case_id_used_for_routing"] is False
    assert report["route_eligible_case_counts"]["normal"] == 1
    assert report["route_eligible_case_counts"]["unknown_fault"] == 1
    assert report["route_eligible_case_counts"]["drift"] == 0
    assert "case_id" not in report["cases"][0]
