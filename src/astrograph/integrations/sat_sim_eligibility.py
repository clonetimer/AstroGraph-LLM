"""Truth-blind engine eligibility audit for Sat-Agent public campaigns."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from astrograph.diagnosis.sat_sim_catalog import load_catalog_artifact
from astrograph.diagnosis.sat_sim_open_set import load_open_set_artifact
from astrograph.diagnosis.telemetry_contract import infer_backend

from .sat_sim_public_case import _load_public_case, _sha256_file


SCHEMA_VERSION = "astrograph-sat-sim-eligibility-matrix.v1"
ROUTES = ("data_quality", "normal", "single_fault", "multifault", "unknown_fault", "drift")


def _model_contract(artifact: Mapping[str, Any], backend: str, fields: set[str]) -> dict[str, Any]:
    backend_model = artifact.get("backend_models", {}).get(backend)
    if not isinstance(backend_model, Mapping):
        return {"eligible": False, "reason": "BACKEND_NOT_IN_MODEL", "observed_field_ratio": 0.0}
    expected = {str(name).rsplit("__", 1)[0] for name in backend_model.get("feature_names", ())}
    ratio = len(expected & fields) / len(expected) if expected else 0.0
    threshold = float(backend_model.get("minimum_observed_field_ratio", 1.0))
    return {
        "eligible": ratio >= threshold,
        "reason": "ELIGIBLE" if ratio >= threshold else "MODEL_FIELD_COVERAGE_BELOW_CONTRACT",
        "observed_field_ratio": ratio,
        "minimum_observed_field_ratio": threshold,
        "expected_field_count": len(expected),
        "observed_expected_field_count": len(expected & fields),
    }


def build_eligibility_matrix(
    public_root: str | Path,
    *,
    catalog_model_path: str | Path,
    open_set_model_path: str | Path,
) -> dict[str, Any]:
    public = Path(public_root).resolve()
    catalog_source = Path(catalog_model_path).resolve()
    open_set_source = Path(open_set_model_path).resolve()
    catalog = load_catalog_artifact(catalog_source)
    open_set = load_open_set_artifact(open_set_source)
    open_base = open_set.get("base_artifact", {})
    calibration = open_set.get("backend_calibration", {})

    rows: list[dict[str, Any]] = []
    route_counts: Counter[str] = Counter()
    blocker_counts: Counter[str] = Counter()
    for case_root in sorted(path for path in public.iterdir() if path.is_dir()):
        manifest, _ = _load_public_case(case_root)
        fields = set(str(item) for item in manifest["telemetry"]["fields"])
        backend = infer_backend(fields)
        catalog_contract = _model_contract(catalog, backend, fields)
        open_contract = _model_contract(open_base, backend, fields)
        if backend not in calibration:
            open_contract = {**open_contract, "eligible": False, "reason": "BACKEND_NOT_CALIBRATED"}
        physical_eligible = backend == "composite"
        routes = {
            "data_quality": True,
            "normal": bool(catalog_contract["eligible"] and open_contract["eligible"]),
            "single_fault": bool(catalog_contract["eligible"] and open_contract["eligible"]),
            "multifault": physical_eligible,
            "unknown_fault": bool(open_contract["eligible"]),
            "drift": False,
        }
        for route, eligible in routes.items():
            if eligible:
                route_counts[route] += 1
        blockers = []
        if not catalog_contract["eligible"]:
            blockers.append(str(catalog_contract["reason"]))
        if not open_contract["eligible"]:
            blockers.append(str(open_contract["reason"]))
        if not physical_eligible:
            blockers.append("UNIFIED_SINGLE_TRACE_PHYSICAL_ENGINE_MISSING")
        blockers.append("DRIFT_ENGINE_MISSING")
        for blocker in set(blockers):
            blocker_counts[blocker] += 1
        field_list = list(manifest["telemetry"]["fields"])
        rows.append({
            "sample_id": _sha256_file(case_root / "manifest.json"),
            "field_schema_sha256": hashlib.sha256(
                json.dumps(field_list, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "backend": backend,
            "field_count": len(fields),
            "catalog_model": catalog_contract,
            "open_set_model": open_contract,
            "physical_single_trace": {
                "eligible": physical_eligible,
                "reason": "ELIGIBLE" if physical_eligible else "BACKEND_NOT_SUPPORTED",
            },
            "routes": routes,
            "blockers": sorted(set(blockers)),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if rows and route_counts.get("data_quality", 0) == len(rows) else "FAIL",
        "data_source": "sat_agent_platform_simulation_public_packages",
        "truth_root_accessed": False,
        "case_id_used_for_routing": False,
        "case_count": len(rows),
        "catalog_model_source_sha256": _sha256_file(catalog_source),
        "open_set_model_source_sha256": _sha256_file(open_set_source),
        "route_eligible_case_counts": {route: route_counts.get(route, 0) for route in ROUTES},
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "all_six_routes_covered": all(route_counts.get(route, 0) > 0 for route in ROUTES),
        "cases": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--catalog-model", type=Path, required=True)
    parser.add_argument("--open-set-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_eligibility_matrix(
        args.public_root,
        catalog_model_path=args.catalog_model,
        open_set_model_path=args.open_set_model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "case_count": report["case_count"],
        "all_six_routes_covered": report["all_six_routes_covered"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA_VERSION", "build_eligibility_matrix"]
