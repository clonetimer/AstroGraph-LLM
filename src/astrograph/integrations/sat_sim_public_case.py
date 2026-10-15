"""Truth-blind ingestion and diagnosis of Sat-Agent public case packages."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from astrograph.diagnosis.unified import UnifiedDiagnosisService
from astrograph.diagnosis.unified_models import UnifiedDiagnosisStatus


PUBLIC_CASE_SCHEMA_VERSION = "sat-sim.astrograph-public-case.v1"
SIMULATION_HANDOFF_SCHEMA_VERSION = "simulation_handoff.v2"
CLOSURE_SCHEMA_VERSION = "astrograph-public-case-closure.v1"
CAMPAIGN_CLOSURE_SCHEMA_VERSION = "astrograph-public-campaign-closure.v1"
PROHIBITED_FIELDS = {"task_id", "case_id", "scenario_id", "object_id", "capability_id", "target_level", "target_name", "mode", "classification", "location", "causal_chain"}
PROHIBITED_TOKENS = ("label.", "truth.", "event.", "object.", "modifier.", "fault", "health_state", "runtime_truth", "file_path")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_public_case(public_case: Path) -> tuple[dict[str, Any], Path]:
    root = public_case.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in {PUBLIC_CASE_SCHEMA_VERSION, SIMULATION_HANDOFF_SCHEMA_VERSION}:
        raise ValueError("unsupported Sat-Agent public case schema")
    if manifest.get("schema_version") == SIMULATION_HANDOFF_SCHEMA_VERSION:
        provenance = manifest.get("provenance")
        qualification = manifest.get("qualification")
        if not isinstance(provenance, Mapping) or not isinstance(qualification, Mapping):
            raise ValueError("simulation_handoff.v2 provenance and qualification are required")
        required_hashes = {
            "source_result_sha256", "source_nominal_sha256", "source_telemetry_sha256",
            "config_sha256", "seed_sha256",
        }
        invalid_hashes = sorted(
            name for name in required_hashes
            if len(str(provenance.get(name) or "")) != 64
            or any(char not in "0123456789abcdef" for char in str(provenance.get(name) or "").lower())
        )
        if invalid_hashes:
            raise ValueError(f"simulation handoff provenance hashes are invalid: {invalid_hashes}")
        if qualification.get("source_validation") != "PASS" or qualification.get("physics_gate") != "PASS":
            raise ValueError("simulation handoff qualification gates are not PASS")
    telemetry_meta = manifest.get("telemetry")
    if not isinstance(telemetry_meta, Mapping):
        raise ValueError("public case telemetry metadata is missing")
    relative = Path(str(telemetry_meta.get("file") or ""))
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name != str(relative):
        raise ValueError("public telemetry path must be one file inside the public case")
    telemetry = (root / relative).resolve()
    if telemetry.parent != root or not telemetry.is_file():
        raise ValueError("public telemetry file is missing or escapes the case directory")
    if _sha256_file(telemetry) != telemetry_meta.get("sha256"):
        raise ValueError("public telemetry hash mismatch")
    with telemetry.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        fields = next(reader, [])
        rows = list(reader)
        row_count = len(rows)
    prohibited = sorted(field for field in fields if field.lower() in PROHIBITED_FIELDS or any(token in field.lower() for token in PROHIBITED_TOKENS))
    if prohibited:
        raise ValueError(f"public telemetry contains answer-proxy fields: {prohibited}")
    proxy_values = sorted({
        value
        for row in rows
        for value in row
        if any(token in value.strip().lower() for token in PROHIBITED_TOKENS)
    })
    if proxy_values:
        raise ValueError(f"public telemetry contains answer-proxy values: {proxy_values[:5]}")
    if fields != list(telemetry_meta.get("fields") or ()) or row_count != telemetry_meta.get("rows"):
        raise ValueError("public telemetry shape does not match manifest")
    if "time_s" not in fields:
        raise ValueError("public telemetry requires time_s")
    return manifest, telemetry


def _detection(status: UnifiedDiagnosisStatus) -> str:
    if status == UnifiedDiagnosisStatus.NORMAL:
        return "NORMAL"
    if status in {UnifiedDiagnosisStatus.DIAGNOSED, UnifiedDiagnosisStatus.AMBIGUOUS, UnifiedDiagnosisStatus.CONFLICTING_EVIDENCE}:
        return "FAULT"
    if status == UnifiedDiagnosisStatus.UNKNOWN_FAULT:
        return "UNKNOWN_FAULT"
    if status == UnifiedDiagnosisStatus.DATA_ERROR:
        return "DATA_ERROR"
    return "UNASSESSED"


def diagnose_public_case(
    public_case: str | Path,
    *,
    output_root: str | Path,
    backend: str = "auto",
    catalog_model_path: str | Path | None = None,
    open_set_model_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(public_case).resolve()
    manifest, telemetry = _load_public_case(root)
    catalog_model = Path(catalog_model_path).resolve() if catalog_model_path is not None else None
    open_set_model = Path(open_set_model_path).resolve() if open_set_model_path is not None else None
    result = UnifiedDiagnosisService().diagnose_paths(
        case_id=str(manifest["case_id"]),
        telemetry_path=telemetry,
        backend=backend,
        catalog_model_path=catalog_model,
        open_set_model_path=open_set_model,
    )
    diagnosis_payload = result.model_dump(mode="json")
    if result.labels_used or result.truth_used:
        raise RuntimeError("unified diagnosis reported label or truth access")
    classifications = list(result.conclusion_effect_ids)
    closure = {
        "schema_version": CLOSURE_SCHEMA_VERSION, "case_id": str(manifest["case_id"]),
        "public_manifest_sha256": _sha256_file(root / "manifest.json"),
        "formal_diagnosis_publishable": False, "training_ready": False,
        "diagnosis": {
            "status": result.status.value, "detection": _detection(result.status),
            "location_candidates": [value for value in (result.primary_subsystem, result.primary_component) if value],
            "classification_candidates": classifications,
            "causal_chain": {"status": "NOT_EVALUATED" if not classifications else "CANDIDATE", "resolution_basis": result.resolution_basis},
            "evidence_ids": [item.evidence_id for item in result.diagnostic_evidence],
        },
        "diagnosis_provenance": {
            "schema_version": result.schema_version, "backend": result.backend, "input_mode": result.input_mode.value,
            "labels_used": result.labels_used, "truth_used": result.truth_used, "deterministic": result.deterministic,
            "diagnosis_sha256": _stable_hash(diagnosis_payload),
            "catalog_model_manifest_sha256": _sha256_file(catalog_model) if catalog_model is not None else None,
            "open_set_model_manifest_sha256": _sha256_file(open_set_model) if open_set_model is not None else None,
        },
    }
    output = Path(output_root).resolve()
    if output == root or output in root.parents or root in output.parents:
        raise ValueError("diagnosis output must be separate from the immutable public case")
    output.mkdir(parents=True, exist_ok=False)
    (output / "astrograph_diagnosis.json").write_text(json.dumps(diagnosis_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "closure.json").write_text(json.dumps(closure, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return closure


def diagnose_public_campaign(
    public_root: str | Path,
    *,
    output_root: str | Path,
    backend: str = "auto",
    expected_case_count: int | None = None,
    catalog_model_path: str | Path | None = None,
    open_set_model_path: str | Path | None = None,
) -> dict[str, Any]:
    public = Path(public_root).resolve()
    output = Path(output_root).resolve()
    if not public.is_dir():
        raise ValueError("public campaign root is missing")
    if output == public or output in public.parents or public in output.parents:
        raise ValueError("campaign diagnosis output must be separate from public input")
    cases = sorted(path for path in public.iterdir() if path.is_dir() and (path / "manifest.json").is_file())
    if expected_case_count is not None and len(cases) != expected_case_count:
        raise ValueError(f"expected {expected_case_count} public cases, found {len(cases)}")
    output.mkdir(parents=True, exist_ok=False)

    closures: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for case in cases:
        try:
            closures.append(diagnose_public_case(
                case,
                output_root=output / case.name,
                backend=backend,
                catalog_model_path=catalog_model_path,
                open_set_model_path=open_set_model_path,
            ))
        except Exception as exc:
            errors.append({"case_id": case.name, "error_type": type(exc).__name__, "detail": str(exc)})
    status_counts = Counter(item["diagnosis"]["status"] for item in closures)
    data_error_count = status_counts.get(UnifiedDiagnosisStatus.DATA_ERROR.value, 0)
    report = {
        "schema_version": CAMPAIGN_CLOSURE_SCHEMA_VERSION,
        "status": "PASS" if not errors and data_error_count == 0 and len(closures) == len(cases) else "FAIL",
        "data_source": "sat_agent_platform_simulation_public_packages",
        "truth_root_accessed": False,
        "public_answer_proxy_rejection_count": sum(
            "answer-proxy" in item["detail"].lower() for item in errors
        ),
        "expected_case_count": expected_case_count,
        "discovered_case_count": len(cases),
        "completed_case_count": len(closures),
        "failed_case_count": len(errors),
        "data_error_count": data_error_count,
        "status_counts": dict(sorted(status_counts.items())),
        "errors": errors,
        "cases": closures,
    }
    (output / "campaign_closure.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--public-case", type=Path)
    source.add_argument("--public-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--backend", choices=("auto", "unified", "composite"), default="auto")
    parser.add_argument("--expected-case-count", type=int)
    parser.add_argument("--catalog-model", type=Path)
    parser.add_argument("--open-set-model", type=Path)
    args = parser.parse_args(argv)
    if args.public_root is not None:
        closure = diagnose_public_campaign(
            args.public_root,
            output_root=args.output_root,
            backend=args.backend,
            expected_case_count=args.expected_case_count,
            catalog_model_path=args.catalog_model,
            open_set_model_path=args.open_set_model,
        )
        print(json.dumps({"ok": closure["status"] == "PASS", "status": closure["status"], "completed": closure["completed_case_count"], "failed": closure["failed_case_count"], "output": str(args.output_root.resolve())}, ensure_ascii=False))
        return 0 if closure["status"] == "PASS" else 2
    closure = diagnose_public_case(
        args.public_case,
        output_root=args.output_root,
        backend=args.backend,
        catalog_model_path=args.catalog_model,
        open_set_model_path=args.open_set_model,
    )
    print(json.dumps({"ok": closure["diagnosis"]["status"] != "DATA_ERROR", "case_id": closure["case_id"], "status": closure["diagnosis"]["status"], "output": str(args.output_root.resolve())}, ensure_ascii=False))
    return 0 if closure["diagnosis"]["status"] != "DATA_ERROR" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAMPAIGN_CLOSURE_SCHEMA_VERSION",
    "CLOSURE_SCHEMA_VERSION",
    "PUBLIC_CASE_SCHEMA_VERSION",
    "SIMULATION_HANDOFF_SCHEMA_VERSION",
    "diagnose_public_campaign",
    "diagnose_public_case",
]
