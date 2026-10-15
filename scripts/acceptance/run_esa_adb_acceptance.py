#!/usr/bin/env python3
"""Run local ESA-ADB structural acceptance and optional diagnosis smoke.

Required:
  --root DATASET_ROOT
  --output EVIDENCE_DIR

Optional diagnosis smoke:
  --entity ENTITY_ID --edges knowledge_edges.csv
  [--policy diagnosis.json] [--signal-mapping mapping.json]
  [--channels comma,separated] [--start VALUE] [--end VALUE]
  [--trust-local-pickle]

The raw archive remains local and read-only. Only redacted profiles, bounded
validation results, diagnosis metadata, and logs are written.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astrograph.audit import JsonlAuditStore  # noqa: E402
from astrograph.config import DiagnosisPolicy  # noqa: E402
from astrograph.data_sources import (  # noqa: E402
    DataSourceType,
    DatasetSourceConfig,
    ProbeOptions,
    build_data_source,
    probe_data_source,
    validate_data_source,
)
from astrograph.diagnosis import (  # noqa: E402
    DatasetDiagnosisRequest,
    DatasetDiagnosisRunner,
    JsonDiagnosisRunStateStore,
)
from astrograph.knowledge.repository import CsvKnowledgeRepository  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--timestamp-column", default="")
    parser.add_argument("--label-columns", default="")
    parser.add_argument(
        "--trust-local-pickle",
        action="store_true",
        help=(
            "Acknowledge that raw ESA-ADB ZIP archives contain trusted pandas "
            "pickle objects that may execute code while loading"
        ),
    )
    parser.add_argument("--max-entities", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=10_000)
    parser.add_argument("--hash-mode", choices=("none", "metadata", "full"), default="metadata")
    parser.add_argument("--entity", default="")
    parser.add_argument("--edges", type=Path, default=None)
    parser.add_argument("--policy", type=Path, default=ROOT / "configs" / "diagnosis.json")
    parser.add_argument("--signal-mapping", type=Path, default=None)
    parser.add_argument("--channels", default="")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--baseline-fraction", type=float, default=0.2)
    parser.add_argument("--z-threshold", type=float, default=6.0)
    parser.add_argument("--minimum-baseline-points", type=int, default=10)
    args = parser.parse_args()

    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    log: list[str] = [f"{_utc_now()} acceptance started"]
    labels = [item.strip() for item in args.label_columns.split(",") if item.strip()]
    config = DatasetSourceConfig(
        source_type=DataSourceType.ESA_ADB,
        root=str(args.root),
        read_only=True,
        lazy=True,
        recursive=True,
        timestamp_column=args.timestamp_column or None,
        label_columns=labels,
        profile_path=None if args.profile is None else str(args.profile),
        metadata={"trusted_pickle": bool(args.trust_local_pickle)},
    )

    probe_status = "NOT_RUN"
    validation_status = "NOT_RUN"
    diagnosis_status = "NOT_REQUESTED"
    diagnosis_summary: dict[str, object] = {}
    mission_layout_detected = False
    telemetry_archive_gate = "NOT_APPLICABLE"
    telemetry_archive_entity = ""
    try:
        probe_data_source(
            build_data_source(config),
            out / "probe",
            options=ProbeOptions(
                max_entities=args.max_entities,
                max_rows_per_entity=args.max_rows,
                hash_mode=args.hash_mode,
                include_statistics=False,
                include_label_values=False,
                root_token="${ASTROGRAPH_ESA_ADB_ROOT}",
            ),
        )
        probe_status = "PASS"
        log.append(f"{_utc_now()} redacted probe PASS")
    except Exception as exc:
        probe_status = "FAIL"
        failures.append({"stage": "probe", "error": f"{type(exc).__name__}: {exc}"})
        log.append(f"{_utc_now()} redacted probe FAIL: {type(exc).__name__}: {exc}")

    try:
        report = validate_data_source(
            build_data_source(config),
            max_entities=args.max_entities,
            max_rows_per_entity=args.max_rows,
        )
        _atomic_json(out / "validation_report.json", report.model_dump(mode="json"))
        validation_status = report.status
        log.append(f"{_utc_now()} structural validation {report.status}")
        if report.status == "FAIL":
            failures.append({"stage": "validation", "error": "structural validation returned FAIL"})
    except Exception as exc:
        validation_status = "FAIL"
        failures.append({"stage": "validation", "error": f"{type(exc).__name__}: {exc}"})
        log.append(f"{_utc_now()} structural validation FAIL: {type(exc).__name__}: {exc}")

    try:
        source = build_data_source(config)
        with source:
            source_metadata = source.get_metadata()
            mission_layout_detected = bool(
                source_metadata.metadata.get("mission_layout_detected", False)
            )
            if mission_layout_detected:
                telemetry_entities = [
                    item
                    for item in source.list_entities()
                    if item.format == "pickle_zip"
                    and item.metadata.get("entity_kind") == "telemetry"
                ]
                if not telemetry_entities:
                    telemetry_archive_gate = "FAIL"
                    failures.append(
                        {
                            "stage": "telemetry_archive_gate",
                            "error": "no allow-listed telemetry pickle ZIP entity was discovered",
                        }
                    )
                else:
                    first = telemetry_entities[0]
                    channels = source.list_channels(first.entity_id)
                    frame = source.read_range(
                        first.entity_id,
                        channels=channels,
                        limit=min(args.max_rows, 1000),
                    )
                    if frame.empty or not channels or channels[0] not in frame.columns:
                        telemetry_archive_gate = "FAIL"
                        failures.append(
                            {
                                "stage": "telemetry_archive_gate",
                                "error": "telemetry ZIP opened but returned no model channel rows",
                            }
                        )
                    else:
                        telemetry_archive_gate = "PASS"
                        telemetry_archive_entity = first.entity_id
                log.append(
                    f"{_utc_now()} telemetry pickle ZIP gate {telemetry_archive_gate}"
                )
    except Exception as exc:
        telemetry_archive_gate = "FAIL"
        failures.append(
            {
                "stage": "telemetry_archive_gate",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        log.append(
            f"{_utc_now()} telemetry pickle ZIP gate FAIL: {type(exc).__name__}: {exc}"
        )

    diagnosis_requested = bool(args.entity or args.edges)
    if diagnosis_requested:
        if not args.entity or args.edges is None:
            diagnosis_status = "CONFIGURATION_ERROR"
            failures.append(
                {
                    "stage": "diagnosis",
                    "error": "--entity and --edges are both required for diagnosis smoke",
                }
            )
        else:
            try:
                mapping = (
                    {}
                    if args.signal_mapping is None
                    else json.loads(args.signal_mapping.read_text(encoding="utf-8"))
                )
                channels = [item.strip() for item in args.channels.split(",") if item.strip()]
                runner = DatasetDiagnosisRunner(
                    build_data_source(config),
                    CsvKnowledgeRepository(args.edges),
                    state_store=JsonDiagnosisRunStateStore(out / "diagnosis_state"),
                    policy=DiagnosisPolicy.from_json(args.policy),
                    audit_store=JsonlAuditStore(out / "diagnosis_audit.jsonl"),
                )
                result = runner.run(
                    DatasetDiagnosisRequest(
                        request_id="ESA-ADB-LOCAL-ACCEPTANCE",
                        entity_id=args.entity,
                        start=args.start or None,
                        end=args.end or None,
                        channels=channels,
                        baseline_fraction=args.baseline_fraction,
                        z_threshold=args.z_threshold,
                        minimum_baseline_points=args.minimum_baseline_points,
                        signal_mapping=mapping,
                        metadata={"acceptance": "local-esa-adb-r4f3"},
                    ),
                    output_dir=out / "diagnosis",
                )
                diagnosis_status = "PASS" if result.conclusion.level.value != "system_error" else "FAIL"
                diagnosis_summary = {
                    "run_id": result.run_id,
                    "diagnosis_id": result.diagnosis_id,
                    "conclusion": result.conclusion.model_dump(mode="json"),
                    "observation_count": len(result.observations),
                    "detection_count": len(result.detections),
                    "raw_rows_persisted": result.provenance.get("raw_rows_persisted"),
                    "labels_used_by_detector": result.provenance.get("labels_used_by_detector"),
                }
                if diagnosis_status == "FAIL":
                    failures.append({"stage": "diagnosis", "error": "system_error conclusion"})
                log.append(f"{_utc_now()} diagnosis smoke {diagnosis_status}")
            except Exception as exc:
                diagnosis_status = "FAIL"
                failures.append({"stage": "diagnosis", "error": f"{type(exc).__name__}: {exc}"})
                log.append(f"{_utc_now()} diagnosis smoke FAIL: {type(exc).__name__}: {exc}")
                log.append(traceback.format_exc())

    structural_ok = (
        probe_status == "PASS"
        and validation_status in {"PASS", "WARNING"}
        and telemetry_archive_gate in {"NOT_APPLICABLE", "PASS"}
    )
    diagnosis_ok = diagnosis_status in {"NOT_REQUESTED", "PASS"}
    status = "PASS" if structural_ok and diagnosis_ok else "FAIL"
    summary = {
        "schema_version": "astrograph-esa-adb-local-acceptance.v1",
        "generated_at": _utc_now(),
        "release_id": "V0.5.30-R4F3.2",
        "status": status,
        "probe_status": probe_status,
        "validation_status": validation_status,
        "mission_layout_detected": mission_layout_detected,
        "telemetry_archive_gate": telemetry_archive_gate,
        "telemetry_archive_entity": telemetry_archive_entity,
        "diagnosis_status": diagnosis_status,
        "diagnosis": diagnosis_summary,
        "raw_dataset_uploaded": False,
        "raw_rows_persisted": False,
        "root_token": "${ASTROGRAPH_ESA_ADB_ROOT}",
        "claim_boundary": (
            "PASS proves bounded structural readability for this local dataset revision. "
            "A diagnosis PASS proves only execution/semantic separation, not model accuracy, "
            "flight generalization, or calibrated fault probabilities."
        ),
    }
    _atomic_json(out / "acceptance_summary.json", summary)
    _atomic_json(out / "failures.json", failures)
    (out / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
