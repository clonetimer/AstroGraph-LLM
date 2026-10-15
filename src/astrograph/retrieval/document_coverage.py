"""Auditable P0.5 coverage checks for paired diagnostic documents.

The audit reads dataset indexes only to establish the expected taxonomy.  It
never opens post-hoc telemetry, label files, or quarantined truth artifacts,
and its receipt contains no source sample identifiers.
"""
from __future__ import annotations

import csv
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from .document_corpus import CorpusSplitRole, DocumentCorpusManifest
from .document_governance import DocumentType


ELIGIBLE_DEVELOPMENT_GROUPS = frozenset({0, 1, 2, 3, 5, 6, 7, 8})
PERMANENT_POSTHOC_GROUPS = frozenset({4, 9})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _dimension_values(
    rows: list[dict[str, str]], *, dataset_id: str,
    severity_contract: dict[str, Any] | None = None,
) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {
        "diagnostic_class": {row["diagnostic_class"] for row in rows if row.get("diagnostic_class")},
    }
    subsystem = {row.get("subsystem", "") for row in rows} - {""}
    output["subsystem"] = subsystem or {
        "adcs" if dataset_id.casefold().startswith("adcs") else "whole-spacecraft"
    }
    scenario_field = "scenario_id" if any(row.get("scenario_id") for row in rows) else "mission_id"
    output[f"scenario:{scenario_field}"] = {
        row[scenario_field] for row in rows if row.get(scenario_field)
    }
    for field in (
        "operating_phase_at_fault",
        "fault_injection_phase",
        "operating_phase_id",
    ):
        values = {row[field] for row in rows if row.get(field)}
        if values:
            output[f"stage:{field}"] = values
    severity_values = {
        value
        for row in rows
        for value in [_severity_band(row, severity_contract=severity_contract)]
        if value is not None
    }
    if severity_values:
        output["severity_band"] = severity_values
    return output


def _mapping(value: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _severity_band(
    row: dict[str, str], *, severity_contract: dict[str, Any] | None
) -> str | None:
    calibration = _mapping(row.get("calibration", ""))
    if calibration.get("severity_band"):
        return str(calibration["severity_band"])
    if severity_contract is None:
        return None
    rule = severity_contract.get("rules", {}).get(row.get("diagnostic_class"))
    if not isinstance(rule, dict):
        return None
    if rule.get("mode") == "FIXED":
        return str(rule.get("band")) if rule.get("band") else None
    if rule.get("mode") != "PARAMETER":
        return None
    parameters = _mapping(row.get("event_parameters", ""))
    parameter = str(rule.get("parameter", ""))
    if parameter not in parameters:
        return None
    minimum = float(rule["minimum"])
    maximum = float(rule["maximum"])
    if maximum <= minimum:
        raise ValueError("severity parameter maximum must exceed minimum")
    value = float(parameters[parameter])
    normalized = (value - minimum) / (maximum - minimum)
    if rule.get("severity_increases") == "LOWER":
        normalized = 1.0 - normalized
    elif rule.get("severity_increases") != "HIGHER":
        raise ValueError("severity_increases must be LOWER or HIGHER")
    normalized = min(1.0, max(0.0, normalized))
    for band in severity_contract.get("normalized_bands", []):
        lower = float(band["lower_inclusive"])
        if normalized < lower:
            continue
        if "upper_exclusive" in band and normalized < float(band["upper_exclusive"]):
            return str(band["band"])
        if "upper_inclusive" in band and normalized <= float(band["upper_inclusive"]):
            return str(band["band"])
    raise ValueError("normalized severity does not match a configured band")


def _coverage_fraction(expected: set[str], covered: set[str]) -> float:
    return len(expected & covered) / len(expected) if expected else 1.0


def audit_document_coverage(
    *,
    manifest_path: str | Path,
    dataset_index_paths: dict[str, str | Path],
    static_dataset_ids: dict[str, str],
    fault_facts_path: str | Path,
    nominal_facts_path: str | Path,
    severity_contract_paths: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Verify taxonomy/dimension coverage without exposing sample identities."""

    manifest_path = Path(manifest_path).resolve()
    fault_facts_path = Path(fault_facts_path).resolve()
    nominal_facts_path = Path(nominal_facts_path).resolve()
    manifest = DocumentCorpusManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    fault_facts = json.loads(fault_facts_path.read_text(encoding="utf-8"))
    nominal_facts = json.loads(nominal_facts_path.read_text(encoding="utf-8"))
    if set(dataset_index_paths) != set(static_dataset_ids):
        raise ValueError("dataset index and static dataset mappings must have identical keys")
    severity_contract_paths = severity_contract_paths or {}
    unknown_contract_datasets = set(severity_contract_paths) - set(dataset_index_paths)
    if unknown_contract_datasets:
        raise ValueError("severity contract references an unknown dataset")

    static_faults = {
        (str(row["dataset_id"]), str(row["fault_class_id"])) for row in fault_facts
    }
    static_nominal = {
        str(row["dataset_id"])
        for row in nominal_facts
        if row.get("fault_class_id") == "nominal_operation"
    }
    bundles_by_dataset: dict[str, list[tuple[object, dict[str, str]]]] = {}
    input_hashes: dict[str, str] = {
        "manifest": _sha256_file(manifest_path),
        "fault_facts": _sha256_file(fault_facts_path),
        "nominal_facts": _sha256_file(nominal_facts_path),
    }
    index_rows: dict[str, list[dict[str, str]]] = {}
    severity_contracts: dict[str, dict[str, Any]] = {}
    violations: list[str] = []

    for dataset_id, raw_path in sorted(dataset_index_paths.items()):
        path = Path(raw_path).resolve()
        rows = _read_rows(path)
        index_rows[dataset_id] = rows
        input_hashes[f"index:{dataset_id}"] = _sha256_file(path)
        contract_path = severity_contract_paths.get(dataset_id)
        if contract_path is not None:
            resolved_contract = Path(contract_path).resolve()
            contract = json.loads(resolved_contract.read_text(encoding="utf-8"))
            if contract.get("dataset_id") != dataset_id:
                raise ValueError("severity contract dataset_id mismatch")
            expected_generator_sha = manifest.generator_sha256.get(dataset_id)
            if contract.get("source_generator_sha256") != expected_generator_sha:
                raise ValueError("severity contract generator hash mismatch")
            severity_contracts[dataset_id] = contract
            input_hashes[f"severity_contract:{dataset_id}"] = _sha256_file(resolved_contract)
        row_by_ref = {
            hashlib.sha256(row["sample_id"].encode("utf-8")).hexdigest(): row
            for row in rows
        }
        mapped: list[tuple[object, dict[str, str]]] = []
        for bundle in (item for item in manifest.bundles if item.dataset_id == dataset_id):
            row = row_by_ref.get(bundle.source_episode_ref_sha256)
            if row is None:
                violations.append(f"{dataset_id}:UNKNOWN_SOURCE_EPISODE_REFERENCE")
                continue
            mapped.append((bundle, row))
        bundles_by_dataset[dataset_id] = mapped

    unknown_manifest_datasets = sorted(
        {item.dataset_id for item in manifest.bundles} - set(dataset_index_paths)
    )
    violations.extend(
        f"{dataset_id}:MANIFEST_DATASET_NOT_CONFIGURED"
        for dataset_id in unknown_manifest_datasets
    )

    dataset_reports: dict[str, Any] = {}
    for dataset_id in sorted(dataset_index_paths):
        all_rows = index_rows[dataset_id]
        mapped = bundles_by_dataset[dataset_id]
        mapped_rows = [row for _, row in mapped]
        eligible_rows = [
            row for row in all_rows
            if int(row["sample_index"]) in ELIGIBLE_DEVELOPMENT_GROUPS
        ]
        eligible_refs = {
            hashlib.sha256(row["sample_id"].encode("utf-8")).hexdigest()
            for row in eligible_rows
        }
        mapped_refs = {bundle.source_episode_ref_sha256 for bundle, _ in mapped}
        missing_eligible_refs = eligible_refs - mapped_refs
        mapped_posthoc_count = sum(
            int(row["sample_index"]) in PERMANENT_POSTHOC_GROUPS for _, row in mapped
        )
        causal_pair_pass_count = sum(
            {
                artifact.descriptor.document_type
                for artifact in bundle.artifacts
                if artifact.descriptor.visibility_class.value == "OBSERVATION_CAUSAL"
            }
            >= {DocumentType.TEST_LOG, DocumentType.ANOMALY_REPORT}
            for bundle, _ in mapped
        )
        if missing_eligible_refs:
            violations.append(f"{dataset_id}:ELIGIBLE_EPISODE_BUNDLE_INCOMPLETE")
        if mapped_posthoc_count:
            violations.append(f"{dataset_id}:POSTHOC_EPISODE_ENTERED_CORPUS")
        if causal_pair_pass_count != len(mapped):
            violations.append(f"{dataset_id}:EPISODE_CAUSAL_DOCUMENT_PAIR_INCOMPLETE")
        contract = severity_contracts.get(dataset_id)
        expected_dimensions = _dimension_values(
            all_rows, dataset_id=dataset_id, severity_contract=contract
        )
        covered_dimensions = _dimension_values(
            mapped_rows, dataset_id=dataset_id, severity_contract=contract
        )
        if "severity_band" not in expected_dimensions:
            expected_dimensions["severity_band"] = {"DECLARED_SEVERITY_REQUIRED"}
            covered_dimensions["severity_band"] = set()
        dimension_report: dict[str, Any] = {}
        for dimension, expected in sorted(expected_dimensions.items()):
            covered = covered_dimensions.get(dimension, set())
            missing = sorted(expected - covered)
            fraction = _coverage_fraction(expected, covered)
            dimension_report[dimension] = {
                "expected_count": len(expected),
                "covered_count": len(expected & covered),
                "coverage": fraction,
                "missing_values": missing,
            }
            if missing:
                violations.append(f"{dataset_id}:DIMENSION_INCOMPLETE:{dimension}")

        class_report: dict[str, Any] = {}
        static_dataset_id = static_dataset_ids[dataset_id]
        for fault_class in sorted(expected_dimensions["diagnostic_class"]):
            class_bundles = [
                bundle for bundle, row in mapped if row.get("diagnostic_class") == fault_class
            ]
            document_types = {
                artifact.descriptor.document_type.value
                for bundle in class_bundles
                for artifact in bundle.artifacts
            }
            test_log = DocumentType.TEST_LOG.value in document_types
            anomaly = DocumentType.ANOMALY_REPORT.value in document_types
            closure_count = sum(
                artifact.descriptor.document_type == DocumentType.CLOSURE_REPORT
                for bundle in class_bundles
                if bundle.split_role == CorpusSplitRole.TRAIN
                for artifact in bundle.artifacts
            )
            if fault_class == "nominal":
                static_reference = static_dataset_id in static_nominal
                historical_requirement_met = True
            else:
                static_reference = (static_dataset_id, fault_class) in static_faults
                historical_requirement_met = closure_count > 0
            passed = test_log and anomaly and static_reference and historical_requirement_met
            class_report[fault_class] = {
                "runtime_test_log_present": test_log,
                "runtime_anomaly_report_present": anomaly,
                "static_reference_present": static_reference,
                "train_historical_closure_candidate_count": closure_count,
                "historical_requirement_met": historical_requirement_met,
                "passed": passed,
            }
            if not passed:
                violations.append(f"{dataset_id}:CLASS_DOCUMENT_COVERAGE:{fault_class}")

        document_types_present = {
            artifact.descriptor.document_type.value
            for bundle, _ in mapped
            for artifact in bundle.artifacts
        }
        synthetic_types = {
            "DESIGN_FMEA_TEST" if any(item[0] == static_dataset_id for item in static_faults) else "",
            "NOMINAL_REFERENCE" if static_dataset_id in static_nominal else "",
        } - {""}
        required_document_types = {
            DocumentType.TEST_LOG.value,
            DocumentType.ANOMALY_REPORT.value,
            DocumentType.CLOSURE_REPORT.value,
            "DESIGN_FMEA_TEST",
            "NOMINAL_REFERENCE",
        }
        present_document_types = document_types_present | synthetic_types
        missing_types = sorted(required_document_types - present_document_types)
        if missing_types:
            violations.append(f"{dataset_id}:DOCUMENT_TYPES_INCOMPLETE")
        dataset_reports[dataset_id] = {
            "source_row_count": len(all_rows),
            "eligible_train_development_row_count": len(eligible_rows),
            "paired_train_development_row_count": len(mapped_rows),
            "eligible_episode_bundle_coverage": (
                len(eligible_refs & mapped_refs) / len(eligible_refs) if eligible_refs else 1.0
            ),
            "missing_eligible_episode_bundle_count": len(missing_eligible_refs),
            "paired_episode_causal_log_and_anomaly_count": causal_pair_pass_count,
            "paired_episode_causal_document_coverage": (
                causal_pair_pass_count / len(mapped) if mapped else 1.0
            ),
            "posthoc_bundle_count": mapped_posthoc_count,
            "class_count": len(class_report),
            "class_pass_count": sum(item["passed"] for item in class_report.values()),
            "class_coverage": (
                sum(item["passed"] for item in class_report.values()) / len(class_report)
                if class_report else 1.0
            ),
            "dimensions": dimension_report,
            "document_types": {
                "required": sorted(required_document_types),
                "present": sorted(present_document_types),
                "missing": missing_types,
                "coverage": _coverage_fraction(required_document_types, present_document_types),
            },
            "classes": class_report,
        }

    body: dict[str, Any] = {
        "schema_version": "astrograph-document-coverage-audit.v1",
        "status": "PASS" if not violations else "FAIL",
        "input_sha256": dict(sorted(input_hashes.items())),
        "datasets": dataset_reports,
        "violations": sorted(set(violations)),
        "posthoc_telemetry_or_truth_opened": False,
        "source_sample_identifiers_emitted": False,
        "claim_boundary": (
            "This audit establishes document workload coverage over dataset taxonomy and index "
            "dimensions. It does not establish document correctness, human review, retrieval "
            "quality, diagnostic gain, independent causal evidence, or Active KG publication."
        ),
    }
    return {**body, "audit_sha256": _canonical_sha256(body)}


__all__ = ["audit_document_coverage"]
