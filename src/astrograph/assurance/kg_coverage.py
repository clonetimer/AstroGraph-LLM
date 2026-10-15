from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from astrograph.evidence import EvidenceRole
from astrograph.evidence.configured_runtime import ConfiguredRuntimeEvidencePackage


class KGCoverageAuditError(ValueError):
    """Raised when frozen coverage assets disagree with one another."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise KGCoverageAuditError(f"expected JSON object: {path}")
    return value


def _coverage(
    numerator_ids: set[str],
    denominator_ids: set[str] | None,
    denominator_count: int,
) -> dict[str, Any]:
    if denominator_count <= 0:
        raise KGCoverageAuditError("coverage denominator must be positive")
    numerator = len(numerator_ids if denominator_ids is None else numerator_ids & denominator_ids)
    return {
        "numerator": numerator,
        "denominator": denominator_count,
        "ratio": numerator / denominator_count,
    }


def _audit_dataset(root: Path, dataset_dir_name: str) -> dict[str, Any]:
    dataset_dir = root / "configs" / "datasets" / dataset_dir_name
    paths = {
        "candidate_universe": dataset_dir / "candidate_universe.json",
        "fault_class_map": dataset_dir / "fault_class_map.json",
        "fault_ontology_bindings": dataset_dir / "fault_ontology_bindings.json",
        "runtime_evidence": dataset_dir / "runtime_evidence.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise KGCoverageAuditError(f"missing KG coverage assets: {missing}")

    candidate = _load_json(paths["candidate_universe"])
    fault_map = _load_json(paths["fault_class_map"])
    ontology = _load_json(paths["fault_ontology_bindings"])
    runtime_package = ConfiguredRuntimeEvidencePackage.model_validate_json(
        paths["runtime_evidence"].read_text(encoding="utf-8")
    )

    dataset_fault_count = int(candidate["dataset_class_count"])
    mappings = fault_map.get("mappings") or {}
    if not isinstance(mappings, dict):
        raise KGCoverageAuditError("fault class mappings must be an object")
    strict_mapping_fault_ids = {str(value) for value in mappings.values()}
    strict_mapping_class_ids = {str(value) for value in mappings}
    runtime_candidate_fault_ids = {
        str(value) for value in candidate.get("runtime_fault_ids", [])
    }
    ontology_binding_class_ids = {
        str(item["fault_class_id"]) for item in ontology.get("bindings", [])
    }
    positive_evidence_fault_ids = {
        item.target_fault_id
        for item in runtime_package.evidence_definitions
        if item.role in {EvidenceRole.REQUIRED, EvidenceRole.SUPPORTING}
    }
    all_evidence_fault_ids = {
        item.target_fault_id for item in runtime_package.evidence_definitions
    }

    if len(ontology_binding_class_ids) != dataset_fault_count:
        raise KGCoverageAuditError(
            f"{dataset_dir_name}: ontology binding count does not match dataset class count"
        )
    if not strict_mapping_class_ids.issubset(ontology_binding_class_ids):
        raise KGCoverageAuditError(
            f"{dataset_dir_name}: strict mappings contain classes absent from ontology bindings"
        )
    if not runtime_candidate_fault_ids.issubset(strict_mapping_fault_ids):
        unknown = sorted(runtime_candidate_fault_ids - strict_mapping_fault_ids)
        raise KGCoverageAuditError(
            f"{dataset_dir_name}: runtime candidate universe contains unmapped faults: {unknown}"
        )
    if not all_evidence_fault_ids.issubset(strict_mapping_fault_ids):
        unknown = sorted(all_evidence_fault_ids - strict_mapping_fault_ids)
        raise KGCoverageAuditError(
            f"{dataset_dir_name}: runtime evidence targets unmapped faults: {unknown}"
        )

    declared_rule_ids = candidate.get("configured_rule_evidence_fault_ids")
    if declared_rule_ids is not None:
        declared = {str(value) for value in declared_rule_ids}
        if declared != positive_evidence_fault_ids:
            raise KGCoverageAuditError(
                f"{dataset_dir_name}: configured_rule_evidence_fault_ids disagree with "
                "positive executable evidence definitions"
            )
    elif runtime_candidate_fault_ids != positive_evidence_fault_ids:
        raise KGCoverageAuditError(
            f"{dataset_dir_name}: runtime_fault_ids disagree with positive executable "
            "evidence definitions"
        )

    catalog_mapped = candidate.get("catalog_mapped_fault_ids")
    if (
        catalog_mapped is not None
        and {str(value) for value in catalog_mapped} != strict_mapping_fault_ids
    ):
        raise KGCoverageAuditError(
            f"{dataset_dir_name}: catalog_mapped_fault_ids disagree with strict fault mappings"
        )

    expected_relative_paths = {
        "mapping_path": paths["fault_class_map"].relative_to(root).as_posix(),
        "runtime_evidence_path": paths["runtime_evidence"].relative_to(root).as_posix(),
        "generic_tbox_binding_path": paths["fault_ontology_bindings"].relative_to(root).as_posix(),
    }
    for field_name, expected_path in expected_relative_paths.items():
        if candidate.get(field_name) != expected_path:
            raise KGCoverageAuditError(
                f"{dataset_dir_name}: {field_name} does not pin the audited asset: "
                f"{candidate.get(field_name)!r} != {expected_path!r}"
            )

    source_files = {
        name: {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "sha256": _sha256_file(path),
        }
        for name, path in sorted(paths.items())
    }
    return {
        "dataset_id": candidate.get("dataset_id"),
        "dataset_fault_count": dataset_fault_count,
        "semantic_tbox_binding_count": len(ontology_binding_class_ids),
        "strict_registered_mapping_count": len(strict_mapping_fault_ids),
        "configured_runtime_candidate_count": len(runtime_candidate_fault_ids),
        "executable_positive_evidence_fault_count": len(positive_evidence_fault_ids),
        "strict_registered_fault_ids": sorted(strict_mapping_fault_ids),
        "configured_runtime_candidate_fault_ids": sorted(runtime_candidate_fault_ids),
        "executable_positive_evidence_fault_ids": sorted(positive_evidence_fault_ids),
        "coverage": {
            "against_dataset_fault_universe": _coverage(
                positive_evidence_fault_ids, None, dataset_fault_count
            ),
            "against_strict_registered_mapping_universe": _coverage(
                positive_evidence_fault_ids,
                strict_mapping_fault_ids,
                len(strict_mapping_fault_ids),
            ),
            "against_configured_runtime_candidate_universe": _coverage(
                positive_evidence_fault_ids,
                runtime_candidate_fault_ids,
                len(runtime_candidate_fault_ids),
            ),
        },
        "source_files": source_files,
        "claim_boundary": {
            "semantic_tbox_bindings_are_executable_rule_coverage": False,
            "whole_dataset_kg_gain_claim_allowed": (
                len(positive_evidence_fault_ids) == dataset_fault_count
            ),
            "configured_runtime_universe_may_be_narrower_than_dataset": (
                len(runtime_candidate_fault_ids) < dataset_fault_count
            ),
        },
    }


def build_kg_coverage_audit(project_root: str | Path) -> dict[str, Any]:
    """Build a deterministic, label-free ADCS/PHM coverage audit."""

    root = Path(project_root).resolve()
    report: dict[str, Any] = {
        "schema_version": "astrograph-kg-coverage-audit.v1",
        "coverage_semantics": "EXECUTABLE_POSITIVE_EVIDENCE_DEFINITIONS",
        "datasets": {
            "ADCS_V5_2_2": _audit_dataset(root, "adcs_v5_2_2"),
            "PHM_V6_0": _audit_dataset(root, "phm_v6_0"),
        },
        "development_policy_candidates": [
            {
                "policy_id": "KG-COVERAGE-SUPPRESS-V1",
                "partial_coverage_action": "SUPPRESS_BRANCH",
                "minimum_coverage": 1.0,
                "selection_status": "UNSELECTED_ENGINEERING_BASELINE",
            },
            {
                "policy_id": "KG-COVERAGE-SCALE-V1",
                "partial_coverage_action": "SCALE_BY_COVERAGE",
                "minimum_coverage": 1.0,
                "selection_status": "UNSELECTED_DEVELOPMENT_CANDIDATE",
            },
        ],
        "selection_boundary": {
            "allowed_development_groups": [3, 8],
            "post_hoc_only_groups": [4, 9],
            "groups_4_9_parameter_selection_prohibited": True,
            "new_confirmatory_evidence_required_for_confirmatory_claims": True,
        },
        "claim_boundary": (
            "Configuration and implementation coverage audit only. No policy candidate is selected, "
            "no performance gain is claimed, and no qualification state is changed."
        ),
    }
    report["audit_sha256"] = _canonical_sha256(report)
    return report


def write_kg_coverage_audit(project_root: str | Path, output_path: str | Path) -> Path:
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_kg_coverage_audit(project_root)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


__all__ = [
    "KGCoverageAuditError",
    "build_kg_coverage_audit",
    "write_kg_coverage_audit",
]
