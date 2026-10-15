from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetRegistration(BaseModel):
    """Versioned reference to an externally produced PHM dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    profile: str = Field(min_length=1)
    source_project: str = "satellite-simulation-phm"
    root_path: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    external_gate_status: str
    external_gate_decision: str
    external_gate_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    acceptance_scope: str = "closed_reproducible"
    open_conditions: list[str] = Field(default_factory=list)
    registered_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_count: int = Field(ge=0)
    split_counts: dict[str, int] = Field(default_factory=dict)
    source_revision: dict[str, Any] = Field(default_factory=dict)
    runtime_versions: dict[str, Any] = Field(default_factory=dict)
    dataset_contract_version: str = "phm-d0-dataset.v1"
    claim_boundary: str = "synthetic engineering dataset; not flight-qualified"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_gate_scope(self) -> "DatasetRegistration":
        if self.external_gate_status == "PASS" and self.external_gate_decision == "CLOSED_PASS":
            return self
        if (
            self.external_gate_status == "CONDITIONAL_PASS"
            and self.external_gate_decision.startswith("CONDITIONAL_PASS")
            and self.acceptance_scope == "engineering_algorithm_development"
            and self.open_conditions
        ):
            return self
        raise ValueError(
            "Dataset registration requires CLOSED_PASS, or an explicitly scoped conditional pass with open conditions"
        )


class DatasetRegistryDocument(BaseModel):
    schema_version: str = "astrograph-dataset-registry.v1"
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    datasets: list[DatasetRegistration] = Field(default_factory=list)


class DatasetRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> DatasetRegistryDocument:
        if not self.path.exists():
            return DatasetRegistryDocument()
        return DatasetRegistryDocument.model_validate_json(self.path.read_text(encoding="utf-8"))

    def upsert(self, registration: DatasetRegistration) -> DatasetRegistryDocument:
        document = self.load()
        kept = [
            item
            for item in document.datasets
            if not (
                item.dataset_id == registration.dataset_id
                and item.dataset_version == registration.dataset_version
            )
        ]
        document.datasets = sorted(
            [*kept, registration], key=lambda item: (item.dataset_id, item.dataset_version)
        )
        document.updated_at = datetime.now(timezone.utc).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        return document

    def get(self, dataset_id: str, dataset_version: str | None = None) -> DatasetRegistration:
        matches = [item for item in self.load().datasets if item.dataset_id == dataset_id]
        if dataset_version is not None:
            matches = [item for item in matches if item.dataset_version == dataset_version]
        if not matches:
            raise KeyError(f"Dataset not registered: {dataset_id} version={dataset_version!r}")
        return sorted(matches, key=lambda item: item.dataset_version)[-1]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_registration(
    dataset_root: str | Path,
    gate_report_path: str | Path,
    *,
    dataset_version: str | None = None,
) -> DatasetRegistration:
    root = Path(dataset_root).resolve()
    manifest_path = root / "dataset_manifest.json"
    split_path = root / "dataset_split_manifest.json"
    if not manifest_path.exists() or not split_path.exists():
        raise FileNotFoundError("dataset_manifest.json and dataset_split_manifest.json are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    split = json.loads(split_path.read_text(encoding="utf-8-sig"))
    gate_path = Path(gate_report_path).resolve()
    gate = json.loads(gate_path.read_text(encoding="utf-8-sig"))
    return DatasetRegistration(
        dataset_id=str(manifest["dataset_id"]),
        dataset_version=dataset_version or str(manifest["dataset_id"]),
        profile=str(manifest["profile"]),
        root_path=str(root),
        manifest_sha256=sha256_file(manifest_path),
        external_gate_status=str(gate.get("overall_status", "")),
        external_gate_decision=str(gate.get("d0_closure_decision", "")),
        external_gate_report_sha256=sha256_file(gate_path),
        run_count=int(manifest.get("complete_run_count", 0)),
        split_counts={
            "train": len(split.get("train_run_ids", [])),
            "validation": len(split.get("validation_run_ids", [])),
            "test": len(split.get("test_run_ids", [])),
            "zero_shot_validation": len(split.get("compound_fault_validation_run_ids", [])),
        },
        source_revision=dict(manifest.get("source_revision", {})),
        runtime_versions=dict(manifest.get("runtime_versions", {})),
        claim_boundary=str(manifest.get("claim_boundary", "synthetic engineering dataset; not flight-qualified")),
        metadata={
            "gate_schema_version": gate.get("schema_version", ""),
            "mandatory_check_count": gate.get("mandatory_check_count", 0),
            "mandatory_pass_count": gate.get("mandatory_pass_count", 0),
            "split_unit": split.get("split_unit", ""),
        },
    )


def build_d1_registration(
    package_root: str | Path,
    gate_report_path: str | Path,
    derived_split_path: str | Path,
) -> DatasetRegistration:
    root = Path(package_root).resolve()
    dataset_root = root / "02_dataset"
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    gate_path = Path(gate_report_path).resolve()
    gate = json.loads(gate_path.read_text(encoding="utf-8-sig"))
    split_path = Path(derived_split_path).resolve()
    split = json.loads(split_path.read_text(encoding="utf-8-sig"))
    decision = str(gate.get("decision", ""))
    if not decision.startswith("CONDITIONAL_PASS") and decision != "PASS_CLOSED":
        raise ValueError(f"D1 gate decision is not registrable: {decision}")
    conditional = decision.startswith("CONDITIONAL_PASS")
    return DatasetRegistration(
        dataset_id=str(manifest["dataset_id"]),
        dataset_version=str(manifest["dataset_id"]),
        profile=str(manifest["profile"]),
        root_path=str(root),
        manifest_sha256=sha256_file(manifest_path),
        external_gate_status="CONDITIONAL_PASS" if conditional else "PASS",
        external_gate_decision=decision if conditional else "CLOSED_PASS",
        external_gate_report_sha256=sha256_file(gate_path),
        acceptance_scope="engineering_algorithm_development" if conditional else "closed_reproducible",
        open_conditions=list(gate.get("warnings", [])) if conditional else [],
        run_count=int(manifest.get("complete_run_count", 0)),
        split_counts={str(k): int(v) for k, v in split.get("split_counts", {}).items()},
        source_revision=dict(manifest.get("source_revision", {})),
        runtime_versions=dict(manifest.get("runtime_versions", {})),
        dataset_contract_version="phm-d1-dataset.v1",
        claim_boundary=str(manifest.get("claim_boundary", "synthetic engineering dataset; not flight-qualified")),
        metadata={
            "gate_schema_version": gate.get("schema_version", ""),
            "producer_split_not_used_for_paired_residual": True,
            "derived_split_sha256": sha256_file(split_path),
            "source_snapshot_status": gate.get("source_snapshot_status", ""),
            "acceptance_scopes": gate.get("acceptance_scopes", {}),
        },
    )


def build_d3_registration(
    package_root: str | Path,
    gate_report_path: str | Path,
    derived_split_path: str | Path,
) -> DatasetRegistration:
    root = Path(package_root).resolve()
    dataset_root = root / "02_dataset"
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    gate_path = Path(gate_report_path).resolve()
    gate = json.loads(gate_path.read_text(encoding="utf-8-sig"))
    split_path = Path(derived_split_path).resolve()
    split = json.loads(split_path.read_text(encoding="utf-8-sig"))
    decision = str(gate.get("decision", ""))
    if not (decision.startswith("CONDITIONAL_PASS") or decision in {"PASS", "PASS_CLOSED"}):
        raise ValueError(f"D3 gate decision is not registrable: {decision}")
    conditional = decision.startswith("CONDITIONAL_PASS")
    return DatasetRegistration(
        dataset_id=str(manifest["dataset_id"]),
        dataset_version=str(manifest["dataset_id"]),
        profile="dataset",
        root_path=str(root),
        manifest_sha256=sha256_file(manifest_path),
        external_gate_status="CONDITIONAL_PASS" if conditional else "PASS",
        external_gate_decision=decision if conditional else "CLOSED_PASS",
        external_gate_report_sha256=sha256_file(gate_path),
        acceptance_scope="engineering_algorithm_development" if conditional else "closed_reproducible",
        open_conditions=list(gate.get("warnings", [])) if conditional else [],
        run_count=int(manifest.get("complete_run_count", 0)),
        split_counts={str(key): int(value) for key, value in split.get("split_counts", {}).items()},
        source_revision=dict(manifest.get("source_revision", {})),
        runtime_versions=dict(manifest.get("runtime_versions", {})),
        dataset_contract_version="phm-d3-rw-dataset.v1",
        claim_boundary=str(manifest.get("claim_boundary", "E0 synthetic engineering dataset")),
        metadata={
            "gate_schema_version": gate.get("schema_version", ""),
            "producer_split_not_used_for_paired_residual": True,
            "derived_split_sha256": sha256_file(split_path),
            "source_snapshot_status": gate.get("source_snapshot_status", ""),
            "producer_pair_cross_split_count": gate.get("producer_pair_cross_split_count", 0),
            "astrograph_pair_cross_split_count": gate.get("astrograph_pair_cross_split_count", 0),
            "holdout_semantics": split.get("holdout_semantics", ""),
        },
    )
