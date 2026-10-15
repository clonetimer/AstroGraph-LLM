"""R23 source-asset inventory without evaluation-label consumption.

The inventory intentionally inspects archive identity, layout and selected
schema-level metadata only.  It does not read per-case truth, frozen evaluation
predictions or embedded performance reports for model selection.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class R23AssetInventory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-r23-source-asset-inventory.v1"
    asset_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_size_bytes: int = Field(ge=1)
    member_count: int = Field(ge=1)
    uncompressed_size_bytes: int = Field(ge=1)
    top_roots: tuple[str, ...]
    extension_counts: dict[str, int]
    schema_facts: dict[str, Any]
    evaluation_content_consumed_for_selection: bool = False
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = (
        "Archive/schema inventory only. Per-case labels, frozen predictions and embedded performance reports are not "
        "consumed for R23 model fitting/tuning/selection/calibration/optimization."
    )

    @model_validator(mode="after")
    def verify_receipt(self) -> "R23AssetInventory":
        if self.evaluation_content_consumed_for_selection:
            raise ValueError("R23 asset inventory cannot authorize evaluation reuse for selection")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if _digest(payload) != self.receipt_sha256:
            raise ValueError("R23 asset inventory receipt mismatch")
        return self


def _safe_schema_facts(archive: zipfile.ZipFile) -> dict[str, Any]:
    names = archive.namelist()
    facts: dict[str, Any] = {}
    dataset_manifests = sorted(name for name in names if name.endswith("/dataset_manifest.json"))
    if dataset_manifests:
        manifest = json.loads(archive.read(dataset_manifests[0]))
        allowed = (
            "schema_version", "status", "capability_id", "design", "mission_count",
            "operating_scenario_count", "condition_count", "anomaly_condition_count",
            "nominal_baseline_run_count", "fault_run_count", "total_run_count",
            "new_sample_count", "expected_new_sample_count", "referenced_adcs_sample_count",
            "logical_total_sample_count", "subsystems", "fault_count", "phase_count",
            "monte_carlo_method",
        )
        facts["dataset_manifest_member"] = dataset_manifests[0]
        facts["dataset_manifest"] = {key: manifest[key] for key in allowed if key in manifest}
    if "Manifest.json" in names:
        manifest = json.loads(archive.read("Manifest.json"))
        artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else None
        facts["package_manifest"] = {
            "product": manifest.get("product") if isinstance(manifest, dict) else None,
            "package_type": manifest.get("package_type") if isinstance(manifest, dict) else None,
            "artifact_count": len(artifacts) if isinstance(artifacts, dict) else None,
        }
    if "interfaces/ComponentRegistry.json" in names:
        registry = json.loads(archive.read("interfaces/ComponentRegistry.json"))
        components = registry.get("components") if isinstance(registry, dict) else None
        facts["component_registry"] = {
            "registry_version": registry.get("registry_version") if isinstance(registry, dict) else None,
            "component_count": len(components) if isinstance(components, list) else None,
        }
    return facts


def build_zip_asset_inventory(path: Path, *, asset_id: str, role: str) -> R23AssetInventory:
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if not infos:
            raise ValueError("R23 asset archive is empty")
        roots = sorted({PurePosixPath(info.filename).parts[0] for info in infos if PurePosixPath(info.filename).parts})
        extensions = Counter((PurePosixPath(info.filename).suffix.lower() or "<none>") for info in infos)
        facts = _safe_schema_facts(archive)
        payload = {
            "schema_version": "astrograph-r23-source-asset-inventory.v1",
            "asset_id": asset_id,
            "role": role,
            "filename": path.name,
            "source_artifact_sha256": _file_sha256(path),
            "source_size_bytes": path.stat().st_size,
            "member_count": len(infos),
            "uncompressed_size_bytes": sum(info.file_size for info in infos),
            "top_roots": tuple(roots),
            "extension_counts": dict(sorted(extensions.items())),
            "schema_facts": facts,
            "evaluation_content_consumed_for_selection": False,
            "claim_boundary": (
                "Archive/schema inventory only. Per-case labels, frozen predictions and embedded performance reports are not "
                "consumed for R23 model fitting/tuning/selection/calibration/optimization."
            ),
        }
    return R23AssetInventory(**payload, receipt_sha256=_digest(payload))


__all__ = ["R23AssetInventory", "build_zip_asset_inventory"]
