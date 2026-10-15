"""Delivery-evidence consistency rules for source-only and qualified packages.

This module does not recreate or infer missing historical evidence.  It only makes
package dependencies explicit and fail-closed when a missing artifact has not been
declared by delivery policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


REGISTRY_RELATIVE_PATH = Path("configs/release_evidence_dependencies.v1.json")


@dataclass(frozen=True)
class EvidenceDependencyResolution:
    dependency_id: str
    path: str
    status: str
    classification: str
    delivery_policy: str
    consumers: tuple[str, ...]
    rationale: str

    @property
    def available(self) -> bool:
        return self.status == "BUNDLED"

    @property
    def external_declared(self) -> bool:
        return self.status == "EXTERNAL_ARCHIVE_DECLARED"


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not value.strip():
        raise ValueError(f"unsafe delivery-evidence path: {value!r}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_delivery_evidence_registry(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    registry_path = root / REGISTRY_RELATIVE_PATH
    if not registry_path.is_file():
        raise FileNotFoundError(f"delivery-evidence registry missing: {registry_path}")
    registry = _load_json(registry_path)
    if registry.get("schema_version") != "astrograph-release-evidence-dependencies.v1":
        raise ValueError("unsupported delivery-evidence registry schema")
    dependencies = registry.get("dependencies")
    if not isinstance(dependencies, list):
        raise ValueError("delivery-evidence registry dependencies must be a list")
    ids: set[str] = set()
    paths: set[str] = set()
    for item in dependencies:
        if not isinstance(item, dict):
            raise ValueError("delivery-evidence dependency must be an object")
        dependency_id = str(item.get("dependency_id", "")).strip()
        relative = str(item.get("path", "")).strip()
        _safe_relative(relative)
        if not dependency_id or dependency_id in ids:
            raise ValueError(f"duplicate/empty delivery-evidence dependency id: {dependency_id!r}")
        if relative in paths:
            raise ValueError(f"duplicate delivery-evidence path: {relative}")
        if item.get("delivery_policy") not in {"MUST_BUNDLE", "ARCHIVE_EXTERNAL_ALLOWED"}:
            raise ValueError(f"invalid delivery policy for {dependency_id}")
        if item.get("classification") not in {"CURRENT_RUNTIME_EVIDENCE", "HISTORICAL_FROZEN_EVIDENCE"}:
            raise ValueError(f"invalid evidence classification for {dependency_id}")
        consumers = item.get("consumers")
        if not isinstance(consumers, list) or not consumers or not all(isinstance(v, str) and v.strip() for v in consumers):
            raise ValueError(f"dependency {dependency_id} must name at least one consumer")
        if not str(item.get("rationale", "")).strip():
            raise ValueError(f"dependency {dependency_id} requires a rationale")
        ids.add(dependency_id)
        paths.add(relative)
    return registry


def _delivery_profile(root: Path) -> dict[str, Any] | None:
    path = root / "DELIVERY_PROFILE.json"
    return _load_json(path) if path.is_file() else None


def resolve_delivery_evidence(root: str | Path, relative_path: str) -> EvidenceDependencyResolution:
    root = Path(root).resolve()
    relative = _safe_relative(relative_path).as_posix()
    registry = load_delivery_evidence_registry(root)
    item = next((x for x in registry["dependencies"] if x["path"] == relative), None)
    if item is None:
        raise KeyError(f"delivery-evidence dependency is not registered: {relative}")
    artifact = root / relative
    if artifact.is_file():
        status = "BUNDLED"
    else:
        profile = _delivery_profile(root)
        source_only = bool(
            profile
            and profile.get("source_kind") == "CURRENT_SOURCE_ONLY"
            and profile.get("historical_records_included") is False
        )
        external_allowed = (
            item["classification"] == "HISTORICAL_FROZEN_EVIDENCE"
            and item["delivery_policy"] == "ARCHIVE_EXTERNAL_ALLOWED"
            and source_only
        )
        status = "EXTERNAL_ARCHIVE_DECLARED" if external_allowed else "MISSING_REQUIRED"
    return EvidenceDependencyResolution(
        dependency_id=item["dependency_id"],
        path=relative,
        status=status,
        classification=item["classification"],
        delivery_policy=item["delivery_policy"],
        consumers=tuple(item["consumers"]),
        rationale=item["rationale"],
    )


def validate_delivery_evidence(root: str | Path, *, require_bundled: bool = False) -> dict[str, Any]:
    root = Path(root).resolve()
    registry = load_delivery_evidence_registry(root)
    resolutions = [resolve_delivery_evidence(root, item["path"]) for item in registry["dependencies"]]
    missing_required = [x for x in resolutions if x.status == "MISSING_REQUIRED"]
    external = [x for x in resolutions if x.external_declared]
    errors = [f"MISSING_REQUIRED:{x.path}" for x in missing_required]
    if require_bundled:
        errors.extend(f"EXTERNAL_NOT_ALLOWED:{x.path}" for x in external)
    status = "FAIL" if errors else ("PASS_WITH_EXTERNAL_DEPENDENCIES" if external else "PASS")
    return {
        "schema_version": "astrograph-delivery-evidence-consistency.v1",
        "status": status,
        "delivery_profile": (_delivery_profile(root) or {}).get("profile_id"),
        "require_bundled": require_bundled,
        "dependency_count": len(resolutions),
        "bundled_count": sum(x.available for x in resolutions),
        "external_declared_count": len(external),
        "errors": errors,
        "dependencies": [
            {
                "dependency_id": x.dependency_id,
                "path": x.path,
                "status": x.status,
                "classification": x.classification,
                "delivery_policy": x.delivery_policy,
                "consumers": list(x.consumers),
                "rationale": x.rationale,
            }
            for x in resolutions
        ],
        "claim_boundary": (
            "PASS_WITH_EXTERNAL_DEPENDENCIES is valid only for the declared CURRENT_SOURCE_ONLY delivery profile; "
            "it does not prove that archived historical evidence is present or replayable. Strict qualification must use require_bundled=true."
        ),
    }


__all__ = [
    "EvidenceDependencyResolution",
    "REGISTRY_RELATIVE_PATH",
    "load_delivery_evidence_registry",
    "resolve_delivery_evidence",
    "validate_delivery_evidence",
]
