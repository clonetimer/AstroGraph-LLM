"""Load a Sat-Sim scientific-screening model package into AstroGraph.

This is a deliberately narrow cross-project bridge.  It validates the model
manifest and artifact hash, registers the estimator through AstroGraph's
existing DataDrivenModelRegistry, and preserves the engineering-only claim
boundary when the source dataset is not A-level Basilisk evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from astrograph.assurance import (
    AdmissionSigner,
    build_flat_feature_mapping_contract_registry,
)
from astrograph.data_driven import (
    CallableRawModelProvider,
    DataDrivenModelManifest,
    DataDrivenModelRegistry,
    ModelInferenceRequest,
    RawModelOutput,
)


@dataclass(frozen=True)
class SatSimScientificModelPackage:
    manifest: DataDrivenModelManifest
    artifact_path: Path
    artifact_sha256: str
    feature_order: tuple[str, ...]
    estimator: Any
    claim_boundary: str

    @property
    def formal_model_binding_eligible(self) -> bool:
        metadata = self.manifest.metadata
        return bool(metadata.get("formal_model_binding_enabled", False)) and bool(
            self.manifest.calibration_id
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sat_sim_scientific_model_package(
    manifest_path: str | Path,
) -> SatSimScientificModelPackage:
    manifest_path = Path(manifest_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = DataDrivenModelManifest.model_validate(raw)
    metadata = manifest.metadata
    artifact_name = str(metadata.get("artifact_file") or "")
    expected_sha = str(metadata.get("artifact_sha256") or "")
    if not artifact_name or len(expected_sha) != 64:
        raise ValueError("model manifest must declare artifact_file and artifact_sha256")
    artifact_path = manifest_path.parent / artifact_name
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    actual_sha = _sha256(artifact_path)
    if actual_sha != expected_sha:
        raise ValueError("model artifact SHA-256 mismatch")
    bundle = joblib.load(artifact_path)
    if not isinstance(bundle, Mapping) or "model" not in bundle:
        raise ValueError("unsupported Sat-Sim model artifact")
    feature_order = tuple(str(item) for item in bundle.get("feature_columns") or [])
    declared_order = tuple(str(item) for item in manifest.input_contract.get("feature_order") or [])
    if not feature_order or feature_order != declared_order:
        raise ValueError("artifact and manifest feature order differ")
    return SatSimScientificModelPackage(
        manifest=manifest,
        artifact_path=artifact_path,
        artifact_sha256=actual_sha,
        feature_order=feature_order,
        estimator=bundle["model"],
        claim_boundary=manifest.claim_boundary,
    )


def build_registry_from_sat_sim_package(
    package: SatSimScientificModelPackage,
    *,
    admission_signer: AdmissionSigner | None = None,
) -> DataDrivenModelRegistry:
    def predict_fn(request: ModelInferenceRequest) -> RawModelOutput:
        values = request.payload.get("features")
        if not isinstance(values, Mapping):
            raise ValueError("payload.features mapping is required")
        missing = [name for name in package.feature_order if name not in values]
        unknown = sorted(set(values) - set(package.feature_order))
        if missing or unknown:
            raise ValueError(
                f"Sat-Sim feature contract mismatch; missing={missing}, unknown={unknown}"
            )
        row_values = [float(values[name]) for name in package.feature_order]
        if any(not math.isfinite(value) for value in row_values):
            raise ValueError("Sat-Sim model features must be finite")
        row = pd.DataFrame(
            [row_values],
            columns=list(package.feature_order),
        )
        probabilities = package.estimator.predict_proba(row)[0]
        classes = getattr(package.estimator, "classes_", None)
        if classes is None:
            # sklearn Pipeline exposes classes_ in supported versions; retain a
            # safe fallback for compatible wrappers.
            classes = package.estimator[-1].classes_
        return RawModelOutput(
            episode_id=request.episode_id,
            model_id=package.manifest.model_id,
            model_version=package.manifest.model_version,
            output_semantics=package.manifest.output_semantics,
            class_values={str(name): float(value) for name, value in zip(classes, probabilities, strict=True)},
            metadata={
                "provider": "SatSimScientificModelPackage",
                "artifact_sha256": package.artifact_sha256,
                "formal_model_binding_eligible": package.formal_model_binding_eligible,
                "claim_boundary": package.claim_boundary,
            },
        )

    provider = CallableRawModelProvider(
        manifest=package.manifest,
        predict_fn=predict_fn,
        requires_admission=True,
        model_artifact_sha256=package.artifact_sha256,
    )
    contracts = build_flat_feature_mapping_contract_registry(
        provider=provider,
        model_artifact_sha256=package.artifact_sha256,
        feature_order=package.feature_order,
    )
    registry = DataDrivenModelRegistry(
        admission_registry=contracts,
        admission_signer=admission_signer or AdmissionSigner.from_environment_or_ephemeral(),
        require_admission=True,
    )
    registry.register(provider)
    return registry


__all__ = [
    "SatSimScientificModelPackage",
    "load_sat_sim_scientific_model_package",
    "build_registry_from_sat_sim_package",
]
