from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

from astrograph.simulation.models import SimulationRunCompleted

from .configured_runtime import ConfiguredRuntimeEvidenceBuilder
from .soft_semantic import build_soft_evidence_bundle, load_frozen_soft_evidence_policy

LEGACY_WHOLE_SPACECRAFT_EVIDENCE_BUILDER_ID = "legacy.whole_spacecraft.v4"
CONFIGURED_EVIDENCE_BUILDER_ID = "configured.package.v1"


class RuntimeEvidenceBuilder(Protocol):
    builder_version: str

    def build(self, event: SimulationRunCompleted) -> tuple[Any, Any]: ...


LegacyBuilderFactory = Callable[[str | Path], RuntimeEvidenceBuilder]


class RuntimeEvidenceBuildService:
    """Resolve a configured evidence adapter or an explicitly selected compat adapter."""

    def __init__(self, *, legacy_builder_factory: LegacyBuilderFactory | None = None) -> None:
        self.legacy_builder_factory = legacy_builder_factory
        self._configured_builders: dict[str, ConfiguredRuntimeEvidenceBuilder] = {}
        self._legacy_builders: dict[str, RuntimeEvidenceBuilder] = {}

    def resolve(self, payload: dict[str, Any]) -> RuntimeEvidenceBuilder:
        builder_id = payload.get("runtime_evidence_builder_id")
        package_value = payload.get("evidence_package_path")
        if builder_id == CONFIGURED_EVIDENCE_BUILDER_ID:
            if not package_value:
                raise ValueError("configured runtime evidence requires evidence_package_path")
            package_path = str(Path(str(package_value)).resolve())
            builder = self._configured_builders.get(package_path)
            if builder is None:
                builder = ConfiguredRuntimeEvidenceBuilder(package_path)
                self._configured_builders[package_path] = builder
            return builder
        if builder_id == LEGACY_WHOLE_SPACECRAFT_EVIDENCE_BUILDER_ID:
            if package_value:
                raise ValueError("legacy runtime evidence must not receive evidence_package_path")
            if self.legacy_builder_factory is None:
                raise RuntimeError("legacy runtime evidence adapter is not installed in this runtime")
            protocol_root = str(Path(str(payload["protocol_root"])).resolve())
            builder = self._legacy_builders.get(protocol_root)
            if builder is None:
                builder = self.legacy_builder_factory(protocol_root)
                self._legacy_builders[protocol_root] = builder
            return builder
        raise ValueError("runtime_evidence_builder_id must explicitly select a registered evidence builder")

    def build(self, payload: dict[str, Any]) -> dict[str, Any]:
        builder = self.resolve(payload)
        event = SimulationRunCompleted.model_validate(payload["simulation_event"])
        bundle, gate = builder.build(event)
        policy_path = payload.get("soft_evidence_calibration_policy_path")
        policy_sha256 = payload.get("soft_evidence_calibration_policy_sha256")
        if bool(policy_path) != bool(policy_sha256):
            raise ValueError(
                "soft evidence calibration policy path and SHA-256 must be provided together"
            )
        policy = (
            load_frozen_soft_evidence_policy(
                str(policy_path), expected_sha256=str(policy_sha256)
            )
            if policy_path
            else None
        )
        soft_evidence = build_soft_evidence_bundle(bundle, policy)
        return {
            "bundle": bundle.model_dump(mode="json"),
            "soft_evidence": soft_evidence.model_dump(mode="json"),
            "data_gate_report": gate.model_dump(mode="json"),
            "builder_version": builder.builder_version,
        }
