"""Spacecraft-scope convergence gate for PATCH-R32.

This gate separates two questions that must not be conflated:

1. Is the current AstroGraph spacecraft engineering software/function set stable
   enough to freeze core feature expansion and hand off to Bench/UI work?
2. Is every diagnostic branch independently externally qualified for a broad
   performance/generalization claim?

The first may pass while the second remains blocked by missing genuinely-new
holdout evidence.  The gate never upgrades benchmark/development evidence into
flight, safety, probability-calibration, or independent-validation authority.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConvergenceDisposition(str, Enum):
    PASS = "PASS"
    CONDITIONAL = "CONDITIONAL"
    BLOCKED = "BLOCKED"


class SpacecraftDatasetConvergence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    domain_role: str
    functional_status: str
    evidence_status: str
    current_algorithm: str
    observed_metrics: dict[str, Any] = Field(default_factory=dict)
    operational_limitations: tuple[str, ...] = ()
    qualification_blockers: tuple[str, ...] = ()
    source_artifacts: tuple[str, ...] = ()


class SpacecraftConvergenceDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-spacecraft-scope-convergence-decision.v1"
    release_scope: str = "SPACECRAFT_ENGINEERING"
    software_feature_freeze: ConvergenceDisposition
    diagnostic_external_qualification: ConvergenceDisposition
    stable_engineering_candidate: ConvergenceDisposition
    next_phase: str
    core_feature_expansion_frozen: bool
    dataset_decisions: tuple[SpacecraftDatasetConvergence, ...]
    core_functional_blockers: tuple[str, ...]
    external_qualification_blockers: tuple[str, ...]
    roadmap_only_items: tuple[str, ...]
    claim_boundary: tuple[str, ...]
    decision_sha256: str

    @model_validator(mode="after")
    def verify_hash(self) -> "SpacecraftConvergenceDecision":
        payload = self.model_dump(mode="json", exclude={"decision_sha256"})
        expected = canonical_sha256(payload)
        if self.decision_sha256 != expected:
            raise ValueError("SPACECRAFT_CONVERGENCE_DECISION_SHA256_INVALID")
        return self


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _status_is_functionally_available(status: str) -> bool:
    return status in {
        "STRONG_ENGINEERING_BENCHMARK",
        "ENGINEERING_CANDIDATE_AVAILABLE",
        "FROZEN_CANDIDATE_AVAILABLE",
        "MECHANISM_DIAGNOSIS_AVAILABLE",
        "GRAPH_ASSET_AVAILABLE",
        "SUPPORTED",
    }


def evaluate_spacecraft_scope_convergence(config: Mapping[str, Any]) -> SpacecraftConvergenceDecision:
    if config.get("schema_version") != "astrograph-spacecraft-scope-convergence-config.v1":
        raise ValueError("SPACECRAFT_CONVERGENCE_CONFIG_SCHEMA_INVALID")

    datasets = tuple(SpacecraftDatasetConvergence.model_validate(row) for row in config.get("datasets", ()))
    required_ids = tuple(str(x) for x in config.get("required_spacecraft_dataset_ids", ()))
    by_id = {row.dataset_id: row for row in datasets}
    missing = [item for item in required_ids if item not in by_id]
    if missing:
        raise ValueError("SPACECRAFT_CONVERGENCE_REQUIRED_DATASET_MISSING:" + ",".join(missing))

    core_blockers: list[str] = []
    external_blockers: list[str] = []
    for item in datasets:
        if item.dataset_id in required_ids and not _status_is_functionally_available(item.functional_status):
            core_blockers.append(f"{item.dataset_id}:{item.functional_status}")
        external_blockers.extend(f"{item.dataset_id}:{reason}" for reason in item.qualification_blockers)

    required_capabilities = config.get("required_capabilities", {})
    for capability, status in sorted(required_capabilities.items()):
        if status != "AVAILABLE":
            core_blockers.append(f"CAPABILITY:{capability}:{status}")

    software = ConvergenceDisposition.PASS if not core_blockers else ConvergenceDisposition.BLOCKED
    external = ConvergenceDisposition.PASS if not external_blockers else ConvergenceDisposition.BLOCKED

    # A stable engineering software candidate is intentionally not the same as
    # broad external diagnostic qualification.  It may be frozen when core
    # spacecraft functionality is present and the remaining gaps are explicitly
    # evidence/operational-qualification gaps rather than missing functions.
    stable = ConvergenceDisposition.PASS if software == ConvergenceDisposition.PASS else ConvergenceDisposition.BLOCKED
    next_phase = "ASTROGRAPH_BENCH_UI" if stable == ConvergenceDisposition.PASS else "SPACECRAFT_CORE_REMEDIATION"

    payload = {
        "schema_version": "astrograph-spacecraft-scope-convergence-decision.v1",
        "release_scope": "SPACECRAFT_ENGINEERING",
        "software_feature_freeze": software.value,
        "diagnostic_external_qualification": external.value,
        "stable_engineering_candidate": stable.value,
        "next_phase": next_phase,
        "core_feature_expansion_frozen": stable == ConvergenceDisposition.PASS,
        "dataset_decisions": [row.model_dump(mode="json") for row in datasets],
        "core_functional_blockers": core_blockers,
        "external_qualification_blockers": sorted(set(external_blockers)),
        "roadmap_only_items": list(config.get("roadmap_only_items", ())),
        "claim_boundary": list(config.get("claim_boundary", ())),
    }
    return SpacecraftConvergenceDecision(**payload, decision_sha256=canonical_sha256(payload))


def load_spacecraft_scope_convergence(path: str | Path) -> SpacecraftConvergenceDecision:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return evaluate_spacecraft_scope_convergence(payload)


__all__ = [
    "ConvergenceDisposition",
    "SpacecraftDatasetConvergence",
    "SpacecraftConvergenceDecision",
    "canonical_sha256",
    "evaluate_spacecraft_scope_convergence",
    "load_spacecraft_scope_convergence",
]
