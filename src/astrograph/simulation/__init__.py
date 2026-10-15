from __future__ import annotations

from typing import Any

from .models import OnlineDataGateReport, SimulationRunCompleted
from .datasets import SyntheticDatasetSpec, generate_synthetic_dataset
from .fault_patterns import FaultPatternSpec, FaultPatternType

__all__ = [
    "OnlineDataGateReport",
    "SimulationRunCompleted",
    "SyntheticDatasetSpec",
    "generate_synthetic_dataset",
    "FaultPatternSpec",
    "FaultPatternType",
    # Legacy compatibility exports, late-loaded below.
    "WholeSpacecraftRuntimeEvidenceBuilder",
    "validate_completed_simulation_run",
]


def __getattr__(name: str) -> Any:
    if name in {"WholeSpacecraftRuntimeEvidenceBuilder", "validate_completed_simulation_run"}:
        from .runtime_adapter import WholeSpacecraftRuntimeEvidenceBuilder, validate_completed_simulation_run
        value = {
            "WholeSpacecraftRuntimeEvidenceBuilder": WholeSpacecraftRuntimeEvidenceBuilder,
            "validate_completed_simulation_run": validate_completed_simulation_run,
        }[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

