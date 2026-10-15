"""Strict dataset-class to runtime-candidate mapping for rank fusion."""

from __future__ import annotations

from dataclasses import dataclass


class NoMappedFaultMassError(ValueError):
    """Raised when a prediction cannot cast an audited runtime fault vote."""


@dataclass(frozen=True)
class CandidateMappingResult:
    mapped_attribution: dict[str, float]
    unsupported_probabilities: dict[str, float]
    mapped_mass: float
    unsupported_mass: float


def map_runtime_fault_attribution(
    probabilities: dict[str, float],
    class_map: dict[str, str],
    runtime_fault_ids: set[str],
) -> CandidateMappingResult:
    """Map only exact identities and reject a zero-mass runtime vote.

    Unsupported probability remains explicit and becomes an OOD signal.  It is
    never redistributed over the runtime candidate set.
    """

    mapped = {fault_id: 0.0 for fault_id in sorted(runtime_fault_ids)}
    unsupported: dict[str, float] = {}
    for name, probability in probabilities.items():
        value = float(probability)
        target = class_map.get(name)
        if target in runtime_fault_ids:
            mapped[target] += value
        else:
            unsupported[name] = value
    mapped_mass = float(sum(mapped.values()))
    unsupported_mass = float(sum(unsupported.values()))
    if mapped_mass <= 0.0:
        raise NoMappedFaultMassError(
            "prediction has no probability mass on audited runtime fault candidates"
        )
    return CandidateMappingResult(mapped, unsupported, mapped_mass, unsupported_mass)


__all__ = [
    "CandidateMappingResult", "NoMappedFaultMassError", "map_runtime_fault_attribution",
]
