"""Operator registry — maps operator families to concrete zero-fit operators.

The registry binds the JSON feature contracts (per-archetype primitive lists)
to the runtime operator implementations. It is read-only and deterministic; no
fitting state is held.
"""

from __future__ import annotations

import json
from pathlib import Path

from .balance import (
    ConservationResidualOperator,
    FlowConservationResidualOperator,
    IntegralBalanceResidualOperator,
    InventoryFlowBalanceOperator,
    LaggedCoherenceOperator,
    LumpedBalanceResidualOperator,
    ReferenceInnovationResidualOperator,
    RobustInputOutputRatioOperator,
)
from .contracts import FeatureOperator
from .robust_statistics import (
    BoundaryDwellProfileOperator,
    DynamicConsistencyResidualOperator,
    MultiscaleVariationOperator,
    PlateauAndClippingProfileOperator,
    RegimeConditionalSlopeOperator,
    RobustCommandResponseGainOperator,
    RobustLevelDistributionOperator,
    RobustLocalTrendOperator,
    SignedSlopeDistributionOperator,
)
from .temporal import (
    AutocorrelationDecayOperator,
    BurstLossRunProfileOperator,
    MissingRunLengthProfileOperator,
    MultiscaleChangeScoreOperator,
    RegimeChangeProfileOperator,
)

# Mapping: operator_family -> concrete operator instance.
_OPERATOR_FAMILIES: dict[str, FeatureOperator] = {
    "ROBUST_LEVEL_DISTRIBUTION": RobustLevelDistributionOperator(),
    "SIGNED_SLOPE_DISTRIBUTION": SignedSlopeDistributionOperator(),
    "BOUNDARY_DWELL_PROFILE": BoundaryDwellProfileOperator(),
    "ROBUST_LOCAL_TREND": RobustLocalTrendOperator(),
    "MULTISCALE_VARIATION": MultiscaleVariationOperator(),
    "ROBUST_COMMAND_RESPONSE_GAIN": RobustCommandResponseGainOperator(),
    "DYNAMIC_CONSISTENCY_RESIDUAL": DynamicConsistencyResidualOperator(),
    "REGIME_CONDITIONAL_SLOPE": RegimeConditionalSlopeOperator(),
    "PLATEAU_AND_CLIPPING_PROFILE": PlateauAndClippingProfileOperator(),
    "MULTISCALE_CHANGE_SCORE": MultiscaleChangeScoreOperator(),
    "MISSING_RUN_LENGTH_PROFILE": MissingRunLengthProfileOperator(),
    "AUTOCORRELATION_DECAY": AutocorrelationDecayOperator(),
    "BURST_LOSS_RUN_PROFILE": BurstLossRunProfileOperator(),
    "REGIME_CHANGE_PROFILE": RegimeChangeProfileOperator(),
    "INTEGRAL_BALANCE_RESIDUAL": IntegralBalanceResidualOperator(),
    "ROBUST_INPUT_OUTPUT_RATIO": RobustInputOutputRatioOperator(),
    "LAGGED_COHERENCE": LaggedCoherenceOperator(),
    "CONSERVATION_RESIDUAL": ConservationResidualOperator(),
    "INVENTORY_FLOW_BALANCE": InventoryFlowBalanceOperator(),
    "REFERENCE_INNOVATION_RESIDUAL": ReferenceInnovationResidualOperator(),
    "LUMPED_BALANCE_RESIDUAL": LumpedBalanceResidualOperator(),
    "FLOW_CONSERVATION_RESIDUAL": FlowConservationResidualOperator(),
}


class FeatureOperatorRegistry:
    """Deterministic, read-only registry of zero-fit operators."""

    def __init__(self) -> None:
        self._operators = dict(_OPERATOR_FAMILIES)

    def get(self, family: str) -> FeatureOperator:
        try:
            return self._operators[family]
        except KeyError as exc:
            raise KeyError(f"no operator registered for family: {family}") from exc

    def has(self, family: str) -> bool:
        return family in self._operators

    def families(self) -> list[str]:
        return sorted(self._operators)

    def load_feature_contract_catalog(self, path: str | Path) -> dict[str, list[str]]:
        """Return archetype_id -> list of (primitive_id, operator_family) tuples.

        This parses the JSON feature contract catalog, so the operator runtime
        is driven by the same frozen config the contracts reference. Missing
        families are reported, never silently ignored.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        result: dict[str, list[tuple[str, str]]] = {}
        missing: list[tuple[str, str]] = []
        for contract in data.get("archetype_contracts", []):
            archetype = contract["component_archetype_id"]
            primitives: list[tuple[str, str]] = []
            for prim in contract.get("primitives", []):
                family = prim["operator_family"]
                if not self.has(family):
                    missing.append((contract["contract_id"], family))
                primitives.append((prim["primitive_id"], family))
            result[archetype] = primitives
        if missing:
            raise ValueError(
                f"feature contract references unregistered operator families: {missing}"
            )
        return result


_DEFAULT_REGISTRY = FeatureOperatorRegistry()


def default_registry() -> FeatureOperatorRegistry:
    return _DEFAULT_REGISTRY


__all__ = ["FeatureOperatorRegistry", "default_registry"]
