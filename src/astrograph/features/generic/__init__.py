"""Generic zero-fit feature-operator runtime for the diagnosis pipeline."""

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
from .contracts import (
    EvidenceToken,
    FeatureInput,
    FeatureOperator,
    OperatorResult,
    TokenResult,
)
from .registry import FeatureOperatorRegistry, default_registry
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
from .token_builder import TokenBuilder

__all__ = [
    "AutocorrelationDecayOperator",
    "BoundaryDwellProfileOperator",
    "BurstLossRunProfileOperator",
    "ConservationResidualOperator",
    "DynamicConsistencyResidualOperator",
    "EvidenceToken",
    "FeatureInput",
    "FeatureOperator",
    "FeatureOperatorRegistry",
    "FlowConservationResidualOperator",
    "IntegralBalanceResidualOperator",
    "InventoryFlowBalanceOperator",
    "LaggedCoherenceOperator",
    "LumpedBalanceResidualOperator",
    "MissingRunLengthProfileOperator",
    "MultiscaleChangeScoreOperator",
    "MultiscaleVariationOperator",
    "OperatorResult",
    "PlateauAndClippingProfileOperator",
    "ReferenceInnovationResidualOperator",
    "RegimeChangeProfileOperator",
    "RegimeConditionalSlopeOperator",
    "RobustCommandResponseGainOperator",
    "RobustInputOutputRatioOperator",
    "RobustLevelDistributionOperator",
    "RobustLocalTrendOperator",
    "SignedSlopeDistributionOperator",
    "TokenBuilder",
    "TokenResult",
    "default_registry",
]
