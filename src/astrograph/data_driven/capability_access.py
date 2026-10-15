from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CapabilityAccessDecision(BaseModel):
    """C5: four-dimension capability access assessment for a model candidate."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-capability-access.v1'
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    dimensions: dict[str, dict] = Field(default_factory=dict)
    overall: str = Field(pattern=r'^(ADMITTED|REJECTED)$')
    assessed_at: str = Field(default_factory=lambda: time.strftime('%Y-%m-%dT%H:%M:%S+08:00'))


def assess_capability_access(
    model_id: str, model_version: str,
    *,
    data_coverage: float,
    robustness: float,
    explainability: float,
    governance_compliance: bool,
    minimum_coverage: float = 0.5,
    minimum_robustness: float = 0.5,
    minimum_explainability: float = 0.5,
) -> CapabilityAccessDecision:
    """Four-dimension admission gate (MoMA-inspired).

    Dimensions: data coverage (frozen-split coverage), robustness (CV stability),
    explainability (evidence/case base), governance compliance (deny/gate/registry).
    Admitted only when all three score dims meet minima AND governance is compliant.

    """
    dimensions = {
        'data_coverage': {'score': round(data_coverage, 4), 'passed': data_coverage >= minimum_coverage},
        'robustness': {'score': round(robustness, 4), 'passed': robustness >= minimum_robustness},
        'explainability': {'score': round(explainability, 4), 'passed': explainability >= minimum_explainability},
        'governance_compliance': {'passed': governance_compliance},
    }
    admitted = all(d['passed'] for d in dimensions.values())
    return CapabilityAccessDecision(
        model_id=model_id, model_version=model_version,
        dimensions=dimensions,
        overall='ADMITTED' if admitted else 'REJECTED',
    )


__all__ = ['CapabilityAccessDecision', 'assess_capability_access']