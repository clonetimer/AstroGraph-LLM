
"""Development-selectable contextual reliability policies for KG branches.

No policy in this module is selected merely by existing post-hoc results.  The
module exposes candidate families and produces auditable, label-free runtime
multipliers from pre-frozen capability and calibration receipts.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, Mapping

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BANNED = {"ground_truth", "true_label", "fault_label", "y_true", "target", "correctness", "accuracy"}

class ReliabilityAggregator(str, Enum):
    BOTTLENECK = "BOTTLENECK"
    GEOMETRIC_MEAN = "GEOMETRIC_MEAN"
    WEIGHTED_GEOMETRIC_MEAN = "WEIGHTED_GEOMETRIC_MEAN"

class PolicyStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    DEVELOPMENT_SELECTED = "DEVELOPMENT_SELECTED"
    FROZEN = "FROZEN"

@dataclass(frozen=True)
class ReliabilityComponents:
    executable_coverage: float
    contextual_applicability: float
    development_validation: float
    provenance_independence: float
    calibration_reliability: float
    conflict_reliability: float
    drift_compatibility: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")

@dataclass(frozen=True)
class ContextualEvidencePolicy:
    schema: str
    policy_id: str
    status: PolicyStatus
    aggregator: ReliabilityAggregator
    component_weights: Mapping[str, float]
    hard_floor: float
    selected_on_groups: tuple[int, ...] = ()
    forbidden_groups: tuple[int, ...] = (4, 9)
    selection_receipt_sha256: str | None = None
    policy_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema != "astrograph-contextual-evidence-policy.v1":
            raise ValueError("unsupported policy schema")
        if not self.policy_id.strip():
            raise ValueError("policy_id required")
        if not 0.0 <= self.hard_floor <= 1.0:
            raise ValueError("hard_floor must be in [0, 1]")
        if set(self.selected_on_groups) & set(self.forbidden_groups):
            raise ValueError("forbidden post-hoc groups cannot select a policy")
        expected_keys = set(asdict(ReliabilityComponents(1,1,1,1,1,1,1)))
        if set(self.component_weights) != expected_keys:
            raise ValueError("component_weights must name every reliability component exactly once")
        if any((not math.isfinite(v) or v < 0) for v in self.component_weights.values()):
            raise ValueError("component weights must be finite and non-negative")
        if self.aggregator is ReliabilityAggregator.WEIGHTED_GEOMETRIC_MEAN and sum(self.component_weights.values()) <= 0:
            raise ValueError("weighted geometric mean requires positive total weight")
        if self.status is not PolicyStatus.CANDIDATE:
            if not self.selected_on_groups or self.selection_receipt_sha256 is None:
                raise ValueError("selected/frozen policy requires development groups and receipt")
        if self.selection_receipt_sha256 is not None and not _SHA256.fullmatch(self.selection_receipt_sha256):
            raise ValueError("selection_receipt_sha256 must be lowercase SHA-256")
        expected = self.compute_sha256()
        if self.policy_sha256 is not None and self.policy_sha256 != expected:
            raise ValueError("policy_sha256 mismatch")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "status": self.status.value,
            "aggregator": self.aggregator.value,
            "component_weights": dict(sorted(self.component_weights.items())),
            "hard_floor": self.hard_floor,
            "selected_on_groups": list(self.selected_on_groups),
            "forbidden_groups": list(self.forbidden_groups),
            "selection_receipt_sha256": self.selection_receipt_sha256,
        }

    def compute_sha256(self) -> str:
        raw = json.dumps(self.payload(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def multiplier(self, components: ReliabilityComponents, runtime_metadata: Mapping[str, Any] | None = None) -> float:
        _reject_runtime_truth(runtime_metadata or {})
        values = asdict(components)
        if any(v < self.hard_floor for v in values.values()):
            return 0.0
        if self.aggregator is ReliabilityAggregator.BOTTLENECK:
            return min(values.values())
        if self.aggregator is ReliabilityAggregator.GEOMETRIC_MEAN:
            product = math.prod(max(v, 1e-15) for v in values.values())
            return product ** (1.0 / len(values))
        total = sum(self.component_weights.values())
        log_value = sum(self.component_weights[k] * math.log(max(values[k], 1e-15)) for k in values) / total
        return math.exp(log_value)

    def effective_weight(self, configured_weight: float, components: ReliabilityComponents, runtime_metadata: Mapping[str, Any] | None = None) -> float:
        if not math.isfinite(configured_weight) or configured_weight < 0:
            raise ValueError("configured_weight must be finite and non-negative")
        return configured_weight * self.multiplier(components, runtime_metadata)

def _reject_runtime_truth(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).strip().lower() in _BANNED:
                raise ValueError(f"runtime metadata contains forbidden truth/outcome key: {key}")
            _reject_runtime_truth(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_runtime_truth(child)
