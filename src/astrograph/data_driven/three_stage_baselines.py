"""Replaceable, label-free baselines for the three-stage diagnosis contract.

These models intentionally expose no ``fit`` or threshold-search method.  All
centres, scales, weights and decision thresholds must be selected on the
development/calibration partitions and injected as frozen configuration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import exp, isfinite
from types import MappingProxyType
from typing import Protocol

from .three_stage_protocol import (
    UNKNOWN,
    CausalStreamContract,
    CausalTelemetrySample,
    DetectionAlarm,
    HierarchicalClassificationPrediction,
    HierarchicalFaultOntology,
    LocalizationDecision,
    LocalizationPrediction,
)


class CausalDetector(Protocol):
    def reset(self) -> None: ...

    def update(self, sample: CausalTelemetrySample) -> DetectionAlarm | None: ...


class SourceLocalizer(Protocol):
    def localize(self, evidence: Sequence["ComponentResidualEvidence"]) -> LocalizationPrediction: ...


class ComponentConditionedClassifier(Protocol):
    def classify(
        self, conditioned_component: str, residual_features: Sequence[float]
    ) -> HierarchicalClassificationPrediction: ...


def _freeze_float_mapping(values: Mapping[str, float]) -> Mapping[str, float]:
    copied = {str(key): float(value) for key, value in values.items()}
    if any(not isfinite(value) for value in copied.values()):
        raise ValueError("configuration values must be finite")
    return MappingProxyType(copied)


@dataclass(frozen=True)
class CausalResidualCusumConfig:
    channel_center: Mapping[str, float]
    channel_scale: Mapping[str, float]
    drift: float
    alarm_threshold: float
    refractory_period_s: float = 0.0
    detector_id: str = "causal-residual-cusum-v1"
    suspected_component: str | None = None

    def __post_init__(self) -> None:
        centers = _freeze_float_mapping(self.channel_center)
        scales = _freeze_float_mapping(self.channel_scale)
        if not centers or set(centers) != set(scales):
            raise ValueError("channel center and scale must cover the same non-empty allowlist")
        if any(value <= 0.0 for value in scales.values()):
            raise ValueError("channel scales must be positive")
        if self.drift < 0.0 or self.alarm_threshold <= 0.0 or self.refractory_period_s < 0.0:
            raise ValueError("CUSUM drift/refractory must be non-negative and threshold positive")
        if not self.detector_id.strip():
            raise ValueError("detector_id must be non-empty")
        object.__setattr__(self, "channel_center", centers)
        object.__setattr__(self, "channel_scale", scales)


class CausalResidualCusumDetector:
    """One-sided CUSUM over the maximum robust standardized residual."""

    def __init__(self, config: CausalResidualCusumConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._stream = CausalStreamContract()
        self._cusum = 0.0
        self._next_alarm_time_s = 0.0

    def update(self, sample: CausalTelemetrySample) -> DetectionAlarm | None:
        self._stream.accept(sample)
        missing = set(self.config.channel_center).difference(sample.telemetry)
        if missing:
            raise ValueError(f"causal detector input is missing calibrated channels: {sorted(missing)}")
        standardized = max(
            abs(sample.telemetry[channel] - center) / self.config.channel_scale[channel]
            for channel, center in self.config.channel_center.items()
        )
        self._cusum = max(0.0, self._cusum + standardized - self.config.drift)
        if (
            self._cusum < self.config.alarm_threshold
            or sample.timestamp_s < self._next_alarm_time_s
        ):
            return None
        alarm = DetectionAlarm(
            timestamp_s=sample.timestamp_s,
            score=self._cusum,
            detector_id=self.config.detector_id,
            suspected_component=self.config.suspected_component,
        )
        self._cusum = 0.0
        self._next_alarm_time_s = sample.timestamp_s + self.config.refractory_period_s
        return alarm


@dataclass(frozen=True)
class ComponentResidualEvidence:
    component: str
    source_local_residual: float
    command_response_residual: float
    propagation_residual: float

    def __post_init__(self) -> None:
        if not self.component.strip():
            raise ValueError("component must be non-empty")
        if any(
            not isfinite(value)
            for value in (
                self.source_local_residual,
                self.command_response_residual,
                self.propagation_residual,
            )
        ):
            raise ValueError("component residual evidence must be finite")


@dataclass(frozen=True)
class ResidualSourceLocalizerConfig:
    source_local_weight: float = 1.0
    command_response_weight: float = 0.5
    propagation_penalty: float = 0.25
    affected_threshold: float = 1.0
    minimum_source_margin: float = 0.2

    def __post_init__(self) -> None:
        values = (
            self.source_local_weight,
            self.command_response_weight,
            self.propagation_penalty,
            self.affected_threshold,
            self.minimum_source_margin,
        )
        if any(not isfinite(value) for value in values):
            raise ValueError("localizer configuration must be finite")
        if self.source_local_weight < 0 or self.command_response_weight < 0:
            raise ValueError("source evidence weights must be non-negative")
        if self.propagation_penalty < 0 or self.affected_threshold < 0:
            raise ValueError("propagation penalty and affected threshold must be non-negative")
        if self.minimum_source_margin < 0:
            raise ValueError("minimum_source_margin must be non-negative")


class ResidualSourceLocalizer:
    """Rank source-local evidence while treating propagation as a confounder."""

    def __init__(self, config: ResidualSourceLocalizerConfig) -> None:
        self.config = config

    def localize(self, evidence: Sequence[ComponentResidualEvidence]) -> LocalizationPrediction:
        if not evidence:
            return LocalizationPrediction((), decision=LocalizationDecision.ABSTAIN)
        if len({item.component for item in evidence}) != len(evidence):
            raise ValueError("component evidence must contain one row per component")
        scores = {
            item.component: (
                self.config.source_local_weight * max(0.0, item.source_local_residual)
                + self.config.command_response_weight * max(0.0, item.command_response_residual)
                - self.config.propagation_penalty * max(0.0, item.propagation_residual)
            )
            for item in evidence
        }
        ranking = tuple(sorted(scores, key=lambda component: (-scores[component], component)))
        margin = scores[ranking[0]] - scores[ranking[1]] if len(ranking) > 1 else float("inf")
        decision = (
            LocalizationDecision.IDENTIFIED
            if margin >= self.config.minimum_source_margin
            else LocalizationDecision.AMBIGUOUS
        )
        propagation = {item.component: item.propagation_residual for item in evidence}
        affected = frozenset(
            component
            for component, value in propagation.items()
            if value >= self.config.affected_threshold and component != ranking[0]
        )
        return LocalizationPrediction(ranking, affected, decision)


@dataclass(frozen=True)
class ComponentLinearHead:
    """Frozen local-mode logits for one component."""

    mode_weights: Mapping[str, tuple[float, ...]]
    mode_bias: Mapping[str, float]
    unknown_probability_threshold: float

    def __post_init__(self) -> None:
        weights = {str(mode): tuple(float(value) for value in row) for mode, row in self.mode_weights.items()}
        bias = _freeze_float_mapping(self.mode_bias)
        if not weights or set(weights) != set(bias):
            raise ValueError("mode weights and bias must cover the same non-empty mode set")
        dimensions = {len(row) for row in weights.values()}
        if len(dimensions) != 1 or next(iter(dimensions)) == 0:
            raise ValueError("all mode weight vectors must share one positive dimension")
        if any(not isfinite(value) for row in weights.values() for value in row):
            raise ValueError("mode weights must be finite")
        if not 0.0 <= self.unknown_probability_threshold <= 1.0:
            raise ValueError("unknown_probability_threshold must lie in [0, 1]")
        object.__setattr__(self, "mode_weights", MappingProxyType(weights))
        object.__setattr__(self, "mode_bias", bias)


class ComponentConditionedLinearClassifier:
    """Small frozen linear head with component-local softmax and UNKNOWN gate."""

    def __init__(
        self,
        ontology: HierarchicalFaultOntology,
        heads: Mapping[str, ComponentLinearHead],
    ) -> None:
        if set(heads) != set(ontology.component_families):
            raise ValueError("linear heads must cover exactly the frozen ontology components")
        for component, head in heads.items():
            ontology_modes = {
                mode for modes in ontology.component_families[component].values() for mode in modes
            }
            if set(head.mode_weights) != ontology_modes:
                raise ValueError(f"linear head modes do not match ontology for {component}")
        self.ontology = ontology
        self.heads = MappingProxyType(dict(heads))

    def classify(
        self, conditioned_component: str, residual_features: Sequence[float]
    ) -> HierarchicalClassificationPrediction:
        if conditioned_component not in self.heads:
            raise ValueError("conditioned component has no frozen classification head")
        features = tuple(float(value) for value in residual_features)
        if any(not isfinite(value) for value in features):
            raise ValueError("residual features must be finite")
        head = self.heads[conditioned_component]
        expected_dimension = len(next(iter(head.mode_weights.values())))
        if len(features) != expected_dimension:
            raise ValueError("residual feature dimension does not match frozen linear head")
        logits = {
            mode: sum(weight * value for weight, value in zip(weights, features, strict=True))
            + head.mode_bias[mode]
            for mode, weights in head.mode_weights.items()
        }
        maximum = max(logits.values())
        exponentials = {mode: exp(value - maximum) for mode, value in logits.items()}
        normalizer = sum(exponentials.values())
        mode_scores = {mode: value / normalizer for mode, value in exponentials.items()}
        predicted_mode = min(mode_scores, key=lambda mode: (-mode_scores[mode], mode))
        predicted_probability = mode_scores[predicted_mode]
        unknown_score = 1.0 - predicted_probability
        family_scores = {
            family: sum(mode_scores[mode] for mode in modes)
            for family, modes in self.ontology.component_families[conditioned_component].items()
        }
        if predicted_probability < head.unknown_probability_threshold:
            return HierarchicalClassificationPrediction(
                conditioned_component,
                UNKNOWN,
                UNKNOWN,
                abstained=True,
                family_scores=family_scores,
                local_mode_scores=mode_scores,
                unknown_score=unknown_score,
            )
        predicted_family = next(
            family
            for family, modes in self.ontology.component_families[conditioned_component].items()
            if predicted_mode in modes
        )
        return HierarchicalClassificationPrediction(
            conditioned_component,
            predicted_family,
            predicted_mode,
            family_scores=family_scores,
            local_mode_scores=mode_scores,
            unknown_score=unknown_score,
        )


__all__ = [
    "CausalDetector",
    "CausalResidualCusumConfig",
    "CausalResidualCusumDetector",
    "ComponentConditionedClassifier",
    "ComponentConditionedLinearClassifier",
    "ComponentLinearHead",
    "ComponentResidualEvidence",
    "ResidualSourceLocalizer",
    "ResidualSourceLocalizerConfig",
    "SourceLocalizer",
]
