"""Executable contracts for AstroGraph's three-stage diagnosis experiment.

This module deliberately contains no trainable model.  It defines the boundary
that every detector, localizer and classifier must obey before model selection:

* detection consumes one timestamp at a time and never receives fault windows;
* localization keeps the unique source separate from propagated effects;
* classification is conditioned on a component and follows a hierarchical,
  component-local ontology;
* cascade evaluation reports both oracle-upstream and predicted-upstream views.

Truth objects in this file are evaluator-only.  They must be created after the
inference outputs have been materialized.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any

from .evaluation_protocol import (
    binary_aupr,
    expected_calibration_error,
    multiclass_metrics,
    wilson_interval,
)


UNKNOWN = "UNKNOWN"
DEFAULT_FROZEN_ECE_BIN_EDGES = tuple(index / 10.0 for index in range(11))

# Exact normalized field names are denied.  Substring matching is intentionally
# avoided because legitimate physical signals can contain words such as
# ``fault_current_limit``.  Dataset adapters must first project an allowlist;
# this guard is the second line of defence.
_EVALUATOR_ONLY_FIELDS = frozenset(
    {
        "fault_active",
        "fault_value",
        "fault_id",
        "fault_case_id",
        "fault_start",
        "fault_end",
        "fault_onset",
        "fault_duration",
        "label",
        "labels",
        "truth",
        "target",
        "scenario",
        "split",
        "partition",
        "evaluation_metadata",
    }
)


def _normalise_field_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(".", "_")


@dataclass(frozen=True)
class CausalTelemetrySample:
    """One label-free observation made available at ``timestamp_s``."""

    timestamp_s: float
    telemetry: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("timestamp_s must be finite and non-negative")
        if not self.telemetry:
            raise ValueError("telemetry must contain at least one physical field")
        copied: dict[str, float] = {}
        for key, value in self.telemetry.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("telemetry field names must be non-empty strings")
            if _normalise_field_name(key) in _EVALUATOR_ONLY_FIELDS:
                raise ValueError(f"evaluator-only field reached inference: {key}")
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError(f"telemetry field {key!r} is not finite")
            copied[key] = numeric
        object.__setattr__(self, "telemetry", MappingProxyType(copied))


class CausalStreamContract:
    """Stateful guard proving that inference is consumed in causal order.

    It cannot prove that an upstream adapter did not precompute a future-aware
    feature.  For that reason the experiment manifest must also identify every
    feature's lookback and set ``lookahead_s=0``.  This guard catches direct
    truth leakage and out-of-order/future replay at the model boundary.
    """

    def __init__(self) -> None:
        self._last_timestamp_s: float | None = None

    @property
    def last_timestamp_s(self) -> float | None:
        return self._last_timestamp_s

    def accept(self, sample: CausalTelemetrySample) -> CausalTelemetrySample:
        if self._last_timestamp_s is not None and sample.timestamp_s <= self._last_timestamp_s:
            raise ValueError("causal stream timestamps must be strictly increasing")
        self._last_timestamp_s = sample.timestamp_s
        return sample


@dataclass(frozen=True)
class DetectionAlarm:
    """A detector output; no evaluator truth is carried in the object."""

    timestamp_s: float
    score: float
    detector_id: str
    suspected_component: str | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("alarm timestamp must be finite and non-negative")
        if not isfinite(self.score):
            raise ValueError("alarm score must be finite")
        if not self.detector_id.strip():
            raise ValueError("detector_id must be non-empty")


@dataclass(frozen=True)
class LocalizationTruth:
    """Evaluator-only single-fault source and propagation truth."""

    source_component: str
    affected_components: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.source_component.strip():
            raise ValueError("source_component must be non-empty")
        if self.source_component in self.affected_components:
            raise ValueError("source and affected components must be represented separately")


class LocalizationDecision(str, Enum):
    IDENTIFIED = "IDENTIFIED"
    AMBIGUOUS = "AMBIGUOUS"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class LocalizationPrediction:
    """Separate source ranking from affected-component prediction."""

    source_ranking: tuple[str, ...]
    affected_components: frozenset[str] = field(default_factory=frozenset)
    decision: LocalizationDecision = LocalizationDecision.IDENTIFIED

    def __post_init__(self) -> None:
        if len(set(self.source_ranking)) != len(self.source_ranking):
            raise ValueError("source_ranking must not contain duplicate components")
        if any(not value.strip() for value in self.source_ranking):
            raise ValueError("source_ranking contains an empty component")
        if self.decision == LocalizationDecision.IDENTIFIED and not self.source_ranking:
            raise ValueError("IDENTIFIED localization requires a source candidate")
        if self.top_source is not None and self.top_source in self.affected_components:
            raise ValueError(
                "the selected top source cannot simultaneously be labelled affected"
            )

    @property
    def top_source(self) -> str | None:
        return self.source_ranking[0] if self.source_ranking else None


def localization_metrics(
    truth: Sequence[LocalizationTruth],
    predicted: Sequence[LocalizationPrediction],
    *,
    top_k: int = 3,
) -> dict[str, Any]:
    """Evaluate root-cause isolation and propagation attribution separately."""

    if len(truth) != len(predicted):
        raise ValueError("truth and predicted must have equal length")
    if not truth:
        raise ValueError("at least one fault episode is required")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    top1_correct = 0
    topk_correct = 0
    reciprocal_ranks: list[float] = []
    affected_tp = affected_fp = affected_fn = 0
    affected_as_source = source_as_affected = 0
    abstained = 0
    identified_count = 0
    source_predictions: list[str] = []
    for expected, actual in zip(truth, predicted, strict=True):
        identified = actual.decision == LocalizationDecision.IDENTIFIED
        if not identified:
            abstained += 1
        else:
            identified_count += 1
        identified_top_source = actual.top_source if identified else None
        top1_correct += identified_top_source == expected.source_component
        source_predictions.append(identified_top_source or "<NO_IDENTIFIED_SOURCE>")
        try:
            rank = actual.source_ranking.index(expected.source_component) + 1
        except ValueError:
            rank = 0
        topk_correct += bool(rank and rank <= top_k)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        predicted_affected = set(actual.affected_components)
        expected_affected = set(expected.affected_components)
        affected_tp += len(predicted_affected & expected_affected)
        affected_fp += len(predicted_affected - expected_affected)
        affected_fn += len(expected_affected - predicted_affected)
        source_as_affected += expected.source_component in predicted_affected
        affected_as_source += actual.top_source in expected_affected

    n = len(truth)
    affected_precision = affected_tp / (affected_tp + affected_fp) if affected_tp + affected_fp else 0.0
    affected_recall = affected_tp / (affected_tp + affected_fn) if affected_tp + affected_fn else 0.0
    affected_f1 = (
        2 * affected_precision * affected_recall / (affected_precision + affected_recall)
        if affected_precision + affected_recall
        else 0.0
    )
    source_labels = sorted(
        {item.source_component for item in truth}.union(source_predictions)
    )
    source_classification = multiclass_metrics(
        [item.source_component for item in truth],
        source_predictions,
        labels=source_labels,
    )
    return {
        "n_fault_episodes": n,
        "source_top1_accuracy": top1_correct / n,
        "source_macro_f1": source_classification["macro_f1"],
        "source_balanced_accuracy": source_classification["balanced_accuracy"],
        f"source_top{top_k}_recall": topk_correct / n,
        "source_mrr": sum(reciprocal_ranks) / n,
        "affected_micro_precision": affected_precision,
        "affected_micro_recall": affected_recall,
        "affected_micro_f1": affected_f1,
        "source_as_affected_count": source_as_affected,
        "affected_as_top_source_count": affected_as_source,
        "abstention_rate": abstained / n,
        "identified_coverage": identified_count / n,
        "identified_selective_accuracy": (
            top1_correct / identified_count if identified_count else None
        ),
    }


def stratified_localization_metrics(
    truth: Sequence[LocalizationTruth],
    predicted: Sequence[LocalizationPrediction],
    *,
    strata: Mapping[str, Sequence[str]],
    top_k: int = 3,
) -> dict[str, dict[str, Any]]:
    """Apply the complete localization metric set to frozen strata."""

    if len(truth) != len(predicted):
        raise ValueError("truth and predicted must have equal length")
    result: dict[str, dict[str, Any]] = {}
    for axis, values in strata.items():
        if len(values) != len(truth):
            raise ValueError(f"stratum axis {axis!r} does not align with episodes")
        by_value: dict[str, Any] = {}
        for value in sorted(set(values)):
            indices = [index for index, item in enumerate(values) if item == value]
            by_value[value] = localization_metrics(
                [truth[index] for index in indices],
                [predicted[index] for index in indices],
                top_k=top_k,
            )
        result[axis] = by_value
    return result


@dataclass(frozen=True)
class HierarchicalFaultOntology:
    """Frozen component -> mechanism family -> local fault-mode ontology."""

    component_families: Mapping[str, Mapping[str, tuple[str, ...]]]

    def __post_init__(self) -> None:
        frozen: dict[str, Mapping[str, tuple[str, ...]]] = {}
        for component, families in self.component_families.items():
            if not component.strip() or not families:
                raise ValueError("each component must have at least one mechanism family")
            component_modes: set[str] = set()
            copied_families: dict[str, tuple[str, ...]] = {}
            for family, modes in families.items():
                copied_modes = tuple(modes)
                if not family.strip() or family == UNKNOWN or not copied_modes:
                    raise ValueError("mechanism families must be non-empty and cannot be UNKNOWN")
                if any(not mode.strip() or mode == UNKNOWN for mode in copied_modes):
                    raise ValueError("local fault modes must be non-empty and cannot be UNKNOWN")
                if len(set(copied_modes)) != len(copied_modes):
                    raise ValueError(f"duplicate mode inside {component}/{family}")
                overlap = component_modes.intersection(copied_modes)
                if overlap:
                    raise ValueError(f"local modes must identify one family within a component: {overlap}")
                component_modes.update(copied_modes)
                copied_families[family] = copied_modes
            frozen[component] = MappingProxyType(copied_families)
        object.__setattr__(self, "component_families", MappingProxyType(frozen))

    def contains(self, component: str, family: str, local_mode: str) -> bool:
        return local_mode in self.component_families.get(component, {}).get(family, ())


@dataclass(frozen=True)
class ClassificationTruth:
    component: str
    mechanism_family: str
    local_fault_mode: str
    canonical_fault_mode_id: str
    is_open_set: bool = False

    def validate(self, ontology: HierarchicalFaultOntology) -> None:
        if not self.canonical_fault_mode_id.strip():
            raise ValueError("canonical_fault_mode_id must be non-empty")
        if self.is_open_set:
            if self.component not in ontology.component_families:
                raise ValueError("open-set truth still requires a known conditioned component")
            if self.mechanism_family != UNKNOWN or self.local_fault_mode != UNKNOWN:
                raise ValueError("open-set truth must use UNKNOWN at both hierarchy levels")
            return
        if not ontology.contains(self.component, self.mechanism_family, self.local_fault_mode):
            raise ValueError("classification truth lies outside the frozen component-local ontology")


@dataclass(frozen=True)
class HierarchicalClassificationPrediction:
    """A classification made under an explicit component condition."""

    conditioned_component: str
    mechanism_family: str
    local_fault_mode: str
    abstained: bool = False
    available: bool = True
    family_scores: Mapping[str, float] | None = None
    local_mode_scores: Mapping[str, float] | None = None
    unknown_score: float | None = None

    def __post_init__(self) -> None:
        for field_name in ("family_scores", "local_mode_scores"):
            scores = getattr(self, field_name)
            if scores is None:
                continue
            copied = {str(key): float(value) for key, value in scores.items()}
            if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in copied.values()):
                raise ValueError(f"{field_name} probabilities must be finite and lie in [0, 1]")
            if copied and abs(sum(copied.values()) - 1.0) > 1e-6:
                raise ValueError(f"{field_name} probabilities must sum to one")
            object.__setattr__(self, field_name, MappingProxyType(copied))
        if self.unknown_score is not None and (
            not isfinite(self.unknown_score) or not 0.0 <= self.unknown_score <= 1.0
        ):
            raise ValueError("unknown_score must be finite and lie in [0, 1]")

    def validate(self, ontology: HierarchicalFaultOntology) -> None:
        if self.conditioned_component not in ontology.component_families:
            raise ValueError("prediction uses a component outside the frozen ontology")
        if not self.available:
            if not self.abstained or self.mechanism_family != UNKNOWN or self.local_fault_mode != UNKNOWN:
                raise ValueError("unavailable classification must be an UNKNOWN abstention sentinel")
            return
        if self.abstained:
            if self.mechanism_family != UNKNOWN or self.local_fault_mode != UNKNOWN:
                raise ValueError("abstention must emit UNKNOWN at both hierarchy levels")
            return
        if not ontology.contains(
            self.conditioned_component, self.mechanism_family, self.local_fault_mode
        ):
            raise ValueError("prediction lies outside the conditioned component ontology")
        if self.family_scores is not None:
            allowed_families = set(ontology.component_families[self.conditioned_component])
            if set(self.family_scores) != allowed_families:
                raise ValueError("family_scores must cover exactly the conditioned component families")
        if self.local_mode_scores is not None:
            allowed_modes = {
                mode
                for modes in ontology.component_families[self.conditioned_component].values()
                for mode in modes
            }
            if set(self.local_mode_scores) != allowed_modes:
                raise ValueError("local_mode_scores must cover exactly the conditioned component modes")


def hierarchical_classification_metrics(
    truth: Sequence[ClassificationTruth],
    predicted: Sequence[HierarchicalClassificationPrediction],
    *,
    ontology: HierarchicalFaultOntology,
    require_correct_component_context: bool = True,
    local_mode_top_k: int = 3,
    ece_bin_edges: Sequence[float] = DEFAULT_FROZEN_ECE_BIN_EDGES,
    known_canonical_mode_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Report family, local-mode and complete-path performance plus coverage."""

    if len(truth) != len(predicted):
        raise ValueError("truth and predicted must have equal length")
    if not truth:
        raise ValueError("at least one classified fault episode is required")
    if local_mode_top_k <= 0:
        raise ValueError("local_mode_top_k must be positive")
    for expected in truth:
        expected.validate(ontology)
    frozen_known_ids = (
        set(known_canonical_mode_ids)
        if known_canonical_mode_ids is not None
        else {item.canonical_fault_mode_id for item in truth if not item.is_open_set}
    )
    validate_open_set_partition(
        known_mode_ids=frozen_known_ids,
        open_set_mode_ids={item.canonical_fault_mode_id for item in truth if item.is_open_set},
    )
    for expected, actual in zip(truth, predicted, strict=True):
        actual.validate(ontology)
        if require_correct_component_context and actual.conditioned_component != expected.component:
            raise ValueError("oracle classification metric requires the truth component context")

    family_truth = [item.mechanism_family for item in truth]
    family_predicted = [item.mechanism_family for item in predicted]
    mode_truth = [f"{item.component}/{item.local_fault_mode}" for item in truth]
    mode_predicted = [f"{item.conditioned_component}/{item.local_fault_mode}" for item in predicted]
    path_correct = [
        actual.available
        and expected.component == actual.conditioned_component
        and expected.mechanism_family == actual.mechanism_family
        and expected.local_fault_mode == actual.local_fault_mode
        for expected, actual in zip(truth, predicted, strict=True)
    ]
    covered = [item.available and not item.abstained for item in predicted]
    covered_count = sum(covered)
    open_set_pairs = [
        (expected, actual)
        for expected, actual in zip(truth, predicted, strict=True)
        if expected.is_open_set
    ]
    known_pairs = [
        (expected, actual)
        for expected, actual in zip(truth, predicted, strict=True)
        if not expected.is_open_set
    ]
    local_mode_topk_correct = 0
    known_mode_count = len(known_pairs)
    calibration_correctness: list[bool] = []
    calibration_confidences: list[float] = []
    unknown_truth: list[bool] = []
    unknown_scores: list[float] = []
    for expected, actual in zip(truth, predicted, strict=True):
        if not expected.is_open_set:
            if actual.local_mode_scores is not None:
                ranking = sorted(
                    actual.local_mode_scores,
                    key=lambda mode: (-actual.local_mode_scores[mode], mode),
                )
            elif actual.available and not actual.abstained:
                ranking = [actual.local_fault_mode]
            else:
                ranking = []
            local_mode_topk_correct += (
                actual.conditioned_component == expected.component
                and expected.local_fault_mode in ranking[:local_mode_top_k]
            )
        confidence: float | None = None
        if not actual.available:
            confidence = None
        elif actual.abstained:
            confidence = actual.unknown_score
        elif actual.local_mode_scores is not None:
            confidence = actual.local_mode_scores.get(actual.local_fault_mode)
        if confidence is not None:
            calibration_correctness.append(
                expected.component == actual.conditioned_component
                and expected.mechanism_family == actual.mechanism_family
                and expected.local_fault_mode == actual.local_fault_mode
            )
            calibration_confidences.append(confidence)
        if actual.available and actual.unknown_score is not None:
            unknown_truth.append(expected.is_open_set)
            unknown_scores.append(actual.unknown_score)

    calibration = (
        expected_calibration_error(
            calibration_correctness,
            calibration_confidences,
            bin_edges=ece_bin_edges,
        )
        if calibration_confidences
        else None
    )
    return {
        "n_fault_episodes": len(truth),
        "family": multiclass_metrics(family_truth, family_predicted),
        "component_local_mode": multiclass_metrics(mode_truth, mode_predicted),
        f"component_local_mode_top{local_mode_top_k}_accuracy": (
            local_mode_topk_correct / known_mode_count if known_mode_count else None
        ),
        "hierarchical_exact_match": sum(path_correct) / len(path_correct),
        "coverage": covered_count / len(predicted),
        "selective_exact_match": (
            sum(ok for ok, is_covered in zip(path_correct, covered, strict=True) if is_covered) / covered_count
            if covered_count
            else None
        ),
        "open_set": {
            "n_items": len(open_set_pairs),
            "unknown_rejection_recall": (
                sum(actual.available and actual.abstained for _, actual in open_set_pairs)
                / len(open_set_pairs)
                if open_set_pairs
                else None
            ),
            "false_acceptance_rate": (
                sum(actual.available and not actual.abstained for _, actual in open_set_pairs)
                / len(open_set_pairs)
                if open_set_pairs
                else None
            ),
            "unknown_rejection_aupr": (
                binary_aupr(unknown_truth, unknown_scores) if unknown_scores else None
            ),
            "unknown_score_coverage": len(unknown_scores) / len(truth),
        },
        "known_class": {
            "n_items": len(known_pairs),
            "coverage": (
                sum(actual.available and not actual.abstained for _, actual in known_pairs)
                / len(known_pairs)
                if known_pairs
                else None
            ),
        },
        "calibration": {
            "scored_items": len(calibration_confidences),
            "coverage": len(calibration_confidences) / len(truth),
            "frozen_bin_ece": calibration,
        },
    }


def stratified_hierarchical_classification_metrics(
    truth: Sequence[ClassificationTruth],
    predicted: Sequence[HierarchicalClassificationPrediction],
    *,
    ontology: HierarchicalFaultOntology,
    strata: Mapping[str, Sequence[str]],
    require_correct_component_context: bool = True,
    local_mode_top_k: int = 3,
    ece_bin_edges: Sequence[float] = DEFAULT_FROZEN_ECE_BIN_EDGES,
    known_canonical_mode_ids: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Apply full classification metrics by profile, severity or other frozen axes."""

    if len(truth) != len(predicted):
        raise ValueError("truth and predicted must have equal length")
    global_known_ids = (
        set(known_canonical_mode_ids)
        if known_canonical_mode_ids is not None
        else {item.canonical_fault_mode_id for item in truth if not item.is_open_set}
    )
    validate_open_set_partition(
        known_mode_ids=global_known_ids,
        open_set_mode_ids={item.canonical_fault_mode_id for item in truth if item.is_open_set},
    )
    result: dict[str, dict[str, Any]] = {}
    for axis, values in strata.items():
        if len(values) != len(truth):
            raise ValueError(f"stratum axis {axis!r} does not align with episodes")
        result[axis] = {}
        for value in sorted(set(values)):
            indices = [index for index, item in enumerate(values) if item == value]
            result[axis][value] = hierarchical_classification_metrics(
                [truth[index] for index in indices],
                [predicted[index] for index in indices],
                ontology=ontology,
                require_correct_component_context=require_correct_component_context,
                local_mode_top_k=local_mode_top_k,
                ece_bin_edges=ece_bin_edges,
                known_canonical_mode_ids=global_known_ids,
            )
    return result


def validate_open_set_partition(
    *,
    known_mode_ids: Iterable[str],
    open_set_mode_ids: Iterable[str],
) -> dict[str, Any]:
    """Preflight canonical mode identities; aliases and case names are inadmissible."""

    known = {str(value).strip() for value in known_mode_ids}
    opened = {str(value).strip() for value in open_set_mode_ids}
    if "" in known or "" in opened:
        raise ValueError("canonical fault_mode_id values must be non-empty")
    overlap = known.intersection(opened)
    if overlap:
        raise ValueError(
            "known and open-set canonical fault_mode_id partitions overlap: "
            f"{sorted(overlap)}"
        )
    return {
        "status": "PASS",
        "known_mode_count": len(known),
        "open_set_mode_count": len(opened),
        "overlap_count": 0,
    }


@dataclass(frozen=True)
class CascadeTruth:
    """Evaluator-only truth for one nominal or single-fault episode."""

    episode_id: str
    has_fault: bool
    localization: LocalizationTruth | None = None
    classification: ClassificationTruth | None = None
    mission_profile: str = "UNSPECIFIED"
    severity_bin: str = "UNSPECIFIED"

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("episode_id must be non-empty")
        if not self.mission_profile.strip() or not self.severity_bin.strip():
            raise ValueError("mission_profile and severity_bin must be non-empty")
        if self.has_fault != (self.localization is not None and self.classification is not None):
            raise ValueError("fault episodes require localization and classification truth")
        if self.localization and self.classification:
            if self.localization.source_component != self.classification.component:
                raise ValueError("classification must describe the source component")


@dataclass(frozen=True)
class CascadeEpisodeResult:
    """Predictions needed to distinguish stage capability from cascade loss."""

    truth: CascadeTruth
    detected: bool
    localization_with_oracle_window: LocalizationPrediction | None = None
    localization_with_predicted_window: LocalizationPrediction | None = None
    classification_with_oracle_upstream: HierarchicalClassificationPrediction | None = None
    classification_with_predicted_upstream: HierarchicalClassificationPrediction | None = None

    def validate(self, ontology: HierarchicalFaultOntology) -> None:
        if not self.truth.has_fault:
            return
        if self.localization_with_oracle_window is None:
            raise ValueError("fault episode requires localization under an oracle window")
        if self.classification_with_oracle_upstream is None:
            raise ValueError("fault episode requires classification under oracle upstream")
        expected_component = self.truth.localization.source_component  # type: ignore[union-attr]
        if self.classification_with_oracle_upstream.conditioned_component != expected_component:
            raise ValueError("oracle classification must be conditioned on the true source")
        self.classification_with_oracle_upstream.validate(ontology)
        if self.detected and self.localization_with_predicted_window is None:
            raise ValueError("detected fault requires predicted-window localization")
        if not self.detected and (
            self.localization_with_predicted_window is not None
            or self.classification_with_predicted_upstream is not None
        ):
            raise ValueError("downstream heads must not run after a missed detection")
        if self.classification_with_predicted_upstream is not None:
            if self.localization_with_predicted_window is None:
                raise ValueError("predicted-upstream classification requires localization")
            if (
                self.classification_with_predicted_upstream.conditioned_component
                != self.localization_with_predicted_window.top_source
            ):
                raise ValueError("classification context must equal the predicted source")
            self.classification_with_predicted_upstream.validate(ontology)


def evaluate_cascade(
    episodes: Sequence[CascadeEpisodeResult],
    *,
    ontology: HierarchicalFaultOntology,
    localization_top_k: int = 3,
    classification_top_k: int = 3,
    ece_bin_edges: Sequence[float] = DEFAULT_FROZEN_ECE_BIN_EDGES,
    include_confidence_intervals: bool = False,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Evaluate oracle stage ceilings, operational cascade and first error."""

    if not episodes:
        raise ValueError("at least one episode is required")
    for episode in episodes:
        episode.validate(ontology)

    fault_episodes = [item for item in episodes if item.truth.has_fault]
    nominal_episodes = [item for item in episodes if not item.truth.has_fault]
    true_positive = sum(item.detected for item in fault_episodes)
    false_positive = sum(item.detected for item in nominal_episodes)
    detection = {
        "n_fault_episodes": len(fault_episodes),
        "n_nominal_episodes": len(nominal_episodes),
        "fault_event_recall": true_positive / len(fault_episodes) if fault_episodes else None,
        "nominal_specificity": (
            (len(nominal_episodes) - false_positive) / len(nominal_episodes)
            if nominal_episodes
            else None
        ),
        "episode_accuracy": (
            true_positive + len(nominal_episodes) - false_positive
        ) / len(episodes),
    }

    if fault_episodes:
        loc_truth = [item.truth.localization for item in fault_episodes]
        loc_oracle = [item.localization_with_oracle_window for item in fault_episodes]
        if any(item is None for item in loc_truth + loc_oracle):
            raise ValueError(
                "fault episodes require truth and oracle localization predictions"
            )
        localization_oracle = localization_metrics(
            loc_truth,  # type: ignore[arg-type]
            loc_oracle,  # type: ignore[arg-type]
            top_k=localization_top_k,
        )
        localization_predicted_inputs = [
            item.localization_with_predicted_window
            if item.detected and item.localization_with_predicted_window is not None
            else LocalizationPrediction((), decision=LocalizationDecision.ABSTAIN)
            for item in fault_episodes
        ]
        localization_predicted = localization_metrics(
            loc_truth,  # type: ignore[arg-type]
            localization_predicted_inputs,
            top_k=localization_top_k,
        )
        localization_strata = {
            "mission_profile": [item.truth.mission_profile for item in fault_episodes],
            "severity_bin": [item.truth.severity_bin for item in fault_episodes],
        }
        localization_oracle_by_stratum = stratified_localization_metrics(
            loc_truth,  # type: ignore[arg-type]
            loc_oracle,  # type: ignore[arg-type]
            strata=localization_strata,
            top_k=localization_top_k,
        )
        localization_predicted_by_stratum = stratified_localization_metrics(
            loc_truth,  # type: ignore[arg-type]
            localization_predicted_inputs,
            strata=localization_strata,
            top_k=localization_top_k,
        )

        class_truth = [item.truth.classification for item in fault_episodes]
        class_oracle = [item.classification_with_oracle_upstream for item in fault_episodes]
        if any(item is None for item in class_truth + class_oracle):
            raise ValueError(
                "fault episodes require truth and oracle classification predictions"
            )
        classification_oracle = hierarchical_classification_metrics(
            class_truth,  # type: ignore[arg-type]
            class_oracle,  # type: ignore[arg-type]
            ontology=ontology,
            local_mode_top_k=classification_top_k,
            ece_bin_edges=ece_bin_edges,
        )
        classification_predicted_inputs = [
            item.classification_with_predicted_upstream
            if item.classification_with_predicted_upstream is not None
            else HierarchicalClassificationPrediction(
                item.truth.classification.component,  # type: ignore[union-attr]
                UNKNOWN,
                UNKNOWN,
                abstained=True,
                available=False,
            )
            for item in fault_episodes
        ]
        classification_predicted = hierarchical_classification_metrics(
            class_truth,  # type: ignore[arg-type]
            classification_predicted_inputs,
            ontology=ontology,
            require_correct_component_context=False,
            local_mode_top_k=classification_top_k,
            ece_bin_edges=ece_bin_edges,
        )
        classification_strata = {
            "mission_profile": [item.truth.mission_profile for item in fault_episodes],
            "severity_bin": [item.truth.severity_bin for item in fault_episodes],
        }
        classification_oracle_by_stratum = stratified_hierarchical_classification_metrics(
            class_truth,  # type: ignore[arg-type]
            class_oracle,  # type: ignore[arg-type]
            ontology=ontology,
            strata=classification_strata,
            local_mode_top_k=classification_top_k,
            ece_bin_edges=ece_bin_edges,
        )
        classification_predicted_by_stratum = stratified_hierarchical_classification_metrics(
            class_truth,  # type: ignore[arg-type]
            classification_predicted_inputs,
            ontology=ontology,
            strata=classification_strata,
            require_correct_component_context=False,
            local_mode_top_k=classification_top_k,
            ece_bin_edges=ece_bin_edges,
        )
    else:
        localization_oracle = None
        localization_predicted = None
        localization_oracle_by_stratum = None
        localization_predicted_by_stratum = None
        classification_oracle = None
        classification_predicted = None
        classification_oracle_by_stratum = None
        classification_predicted_by_stratum = None

    first_wrong: Counter[str] = Counter()
    cascade_localization_correct = 0
    cascade_family_correct = 0
    cascade_mode_correct = 0
    covered_classification = 0
    fault_exact_correct = 0
    for episode in episodes:
        if not episode.truth.has_fault:
            first_wrong["detection"] += int(episode.detected)
            first_wrong["correct"] += int(not episode.detected)
            continue
        expected_loc = episode.truth.localization
        expected_class = episode.truth.classification
        if expected_loc is None or expected_class is None:
            raise ValueError(
                "fault episode truth requires both localization and classification labels"
            )
        if not episode.detected:
            first_wrong["detection"] += 1
            continue
        actual_loc = episode.localization_with_predicted_window
        if actual_loc is None:
            raise ValueError(
                "detected fault episode requires a predicted-window localization result"
            )
        if (
            actual_loc.decision != LocalizationDecision.IDENTIFIED
            or actual_loc.top_source != expected_loc.source_component
        ):
            first_wrong["localization"] += 1
            continue
        cascade_localization_correct += 1
        actual_class = episode.classification_with_predicted_upstream
        if actual_class is None:
            first_wrong["classification_abstention"] += 1
            continue
        if expected_class.is_open_set and actual_class.available and actual_class.abstained:
            cascade_family_correct += 1
            cascade_mode_correct += 1
            fault_exact_correct += 1
            first_wrong["correct"] += 1
            continue
        if actual_class.abstained:
            first_wrong["classification_abstention"] += 1
            continue
        covered_classification += 1
        if actual_class.mechanism_family != expected_class.mechanism_family:
            first_wrong["classification_family"] += 1
            continue
        cascade_family_correct += 1
        if actual_class.local_fault_mode != expected_class.local_fault_mode:
            first_wrong["classification_mode"] += 1
            continue
        cascade_mode_correct += 1
        fault_exact_correct += 1
        first_wrong["correct"] += 1

    n_fault = len(fault_episodes)
    first_wrong_order = (
        "detection",
        "localization",
        "classification_abstention",
        "classification_family",
        "classification_mode",
        "correct",
    )
    result = {
        "n_episodes": len(episodes),
        "detection": detection,
        "localization": {
            "oracle_detection_window": localization_oracle,
            "predicted_upstream": localization_predicted,
            "predicted_upstream_source_top1_accuracy": (
                cascade_localization_correct / n_fault if n_fault else None
            ),
            "oracle_by_stratum": localization_oracle_by_stratum,
            "predicted_upstream_by_stratum": localization_predicted_by_stratum,
        },
        "classification": {
            "oracle_upstream": classification_oracle,
            "predicted_upstream": classification_predicted,
            "predicted_upstream_family_accuracy": (
                cascade_family_correct / n_fault if n_fault else None
            ),
            "predicted_upstream_local_mode_accuracy": (
                cascade_mode_correct / n_fault if n_fault else None
            ),
            "predicted_upstream_coverage": (
                covered_classification / n_fault if n_fault else None
            ),
            "oracle_by_stratum": classification_oracle_by_stratum,
            "predicted_upstream_by_stratum": classification_predicted_by_stratum,
        },
        "end_to_end_exact_match": first_wrong["correct"] / len(episodes),
        "fault_only_exact_match": fault_exact_correct / n_fault if n_fault else None,
        "first_wrong_stage": {key: first_wrong[key] for key in first_wrong_order},
    }
    if include_confidence_intervals:
        result["confidence_intervals"] = {
            "detection_fault_event_recall": (
                wilson_interval(true_positive, n_fault, confidence_level=confidence_level)
                if n_fault
                else None
            ),
            "localization_oracle_source_top1": (
                wilson_interval(
                    round(localization_oracle["source_top1_accuracy"] * n_fault),
                    n_fault,
                    confidence_level=confidence_level,
                )
                if localization_oracle is not None
                else None
            ),
            "localization_predicted_source_top1": (
                wilson_interval(
                    cascade_localization_correct,
                    n_fault,
                    confidence_level=confidence_level,
                )
                if n_fault
                else None
            ),
            "classification_oracle_exact": (
                wilson_interval(
                    round(classification_oracle["hierarchical_exact_match"] * n_fault),
                    n_fault,
                    confidence_level=confidence_level,
                )
                if classification_oracle is not None
                else None
            ),
            "classification_predicted_exact": (
                wilson_interval(fault_exact_correct, n_fault, confidence_level=confidence_level)
                if n_fault
                else None
            ),
        }
    return result


__all__ = [
    "UNKNOWN",
    "CascadeEpisodeResult",
    "CascadeTruth",
    "CausalStreamContract",
    "CausalTelemetrySample",
    "ClassificationTruth",
    "DetectionAlarm",
    "HierarchicalClassificationPrediction",
    "HierarchicalFaultOntology",
    "LocalizationDecision",
    "LocalizationPrediction",
    "LocalizationTruth",
    "evaluate_cascade",
    "hierarchical_classification_metrics",
    "localization_metrics",
]
