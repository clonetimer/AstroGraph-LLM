from __future__ import annotations

import math
from collections import defaultdict
from enum import Enum
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .models import DataDrivenPrediction


class SplitName(str, Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    UNASSIGNED = "UNASSIGNED"


class CredibilityStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"


class RunProvenanceType(str, Enum):
    """Provenance categories relevant to statistical independence claims."""

    SIMULATION_NATIVE = "SIMULATION_NATIVE"
    FLIGHT_NATIVE = "FLIGHT_NATIVE"
    HARDWARE_IN_LOOP_NATIVE = "HARDWARE_IN_LOOP_NATIVE"
    REPLAY_CLONE = "REPLAY_CLONE"
    AUGMENTED_DERIVATIVE = "AUGMENTED_DERIVATIVE"
    WINDOW_SLICE_DERIVATIVE = "WINDOW_SLICE_DERIVATIVE"
    UNSPECIFIED = "UNSPECIFIED"

    @property
    def eligible_as_independent(self) -> bool:
        return self in {
            RunProvenanceType.SIMULATION_NATIVE,
            RunProvenanceType.FLIGHT_NATIVE,
            RunProvenanceType.HARDWARE_IN_LOOP_NATIVE,
        }


class IndependentRunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    class_ids: list[str]
    split: SplitName = SplitName.UNASSIGNED
    object_id: str | None = None
    operating_condition_id: str | None = None
    satellite_instance_id: str | None = None
    random_seed: int | None = None
    initial_state_id: str | None = None
    fault_severity_id: str | None = None
    provenance_type: RunProvenanceType = RunProvenanceType.UNSPECIFIED
    independence_group_id: str | None = None
    parent_run_id: str | None = None
    paired_nominal_run_id: str | None = None
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    configuration_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("class_ids")
    @classmethod
    def require_unique_classes(cls, value: list[str]) -> list[str]:
        if not value or len(value) != len(set(value)):
            raise ValueError("run class IDs must be nonempty and unique")
        return value


def resolve_run_scenario_dir(
    record: IndependentRunRecord,
    *,
    inventory_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> Path:
    """Resolve portable and legacy scenario-directory metadata.

    New inventories store a path relative to the campaign output root. Legacy
    inventories may contain a Windows absolute path from the producer machine; when
    that path is unavailable, the suffix below ``runtime_outputs`` is rebound to the
    inventory's local directory.
    """

    raw = str(record.metadata.get("scenario_dir_relative") or record.metadata.get("scenario_dir") or "")
    if not raw:
        raise ValueError(f"run {record.run_id} does not declare scenario_dir metadata")

    candidates: list[Path] = []
    direct = Path(raw)
    if direct.is_absolute():
        candidates.append(direct)
    if output_root is not None:
        candidates.append(Path(output_root).resolve() / Path(raw.replace("\\", "/")))

    inventory = Path(inventory_path).resolve() if inventory_path is not None else None
    if inventory is not None:
        local_runtime_root = inventory.parent / "runtime_outputs"
        normalized = raw.replace("\\", "/")
        relative = Path(normalized)
        if not direct.is_absolute():
            candidates.extend([inventory.parent / relative, local_runtime_root / relative])

        windows_parts = list(PureWindowsPath(raw).parts)
        lowered = [part.lower() for part in windows_parts]
        if "runtime_outputs" in lowered:
            index = lowered.index("runtime_outputs")
            suffix = Path(*windows_parts[index + 1 :])
            candidates.append(local_runtime_root / suffix)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_dir():
            return resolved
    raise FileNotFoundError(
        f"cannot resolve scenario directory for run {record.run_id}; tried: "
        + ", ".join(str(item) for item in candidates)
    )


class IndependentRunAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-independent-run-audit.v2"
    status: CredibilityStatus
    run_count: int
    eligible_independent_run_count: int = 0
    class_run_counts: dict[str, int]
    class_split_counts: dict[str, dict[str, int]]
    duplicate_run_ids: list[str] = Field(default_factory=list)
    duplicate_content_groups: dict[str, list[str]] = Field(default_factory=dict)
    cross_split_independence_groups: dict[str, list[str]] = Field(default_factory=dict)
    ineligible_run_ids: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    minimum_runs_per_class: int
    required_splits: list[SplitName]
    strict_provenance: bool = False
    claim_boundary: str = (
        "PASS only means the run inventory satisfies configured provenance, grouping, content-duplicate, and "
        "split-structure checks. It does not establish flight accuracy, representativeness, label correctness, "
        "simulator fidelity, or absence of domain shift."
    )


def audit_independent_runs(
    records: list[IndependentRunRecord],
    *,
    minimum_runs_per_class: int = 3,
    required_splits: tuple[SplitName, ...] = (SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST),
    strict_provenance: bool = False,
    reject_exact_content_duplicates: bool = True,
    reject_group_leakage: bool = True,
) -> IndependentRunAudit:
    run_ids = [item.run_id for item in records]
    duplicate_run_ids = sorted({item for item in run_ids if run_ids.count(item) > 1})

    blocking: list[str] = []
    warnings: list[str] = []
    if duplicate_run_ids:
        blocking.append("DUPLICATE_RUN_ID_ACROSS_INVENTORY")

    content_groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.content_sha256:
            content_groups[record.content_sha256].append(record.run_id)
    duplicate_content_groups = {
        digest: sorted(ids) for digest, ids in sorted(content_groups.items()) if len(set(ids)) > 1
    }
    if reject_exact_content_duplicates and duplicate_content_groups:
        blocking.append("EXACT_TELEMETRY_CONTENT_REUSED_ACROSS_RUN_IDS")

    independence_groups: dict[str, list[IndependentRunRecord]] = defaultdict(list)
    for record in records:
        if record.independence_group_id:
            independence_groups[record.independence_group_id].append(record)
    cross_split_groups: dict[str, list[str]] = {}
    for group_id, items in sorted(independence_groups.items()):
        splits = {item.split for item in items if item.split != SplitName.UNASSIGNED}
        if len(splits) > 1:
            cross_split_groups[group_id] = sorted(item.run_id for item in items)
    if reject_group_leakage and cross_split_groups:
        blocking.append("INDEPENDENCE_GROUP_CROSSES_DATASET_SPLITS")

    eligible_records: list[IndependentRunRecord] = []
    ineligible_run_ids: list[str] = []
    for record in records:
        explicit_derivative = record.parent_run_id is not None or record.provenance_type in {
            RunProvenanceType.REPLAY_CLONE,
            RunProvenanceType.AUGMENTED_DERIVATIVE,
            RunProvenanceType.WINDOW_SLICE_DERIVATIVE,
        }
        if strict_provenance:
            eligible = record.provenance_type.eligible_as_independent and not explicit_derivative
            if record.provenance_type == RunProvenanceType.UNSPECIFIED:
                blocking.append(f"UNSPECIFIED_RUN_PROVENANCE:{record.run_id}")
            if explicit_derivative:
                blocking.append(f"DERIVATIVE_RUN_CANNOT_COUNT_AS_INDEPENDENT:{record.run_id}")
        else:
            eligible = not explicit_derivative
            if record.provenance_type == RunProvenanceType.UNSPECIFIED:
                eligible = True  # V0.5.15 compatibility; strict mode closes this gap.
        if eligible:
            eligible_records.append(record)
        else:
            ineligible_run_ids.append(record.run_id)

    class_run_counts: dict[str, int] = defaultdict(int)
    class_split_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in eligible_records:
        for class_id in record.class_ids:
            class_run_counts[class_id] += 1
            class_split_counts[class_id][record.split.value] += 1

    all_classes = sorted({class_id for record in records for class_id in record.class_ids})
    for class_id in all_classes:
        count = class_run_counts.get(class_id, 0)
        if count < minimum_runs_per_class:
            blocking.append(f"INSUFFICIENT_INDEPENDENT_RUNS:{class_id}:{count}<{minimum_runs_per_class}")
        for split in required_splits:
            if class_split_counts[class_id].get(split.value, 0) < 1:
                blocking.append(f"MISSING_CLASS_SPLIT:{class_id}:{split.value}")
    if any(item.split == SplitName.UNASSIGNED for item in records):
        blocking.append("UNASSIGNED_RUNS_PRESENT")
    if not records:
        blocking.append("NO_RUN_RECORDS")
    if records and all(item.satellite_instance_id is None for item in records):
        warnings.append("SATELLITE_INSTANCE_ID_NOT_RECORDED")
    if records and all(item.operating_condition_id is None for item in records):
        warnings.append("OPERATING_CONDITION_ID_NOT_RECORDED")
    if strict_provenance and records and all(item.configuration_sha256 is None for item in records):
        warnings.append("RUN_CONFIGURATION_HASH_NOT_RECORDED")
    if strict_provenance and records and all(item.content_sha256 is None for item in records):
        warnings.append("RUN_CONTENT_HASH_NOT_RECORDED")
    if not strict_provenance and any(item.provenance_type == RunProvenanceType.UNSPECIFIED for item in records):
        # Compatibility warning is intentionally omitted from status so existing V0.5.15 API behavior remains stable.
        pass

    # Preserve order while removing repeated reason strings.
    blocking = list(dict.fromkeys(blocking))
    warnings = list(dict.fromkeys(warnings))
    status = CredibilityStatus.BLOCKED if blocking else (CredibilityStatus.WARNING if warnings else CredibilityStatus.PASS)
    return IndependentRunAudit(
        status=status,
        run_count=len(records),
        eligible_independent_run_count=len(eligible_records),
        class_run_counts=dict(sorted(class_run_counts.items())),
        class_split_counts={
            class_id: dict(sorted(counts.items()))
            for class_id, counts in sorted(class_split_counts.items())
        },
        duplicate_run_ids=duplicate_run_ids,
        duplicate_content_groups=duplicate_content_groups,
        cross_split_independence_groups=cross_split_groups,
        ineligible_run_ids=sorted(ineligible_run_ids),
        blocking_reasons=blocking,
        warnings=warnings,
        minimum_runs_per_class=minimum_runs_per_class,
        required_splits=list(required_splits),
        strict_provenance=strict_provenance,
    )


class CalibrationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_count: int = Field(ge=0)
    negative_log_likelihood: float = Field(ge=0.0)
    multiclass_brier_score: float = Field(ge=0.0)
    expected_calibration_error: float = Field(ge=0.0, le=1.0)


def calibration_metrics(
    probabilities: np.ndarray,
    truth_indices: np.ndarray,
    *,
    bins: int = 10,
) -> CalibrationMetrics:
    probabilities = np.asarray(probabilities, dtype=float)
    truth_indices = np.asarray(truth_indices, dtype=int)
    if probabilities.ndim != 2 or len(probabilities) != len(truth_indices):
        raise ValueError("probabilities must be [samples, classes] and align with truth")
    if len(probabilities) == 0:
        return CalibrationMetrics(
            sample_count=0,
            negative_log_likelihood=0.0,
            multiclass_brier_score=0.0,
            expected_calibration_error=0.0,
        )
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    nll = -float(np.mean(np.log(probabilities[np.arange(len(truth_indices)), truth_indices])))
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(truth_indices)), truth_indices] = 1.0
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == truth_indices
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence > lower if index else confidence >= lower) & (confidence <= upper)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(float(np.mean(correct[mask])) - float(np.mean(confidence[mask])))
    return CalibrationMetrics(
        sample_count=len(truth_indices),
        negative_log_likelihood=nll,
        multiclass_brier_score=brier,
        expected_calibration_error=ece,
    )


class TemperatureScalingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-temperature-scaling-record.v1"
    calibration_id: str = Field(min_length=1)
    temperature: float = Field(gt=0.0)
    validation_before: CalibrationMetrics
    validation_after: CalibrationMetrics
    method: str = "GRID_SEARCH_TEMPERATURE_ON_LOG_PROBABILITIES"
    claim_boundary: str = (
        "Post-hoc calibration is valid only for the represented validation distribution and does not repair "
        "class leakage, dataset shift, or insufficient independent runs."
    )


class TemperatureScaler:
    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        values = np.asarray(probabilities, dtype=float)
        values = np.clip(values, 1e-12, 1.0)
        logits = np.log(values) / self.temperature
        logits = logits - logits.max(axis=1, keepdims=True)
        exponentials = np.exp(logits)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    @classmethod
    def fit(
        cls,
        probabilities: np.ndarray,
        truth_indices: np.ndarray,
        *,
        candidates: np.ndarray | None = None,
        calibration_id: str = "temperature-grid-v1",
    ) -> tuple["TemperatureScaler", TemperatureScalingRecord]:
        probabilities = np.asarray(probabilities, dtype=float)
        truth_indices = np.asarray(truth_indices, dtype=int)
        candidates = candidates if candidates is not None else np.linspace(0.25, 5.0, 192)
        before = calibration_metrics(probabilities, truth_indices)
        best_temperature = 1.0
        best_metric = before.negative_log_likelihood
        for temperature in candidates:
            scaler = cls(float(temperature))
            metric = calibration_metrics(scaler.transform(probabilities), truth_indices)
            if metric.negative_log_likelihood < best_metric - 1e-12:
                best_metric = metric.negative_log_likelihood
                best_temperature = float(temperature)
        scaler = cls(best_temperature)
        after = calibration_metrics(scaler.transform(probabilities), truth_indices)
        return scaler, TemperatureScalingRecord(
            calibration_id=calibration_id,
            temperature=best_temperature,
            validation_before=before,
            validation_after=after,
        )


class OpenSetStatus(str, Enum):
    ACCEPT_KNOWN = "ACCEPT_KNOWN"
    ACCEPT_NORMAL = "ACCEPT_NORMAL"
    REJECT_UNKNOWN = "REJECT_UNKNOWN"


class OpenSetPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-open-set-policy.v1"
    minimum_top_probability: float = Field(default=0.45, ge=0.0, le=1.0)
    minimum_top_margin: float = Field(default=0.08, ge=0.0, le=1.0)
    maximum_ood_score: float | None = Field(default=None, ge=0.0)
    require_calibrated_probabilities: bool = True
    suppress_data_branch_on_reject: bool = True
    claim_boundary: str = "Engineering rejection policy; thresholds require independent-run calibration."


class OpenSetAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-open-set-assessment.v1"
    status: OpenSetStatus
    top_class_id: str
    top_probability: float = Field(ge=0.0, le=1.0)
    second_probability: float = Field(ge=0.0, le=1.0)
    margin: float = Field(ge=0.0, le=1.0)
    out_of_distribution_score: float | None = Field(default=None, ge=0.0)
    reasons: list[str] = Field(default_factory=list)
    data_branch_eligible_for_fusion: bool
    policy: OpenSetPolicy


def assess_open_set(prediction: DataDrivenPrediction, policy: OpenSetPolicy | None = None) -> OpenSetAssessment:
    policy = policy or OpenSetPolicy()
    ordered = sorted(prediction.classes, key=lambda item: item.rank)
    if not ordered:
        raise ValueError("prediction has no classes")
    top = ordered[0]
    second_probability = ordered[1].probability if len(ordered) > 1 else 0.0
    margin = max(0.0, top.probability - second_probability)
    reasons: list[str] = []
    if top.probability < policy.minimum_top_probability:
        reasons.append("TOP_PROBABILITY_BELOW_THRESHOLD")
    if margin < policy.minimum_top_margin:
        reasons.append("TOP_MARGIN_BELOW_THRESHOLD")
    if policy.maximum_ood_score is not None:
        if prediction.out_of_distribution_score is None:
            reasons.append("OOD_SCORE_REQUIRED_BUT_MISSING")
        elif prediction.out_of_distribution_score > policy.maximum_ood_score:
            reasons.append("OOD_SCORE_ABOVE_THRESHOLD")
    if policy.require_calibrated_probabilities and not prediction.calibrated_probabilities:
        reasons.append("CALIBRATED_PROBABILITIES_REQUIRED")

    if reasons:
        status = OpenSetStatus.REJECT_UNKNOWN
    elif top.class_id == prediction.normal_class_id:
        status = OpenSetStatus.ACCEPT_NORMAL
    else:
        status = OpenSetStatus.ACCEPT_KNOWN
    eligible = not (status == OpenSetStatus.REJECT_UNKNOWN and policy.suppress_data_branch_on_reject)
    return OpenSetAssessment(
        status=status,
        top_class_id=top.class_id,
        top_probability=top.probability,
        second_probability=second_probability,
        margin=margin,
        out_of_distribution_score=prediction.out_of_distribution_score,
        reasons=reasons,
        data_branch_eligible_for_fusion=eligible,
        policy=policy,
    )
