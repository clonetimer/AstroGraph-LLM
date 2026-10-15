from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from astrograph.semantics import TelemetrySemanticsRegistry


class OnboardingStatus(str, Enum):
    READY_FOR_SEED_ONLY_REPLAY = "READY_FOR_SEED_ONLY_REPLAY"
    ONBOARDING_ONLY = "ONBOARDING_ONLY"
    UNKNOWN = "UNKNOWN"


class LabelStatus(str, Enum):
    ABSENT = "ABSENT"
    COMPLETE_KNOWN_LABELS = "COMPLETE_KNOWN_LABELS"
    PRESENT_WITH_UNKNOWN_LABELS = "PRESENT_WITH_UNKNOWN_LABELS"
    PRESENT_BUT_EMPTY = "PRESENT_BUT_EMPTY"


class DatasetFileProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    telemetry_path: str
    telemetry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    time_column_present: bool
    time_monotonic: bool
    median_sample_period_s: float | None = Field(default=None, ge=0.0)


class SchemaMappingReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: str
    required_model_input_count: int = Field(ge=0)
    mapped_required_count: int = Field(ge=0)
    required_field_coverage: float = Field(ge=0.0, le=1.0)
    mapped_fields: list[str] = Field(default_factory=list)
    missing_required_fields: list[str] = Field(default_factory=list)
    extra_columns: list[str] = Field(default_factory=list)
    component_instances: list[str] = Field(default_factory=list)


class LabelCoverageReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: LabelStatus
    label_columns: list[str] = Field(default_factory=list)
    class_label_columns: list[str] = Field(default_factory=list)
    label_values: list[str] = Field(default_factory=list)
    known_label_values: list[str] = Field(default_factory=list)
    unknown_label_values: list[str] = Field(default_factory=list)


class DatasetCapabilityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supported_axes: list[str] = Field(default_factory=list)
    unsupported_axes: list[str] = Field(default_factory=list)
    kg_covered_fault_ids: list[str] = Field(default_factory=list)
    kg_uncovered_fault_ids: list[str] = Field(default_factory=list)
    ood_risk_flags: list[str] = Field(default_factory=list)
    diagnosis_permission: str


class DatasetOnboardingReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-dataset-onboarding-report.v1"
    status: OnboardingStatus
    dataset_id: str = Field(min_length=1)
    file_profile: DatasetFileProfile | None = None
    schema_mapping: SchemaMappingReport | None = None
    label_coverage: LabelCoverageReport | None = None
    capability: DatasetCapabilityReport
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    claim_boundary: str = (
        "First-seen dataset onboarding is a deterministic compatibility and capability-boundary report. "
        "It is not a diagnosis, model validation, cross-condition evidence, or flight validation."
    )

    @field_validator("blocking_reasons", "warnings")
    @classmethod
    def unique_items(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_telemetry_csv(dataset_root: str | Path) -> Path:
    root = Path(dataset_root)
    preferred = root / "telemetry.csv"
    if preferred.is_file():
        return preferred
    csv_files = sorted(path for path in root.glob("*.csv") if path.is_file())
    if not csv_files:
        raise FileNotFoundError(f"no telemetry CSV found under {root}")
    profiles: list[tuple[int, str, Path]] = []
    for path in csv_files:
        try:
            columns = list(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        score = (10 if "time_s" in columns else 0) + len(columns)
        profiles.append((score, path.name, path))
    if not profiles:
        raise FileNotFoundError(f"no readable telemetry CSV found under {root}")
    return sorted(profiles, key=lambda item: (-item[0], item[1]))[0][2]


def _file_profile(path: Path, frame: pd.DataFrame) -> DatasetFileProfile:
    times = pd.to_numeric(frame["time_s"], errors="coerce") if "time_s" in frame else pd.Series(dtype=float)
    diffs = times.diff().dropna()
    return DatasetFileProfile(
        telemetry_path=str(path),
        telemetry_sha256=sha256_file(path),
        row_count=len(frame),
        column_count=len(frame.columns),
        time_column_present="time_s" in frame.columns,
        time_monotonic=bool(not times.empty and times.notna().all() and times.is_monotonic_increasing),
        median_sample_period_s=(float(diffs.median()) if not diffs.empty else None),
    )


def _schema_mapping(frame: pd.DataFrame, semantics: TelemetrySemanticsRegistry) -> SchemaMappingReport:
    required = semantics.model_input_fields()
    columns = set(map(str, frame.columns))
    mapped = sorted(set(required) & columns)
    missing = sorted(set(required) - columns)
    extra = sorted(columns - set(required) - {"time_s"})
    mapped_semantics = semantics.require_fields(mapped)
    components = sorted({item.component_instance for item in mapped_semantics if item.component_instance})
    coverage = len(mapped) / len(required) if required else 0.0
    return SchemaMappingReport(
        protocol_version=semantics.registry_version,
        required_model_input_count=len(required),
        mapped_required_count=len(mapped),
        required_field_coverage=coverage,
        mapped_fields=mapped,
        missing_required_fields=missing,
        extra_columns=extra,
        component_instances=components,
    )


def _label_coverage(frame: pd.DataFrame, registered_fault_ids: set[str]) -> LabelCoverageReport:
    label_columns = [
        column
        for column in frame.columns
        if str(column).startswith("label.") or str(column).startswith("truth.") or str(column) in {"fault_id", "class_id"}
    ]
    class_label_columns = [
        column
        for column in label_columns
        if str(column) in {"fault_id", "class_id", "truth.fault_id", "truth.class_id", "label.fault_id", "label.class_id"}
    ]
    if not class_label_columns:
        return LabelCoverageReport(status=LabelStatus.ABSENT, label_columns=sorted(label_columns))
    values: set[str] = set()
    for column in class_label_columns:
        series = frame[column].dropna().astype(str).str.strip()
        values.update(item for item in series if item)
    if not values:
        return LabelCoverageReport(
            status=LabelStatus.PRESENT_BUT_EMPTY,
            label_columns=sorted(label_columns),
            class_label_columns=sorted(class_label_columns),
        )
    known = sorted(values & (registered_fault_ids | {"NORMAL"}))
    unknown = sorted(values - set(known))
    return LabelCoverageReport(
        status=LabelStatus.COMPLETE_KNOWN_LABELS if not unknown else LabelStatus.PRESENT_WITH_UNKNOWN_LABELS,
        label_columns=sorted(label_columns),
        class_label_columns=sorted(class_label_columns),
        label_values=sorted(values),
        known_label_values=known,
        unknown_label_values=unknown,
    )


def _axis_support(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    columns = {str(column) for column in frame.columns}
    supported = ["seed"]
    markers = {
        "severity": {"severity", "fault_severity", "fault_severity_id"},
        "initial_state": {"initial_state", "initial_state_id"},
        "operating_condition": {"operating_condition", "operating_condition_id"},
        "fault_timing": {"fault_start_s", "fault_duration_s", "event.start_s"},
        "noise": {"noise_level", "sensor_noise", "noise_std"},
        "dropout": {"dropout_rate", "dropout_mask", "sensor_dropout"},
        "satellite_instance": {"satellite_instance", "satellite_instance_id"},
        "flight_data": {"flight_id", "spacecraft_id", "mission_elapsed_time"},
    }
    for axis, names in markers.items():
        if columns & names:
            supported.append(axis)
    unsupported = sorted(set(markers) - set(supported))
    return sorted(supported), unsupported


def build_dataset_onboarding_report(
    *,
    dataset_root: str | Path,
    protocol_root: str | Path,
    dataset_id: str | None = None,
    registered_fault_ids: list[str] | None = None,
    minimum_required_field_coverage: float = 0.95,
) -> DatasetOnboardingReport:
    registered = set(registered_fault_ids or [])
    blocking: list[str] = []
    warnings: list[str] = []
    try:
        telemetry_path = discover_telemetry_csv(dataset_root)
        frame = pd.read_csv(telemetry_path, low_memory=False)
    except Exception as exc:
        return DatasetOnboardingReport(
            status=OnboardingStatus.UNKNOWN,
            dataset_id=dataset_id or str(dataset_root),
            capability=DatasetCapabilityReport(
                supported_axes=[],
                unsupported_axes=[
                    "seed",
                    "severity",
                    "initial_state",
                    "operating_condition",
                    "fault_timing",
                    "noise",
                    "dropout",
                    "satellite_instance",
                    "flight_data",
                ],
                diagnosis_permission="UNKNOWN_ONLY",
            ),
            blocking_reasons=["NO_READABLE_TELEMETRY_CSV"],
            warnings=[str(exc)],
        )

    semantics = TelemetrySemanticsRegistry(protocol_root)
    file_profile = _file_profile(telemetry_path, frame)
    mapping = _schema_mapping(frame, semantics)
    labels = _label_coverage(frame, registered)
    supported_axes, unsupported_axes = _axis_support(frame)

    if not file_profile.time_column_present:
        blocking.append("MISSING_TIME_COLUMN")
    if not file_profile.time_monotonic:
        blocking.append("TIME_NOT_MONOTONIC")
    if mapping.required_field_coverage < minimum_required_field_coverage:
        blocking.append("REQUIRED_FIELD_COVERAGE_BELOW_THRESHOLD")
    if mapping.mapped_required_count == 0:
        blocking.append("NO_MAPPED_MODEL_INPUT_FIELDS")
    if labels.status == LabelStatus.ABSENT:
        warnings.append("LABELS_ABSENT_ACCURACY_NOT_EVALUABLE")
    if labels.status == LabelStatus.PRESENT_WITH_UNKNOWN_LABELS:
        blocking.append("LABELS_REFERENCE_UNREGISTERED_FAULTS")
    if labels.status == LabelStatus.PRESENT_BUT_EMPTY:
        warnings.append("LABEL_COLUMNS_PRESENT_BUT_EMPTY")

    kg_covered = sorted(set(labels.known_label_values) - {"NORMAL"})
    kg_uncovered = sorted(set(labels.unknown_label_values))
    ood_flags = []
    if mapping.extra_columns:
        ood_flags.append("EXTRA_COLUMNS_NOT_IN_PROTOCOL")
    if blocking:
        status = OnboardingStatus.UNKNOWN
        permission = "UNKNOWN_ONLY"
    elif labels.status == LabelStatus.COMPLETE_KNOWN_LABELS:
        status = OnboardingStatus.READY_FOR_SEED_ONLY_REPLAY
        permission = "SEED_ONLY_REPLAY_ALLOWED"
    else:
        status = OnboardingStatus.ONBOARDING_ONLY
        permission = "CANDIDATE_EXPLANATION_AND_GAP_REPORT_ONLY"

    return DatasetOnboardingReport(
        status=status,
        dataset_id=dataset_id or Path(dataset_root).resolve().name,
        file_profile=file_profile,
        schema_mapping=mapping,
        label_coverage=labels,
        capability=DatasetCapabilityReport(
            supported_axes=supported_axes,
            unsupported_axes=unsupported_axes,
            kg_covered_fault_ids=kg_covered,
            kg_uncovered_fault_ids=kg_uncovered,
            ood_risk_flags=ood_flags,
            diagnosis_permission=permission,
        ),
        blocking_reasons=blocking,
        warnings=warnings,
    )
