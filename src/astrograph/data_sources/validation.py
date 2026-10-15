from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from .base import DatasetSource
from .models import (
    DataQualityIssue,
    DataSourceValidationReport,
)


def _timestamp_quality(series: pd.Series) -> tuple[bool, bool, int, int]:
    numeric = pd.to_numeric(series, errors="coerce")
    if len(series) and float(numeric.notna().mean()) >= 0.95:
        parsed = numeric
    else:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
    invalid = int(parsed.isna().sum())
    valid = parsed.dropna()
    monotonic = bool(valid.is_monotonic_increasing)
    duplicate_count = int(valid.duplicated().sum())
    unique = duplicate_count == 0
    return monotonic, unique, invalid, duplicate_count


def validate_data_source(
    source: DatasetSource,
    *,
    max_entities: int = 100,
    max_rows_per_entity: int = 10000,
) -> DataSourceValidationReport:
    """Inspect a bounded subset of a data source and return a structured gate report."""

    opened_here = not source.opened
    issues: list[DataQualityIssue] = []
    inspected = 0
    row_total = 0
    channel_counts: list[int] = []
    try:
        if opened_here:
            source.open()
        metadata = source.get_metadata()
        entities = source.list_entities()
        if not metadata.read_only:
            issues.append(
                DataQualityIssue(
                    code="SOURCE_NOT_READ_ONLY",
                    severity="ERROR",
                    message="Built-in dataset validation requires a read-only source.",
                )
            )
        if not entities:
            issues.append(
                DataQualityIssue(
                    code="NO_DATA_ENTITIES",
                    severity="ERROR",
                    message="No readable data entities were discovered.",
                )
            )
        mission_layout = bool(
            metadata.metadata.get("mission_layout_detected", False)
        )
        if mission_layout:
            telemetry_count = int(
                metadata.metadata.get("telemetry_archive_entity_count", 0)
            )
            if telemetry_count < 1:
                issues.append(
                    DataQualityIssue(
                        code="ESA_ADB_NO_TELEMETRY_ARCHIVES",
                        severity="ERROR",
                        message=(
                            "Raw ESA mission layout was detected, but no allow-listed "
                            "telemetry pickle ZIP entity was discovered."
                        ),
                    )
                )
            mismatches = {
                key: metadata.metadata.get(key, [])
                for key in (
                    "missing_channel_archives",
                    "extra_channel_archives_not_allowlisted",
                    "missing_telecommand_archives",
                    "extra_telecommand_archives_not_allowlisted",
                )
                if metadata.metadata.get(key)
            }
            if mismatches:
                issues.append(
                    DataQualityIssue(
                        code="ESA_ADB_METADATA_ARCHIVE_MISMATCH",
                        severity="WARNING",
                        message=(
                            "Mission metadata and ZIP archive inventory are not identical; "
                            "non-allow-listed archives remain excluded."
                        ),
                        details=mismatches,
                    )
                )
        for entity in entities[:max_entities]:
            inspected += 1
            try:
                channels = source.list_channels(entity.entity_id)
                channel_counts.append(len(channels))
                if not channels:
                    issues.append(
                        DataQualityIssue(
                            code="NO_MODEL_CHANNELS",
                            severity="WARNING",
                            entity_id=entity.entity_id,
                            message="No non-label data channels were discovered.",
                        )
                    )
                frame = source.read_range(
                    entity.entity_id,
                    channels=channels,
                    limit=max_rows_per_entity,
                )
                row_total += len(frame)
                if frame.empty:
                    issues.append(
                        DataQualityIssue(
                            code="EMPTY_ENTITY",
                            severity="ERROR",
                            entity_id=entity.entity_id,
                            message="The entity contains no readable rows.",
                        )
                    )
                    continue
                duplicate_columns = [
                    name for name, count in Counter(map(str, frame.columns)).items() if count > 1
                ]
                if duplicate_columns:
                    issues.append(
                        DataQualityIssue(
                            code="DUPLICATE_COLUMNS",
                            severity="ERROR",
                            entity_id=entity.entity_id,
                            message="Duplicate column names are not supported.",
                            details={"columns": duplicate_columns},
                        )
                    )
                missing_count = int(frame.isna().sum().sum())
                if missing_count:
                    issues.append(
                        DataQualityIssue(
                            code="MISSING_VALUES",
                            severity="WARNING",
                            entity_id=entity.entity_id,
                            message="Missing values were found in the inspected range.",
                            details={
                                "missing_value_count": missing_count,
                                "inspected_cell_count": int(frame.shape[0] * frame.shape[1]),
                            },
                        )
                    )
                if entity.timestamp_column:
                    if entity.timestamp_column not in frame.columns:
                        issues.append(
                            DataQualityIssue(
                                code="TIMESTAMP_COLUMN_NOT_RETURNED",
                                severity="ERROR",
                                entity_id=entity.entity_id,
                                message="Configured timestamp column is absent from the read result.",
                                details={"timestamp_column": entity.timestamp_column},
                            )
                        )
                    else:
                        monotonic, unique, invalid, duplicates = _timestamp_quality(
                            frame[entity.timestamp_column]
                        )
                        if invalid:
                            issues.append(
                                DataQualityIssue(
                                    code="INVALID_TIMESTAMPS",
                                    severity="ERROR",
                                    entity_id=entity.entity_id,
                                    message="Unparseable timestamps were found.",
                                    details={"invalid_timestamp_count": invalid},
                                )
                            )
                        if not monotonic:
                            issues.append(
                                DataQualityIssue(
                                    code="TIMESTAMPS_NOT_MONOTONIC",
                                    severity="WARNING",
                                    entity_id=entity.entity_id,
                                    message="Timestamps are not monotonically increasing.",
                                )
                            )
                        if not unique:
                            issues.append(
                                DataQualityIssue(
                                    code="DUPLICATE_TIMESTAMPS",
                                    severity="WARNING",
                                    entity_id=entity.entity_id,
                                    message="Duplicate timestamps were found.",
                                    details={"duplicate_timestamp_count": duplicates},
                                )
                            )
                else:
                    issues.append(
                        DataQualityIssue(
                            code="TIMESTAMP_COLUMN_UNRESOLVED",
                            severity="WARNING",
                            entity_id=entity.entity_id,
                            message="No timestamp column could be inferred.",
                        )
                    )
                try:
                    source.read_labels(entity.entity_id, limit=max_rows_per_entity)
                except Exception as exc:
                    issues.append(
                        DataQualityIssue(
                            code="LABEL_READ_FAILED",
                            severity="WARNING",
                            entity_id=entity.entity_id,
                            message="Labels could not be read.",
                            details={"error": f"{type(exc).__name__}: {exc}"},
                        )
                    )
            except Exception as exc:
                issues.append(
                    DataQualityIssue(
                        code="ENTITY_READ_FAILED",
                        severity="ERROR",
                        entity_id=entity.entity_id,
                        message="The entity could not be read.",
                        details={"error": f"{type(exc).__name__}: {exc}"},
                    )
                )
        if len(entities) > max_entities:
            issues.append(
                DataQualityIssue(
                    code="ENTITY_INSPECTION_BOUNDED",
                    severity="INFO",
                    message="Only a bounded subset of entities was inspected.",
                    details={
                        "entity_count": len(entities),
                        "inspected_entity_count": max_entities,
                    },
                )
            )
        severities = {item.severity for item in issues}
        status = "FAIL" if "ERROR" in severities else ("WARNING" if "WARNING" in severities else "PASS")
        return DataSourceValidationReport(
            source_type=metadata.source_type,
            status=status,
            entity_count=len(entities),
            inspected_entity_count=inspected,
            channel_count_min=min(channel_counts) if channel_counts else 0,
            channel_count_max=max(channel_counts) if channel_counts else 0,
            row_count_total=row_total,
            issues=issues,
            metadata={
                "max_entities": max_entities,
                "max_rows_per_entity": max_rows_per_entity,
                "source_metadata": metadata.model_dump(mode="json"),
            },
        )
    finally:
        if opened_here:
            source.close()
