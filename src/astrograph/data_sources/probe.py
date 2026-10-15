from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .base import DatasetSource
from .models import ProbeArtifactSet, ProbeOptions
from .validation import validate_data_source


class DataSourceProbeError(RuntimeError):
    """Typed failure raised when a file-backed entity cannot be fingerprinted."""

    def __init__(
        self,
        message: str,
        *,
        entity_id: str,
        relative_path: str,
        operation: str,
    ) -> None:
        super().__init__(message)
        self.entity_id = entity_id
        self.relative_path = relative_path
        self.operation = operation


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_digest(relative_path: str, size_bytes: int, mtime_ns: int) -> str:
    payload = json.dumps(
        {
            "relative_path": relative_path,
            "size_bytes": size_bytes,
            "mtime_ns": mtime_ns,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float):
        if pd.isna(value):
            return None
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_scalar) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _time_summary(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    if len(series) and float(numeric.notna().mean()) >= 0.95:
        valid = numeric.dropna()
        kind = "numeric"
    else:
        valid = pd.to_datetime(series, errors="coerce", utc=True).dropna()
        kind = "datetime"
    if valid.empty:
        return {
            "kind": kind,
            "valid_count": 0,
            "invalid_count": int(len(series)),
            "minimum": None,
            "maximum": None,
            "median_step": None,
            "monotonic_increasing": False,
            "unique": False,
        }
    diffs = valid.diff().dropna()
    return {
        "kind": kind,
        "valid_count": int(valid.size),
        "invalid_count": int(len(series) - valid.size),
        "minimum": _json_scalar(valid.min()),
        "maximum": _json_scalar(valid.max()),
        "median_step": _json_scalar(diffs.median()) if not diffs.empty else None,
        "monotonic_increasing": bool(valid.is_monotonic_increasing),
        "unique": bool(valid.is_unique),
        "duplicate_count": int(valid.duplicated().sum()),
    }


def _column_profile(series: pd.Series, *, include_statistics: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dtype": str(series.dtype),
        "row_count": int(len(series)),
        "non_null_count": int(series.notna().sum()),
        "missing_count": int(series.isna().sum()),
        "unique_count": int(series.nunique(dropna=True)),
    }
    if include_statistics:
        numeric = pd.to_numeric(series, errors="coerce")
        if len(series) and float(numeric.notna().mean()) >= 0.95 and numeric.notna().any():
            payload["numeric_statistics"] = {
                "minimum": _json_scalar(numeric.min()),
                "maximum": _json_scalar(numeric.max()),
                "mean": _json_scalar(numeric.mean()),
                "std": _json_scalar(numeric.std(ddof=0)),
            }
    return payload


def probe_data_source(
    source: DatasetSource,
    output_dir: str | Path,
    *,
    options: ProbeOptions | None = None,
) -> ProbeArtifactSet:
    """Write redacted structural evidence without copying raw telemetry.

    The output contains relative paths, schemas, counts, quality summaries, and
    optional digests. No row values or absolute dataset paths are serialized.
    """

    options = options or ProbeOptions()
    destination = Path(output_dir).expanduser().resolve()
    opened_here = not source.opened
    if opened_here:
        source.open()
    try:
        # Keep probe evidence outside the immutable data root.
        try:
            destination.relative_to(source.root)
        except (ValueError, AttributeError):
            pass
        else:
            raise ValueError("probe output must not be written inside the dataset root")

        metadata = source.get_metadata()
        entities = source.list_entities()
        schema_entities: list[dict[str, Any]] = []
        time_entities: list[dict[str, Any]] = []
        label_entities: list[dict[str, Any]] = []
        file_entries: list[dict[str, Any]] = []
        hash_lines: list[str] = []

        for entity in entities[: options.max_entities]:
            channels = source.list_channels(entity.entity_id)
            frame = source.read_range(
                entity.entity_id,
                channels=channels,
                limit=options.max_rows_per_entity,
            )
            column_profiles = {
                str(column): _column_profile(
                    frame[column], include_statistics=options.include_statistics
                )
                for column in frame.columns
            }
            schema_entities.append(
                {
                    "entity_id": entity.entity_id,
                    "format": entity.format,
                    "timestamp_column": entity.timestamp_column,
                    "label_columns": entity.label_columns,
                    "channel_count": len(channels),
                    "inspected_row_count": int(len(frame)),
                    "columns": column_profiles,
                }
            )
            if entity.timestamp_column and entity.timestamp_column in frame.columns:
                summary = _time_summary(frame[entity.timestamp_column])
            else:
                summary = {
                    "kind": "unresolved",
                    "valid_count": 0,
                    "invalid_count": int(len(frame)),
                    "minimum": None,
                    "maximum": None,
                    "median_step": None,
                    "monotonic_increasing": False,
                    "unique": False,
                }
            time_entities.append({"entity_id": entity.entity_id, **summary})

            labels = source.read_labels(entity.entity_id, limit=options.max_rows_per_entity)
            label_columns = [
                str(column)
                for column in labels.columns
                if str(column) != entity.timestamp_column
            ]
            label_profile: dict[str, Any] = {
                "entity_id": entity.entity_id,
                "label_columns": label_columns,
                "row_count": int(len(labels)),
                "columns": {},
            }
            for column in label_columns:
                item = _column_profile(labels[column], include_statistics=False)
                if options.include_label_values:
                    counts = labels[column].astype(str).value_counts(dropna=False).head(100)
                    item["value_counts"] = {
                        str(key): int(value) for key, value in counts.items()
                    }
                label_profile["columns"][column] = item
            label_entities.append(label_profile)

            # File identity stays relative.  Virtuality is an explicit adapter
            # capability; file-backed failures are never downgraded to a virtual
            # source because that would hide permission, corruption, or path bugs.
            if not source.supports_file_fingerprints:
                file_entries.append(
                    {
                        "entity_id": entity.entity_id,
                        "relative_path": entity.relative_path,
                        "size_bytes": None,
                        "mtime_ns": None,
                        "digest_mode": "none",
                        "sha256": None,
                        "fingerprint_status": "NOT_APPLICABLE_VIRTUAL_SOURCE",
                    }
                )
                continue

            try:
                path = source.resolve_under(source.root, entity.relative_path)
                stat = path.stat()
                digest = None
                if options.hash_mode == "full":
                    digest = _sha256(path)
                elif options.hash_mode == "metadata":
                    digest = _metadata_digest(
                        entity.relative_path, stat.st_size, stat.st_mtime_ns
                    )
                entry = {
                    "entity_id": entity.entity_id,
                    "relative_path": entity.relative_path,
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "digest_mode": options.hash_mode,
                    "sha256": digest,
                    "fingerprint_status": "COMPLETE",
                }
                file_entries.append(entry)
                if digest:
                    hash_lines.append(f"{digest}  {entity.relative_path}")
            except (OSError, ValueError) as exc:
                raise DataSourceProbeError(
                    (
                        "file-backed dataset entity could not be fingerprinted: "
                        f"{entity.entity_id} ({entity.relative_path})"
                    ),
                    entity_id=entity.entity_id,
                    relative_path=entity.relative_path,
                    operation="RESOLVE_STAT_OR_HASH",
                ) from exc

        validation = validate_data_source(
            source,
            max_entities=options.max_entities,
            max_rows_per_entity=options.max_rows_per_entity,
        )
        common = {
            "generated_at": _now(),
            "source_type": metadata.source_type.value,
            "root": options.root_token,
            "raw_data_included": False,
            "absolute_paths_included": False,
            "entity_count": len(entities),
            "inspected_entity_count": min(len(entities), options.max_entities),
            "probe_options": options.model_dump(mode="json"),
            "claim_boundary": metadata.claim_boundary,
        }
        dataset_manifest = {
            "schema_version": "astrograph-dataset-probe-manifest.v1",
            **common,
            "source_metadata": metadata.model_dump(mode="json"),
            "files": file_entries,
        }
        schema_profile = {
            "schema_version": "astrograph-dataset-schema-profile.v1",
            **common,
            "entities": schema_entities,
            "timestamp_column": next(
                (
                    item["timestamp_column"]
                    for item in schema_entities
                    if item["timestamp_column"]
                ),
                None,
            ),
            "label_columns": sorted(
                {
                    label
                    for item in schema_entities
                    for label in item.get("label_columns", [])
                }
            ),
        }
        time_profile = {
            "schema_version": "astrograph-dataset-time-profile.v1",
            **common,
            "entities": time_entities,
        }
        label_profile = {
            "schema_version": "astrograph-dataset-label-profile.v1",
            **common,
            "label_values_included": options.include_label_values,
            "entities": label_entities,
        }
        quality_profile = {
            "schema_version": "astrograph-dataset-quality-profile.v1",
            **common,
            "validation": validation.model_dump(mode="json"),
        }
        file_profile = {
            "schema_version": "astrograph-dataset-file-profile.v1",
            **common,
            "hash_mode": options.hash_mode,
            "files": file_entries,
        }

        destination.mkdir(parents=True, exist_ok=True)
        payloads = {
            "dataset_manifest.json": dataset_manifest,
            "schema_profile.json": schema_profile,
            "time_profile.json": time_profile,
            "label_profile.json": label_profile,
            "quality_profile.json": quality_profile,
            "file_profile.json": file_profile,
        }
        for name, payload in payloads.items():
            _atomic_json(destination / name, payload)
        (destination / "file_hashes.txt").write_text(
            "\n".join(hash_lines) + ("\n" if hash_lines else ""),
            encoding="utf-8",
        )
        (destination / "probe.log").write_text(
            "\n".join(
                [
                    f"generated_at={common['generated_at']}",
                    f"source_type={common['source_type']}",
                    f"root={options.root_token}",
                    f"entity_count={common['entity_count']}",
                    f"inspected_entity_count={common['inspected_entity_count']}",
                    f"validation_status={validation.status}",
                    "raw_data_included=false",
                    "absolute_paths_included=false",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        files = {name: name for name in payloads}
        files.update({"file_hashes": "file_hashes.txt", "log": "probe.log"})
        return ProbeArtifactSet(output_dir="<local_evidence>", files=files)
    finally:
        if opened_here:
            source.close()
