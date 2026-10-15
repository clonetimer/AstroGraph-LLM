from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Iterator
import zipfile

import pandas as pd

from .file_source import LocalFileDataSource, _infer_timestamp_column, _time_mask
from .models import DataEntity, DataSourceMetadata, DataSourceType, DatasetSourceConfig

_METADATA_FILENAMES = {
    "dataset_manifest.json",
    "run_manifest.json",
    "schema_profile.json",
    "time_profile.json",
    "label_profile.json",
    "quality_profile.json",
    "file_profile.json",
    "manifest.json",
    "metadata.json",
    "config.json",
    # ESA Anomalies Dataset mission-level metadata tables.
    "channels.csv",
    "telecommands.csv",
    "labels.csv",
    "anomaly_types.csv",
}

_MISSION_REQUIRED_FILES = {
    "channels.csv": {"Channel", "Subsystem", "Physical Unit", "Group", "Target"},
    "labels.csv": {"ID", "Channel", "StartTime", "EndTime"},
    "anomaly_types.csv": {
        "ID",
        "Class",
        "Subclass",
        "Category",
        "Dimensionality",
        "Locality",
        "Length",
    },
}
_TELECOMMAND_REQUIRED_COLUMNS = {"Telecommand", "Priority"}
_TIMESTAMP_OUTPUT_COLUMN = "timestamp"


def _natural_name_key(value: str) -> tuple[Any, ...]:
    parts: list[Any] = []
    buffer = ""
    digit_mode: bool | None = None
    for character in value:
        is_digit = character.isdigit()
        if digit_mode is None or is_digit == digit_mode:
            buffer += character
            digit_mode = is_digit
            continue
        parts.append(int(buffer) if digit_mode else buffer.lower())
        buffer = character
        digit_mode = is_digit
    if buffer:
        parts.append(int(buffer) if digit_mode else buffer.lower())
    return tuple(parts)


def _records_by_key(frame: pd.DataFrame, key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if key not in frame.columns:
        return result
    for _, row in frame.iterrows():
        name = str(row[key]).strip()
        if not name or name.lower() == "nan":
            continue
        payload: dict[str, Any] = {}
        for column, value in row.items():
            if pd.isna(value):
                payload[str(column)] = None
            elif hasattr(value, "item"):
                try:
                    payload[str(column)] = value.item()
                except Exception:
                    payload[str(column)] = value
            else:
                payload[str(column)] = value
        result[name] = payload
    return result


class ESAADBDataSource(LocalFileDataSource):
    """Read-only adapter for generic tabular archives and raw ESA-AD missions.

    Raw ESA mission channel/telecommand archives are ZIP-compressed pandas pickle
    objects. Loading pickle can execute code, so mission mode is fail-closed until
    ``config.metadata['trusted_pickle']`` is explicitly true. Only archives whose
    names are allow-listed by ``channels.csv`` or ``telecommands.csv`` are exposed.
    """

    supported_formats = sorted(set(LocalFileDataSource.supported_formats) | {"pickle_zip"})
    adapter_version = "0.5.30-r4f3.2"

    def __init__(
        self,
        root: str | Path | DatasetSourceConfig,
        *,
        profile_path: str | Path | None = None,
        timestamp_column: str | None = None,
        label_columns: list[str] | None = None,
        trusted_pickle: bool = False,
    ) -> None:
        if isinstance(root, DatasetSourceConfig):
            config = root
            if config.source_type != DataSourceType.ESA_ADB:
                raise ValueError("ESAADBDataSource requires source_type=esa_adb")
            if not config.read_only:
                raise ValueError("ESA-ADB adapter is always read-only")
        else:
            config = DatasetSourceConfig(
                source_type=DataSourceType.ESA_ADB,
                root=str(root),
                profile_path=None if profile_path is None else str(profile_path),
                timestamp_column=timestamp_column,
                label_columns=label_columns or [],
                read_only=True,
                lazy=True,
                recursive=True,
                metadata={"trusted_pickle": bool(trusted_pickle)},
            )
        super().__init__(config)
        self._mission_mode = False
        self._entity_kinds: dict[str, str] = {}
        self._raw_entity_names: dict[str, str] = {}
        self._mission_tables: dict[str, pd.DataFrame] = {}
        self._mission_discovery: dict[str, Any] = {}
        self._frame_cache: tuple[str, pd.DataFrame] | None = None

    @property
    def mission_mode(self) -> bool:
        return self._mission_mode

    def _is_raw_mission_layout(self) -> bool:
        return (
            (self.root / "channels").is_dir()
            and (self.root / "channels.csv").is_file()
            and (self.root / "labels.csv").is_file()
            and (self.root / "anomaly_types.csv").is_file()
        )

    def _trusted_pickle_acknowledged(self) -> bool:
        return bool(self.config.metadata.get("trusted_pickle", False))

    @staticmethod
    def _read_required_csv(path: Path, required: set[str]) -> pd.DataFrame:
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        missing = sorted(required - set(map(str, frame.columns)))
        if missing:
            raise ValueError(f"ESA-ADB metadata file {path.name} is missing columns: {missing}")
        return frame

    def _inspect_archive(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"ESA-ADB archive not found: {path.name}")
        if not zipfile.is_zipfile(path):
            raise ValueError(f"ESA-ADB archive is not a valid ZIP file: {path.name}")
        max_uncompressed = int(
            self.config.metadata.get("max_pickle_uncompressed_bytes", 8 * 1024**3)
        )
        max_ratio = float(self.config.metadata.get("max_pickle_compression_ratio", 1000.0))
        with zipfile.ZipFile(path, "r") as archive:
            entries = [item for item in archive.infolist() if not item.is_dir()]
            if len(entries) != 1:
                raise ValueError(
                    f"ESA-ADB pickle archive must contain exactly one file: {path.name}"
                )
            entry = entries[0]
            pure = PurePosixPath(entry.filename)
            if pure.is_absolute() or any(part in {"..", ""} for part in pure.parts):
                raise ValueError(f"unsafe path inside ESA-ADB archive: {path.name}")
            if entry.flag_bits & 0x1:
                raise ValueError(f"encrypted ESA-ADB archive is unsupported: {path.name}")
            if entry.file_size > max_uncompressed:
                raise ValueError(
                    f"ESA-ADB archive exceeds configured uncompressed limit: {path.name}"
                )
            denominator = max(entry.compress_size, 1)
            ratio = float(entry.file_size) / denominator
            if ratio > max_ratio:
                raise ValueError(
                    f"ESA-ADB archive exceeds configured compression-ratio limit: {path.name}"
                )
            expected_entry = path.stem
            if pure.name != expected_entry:
                raise ValueError(
                    f"ESA-ADB archive entry must match archive stem ({expected_entry}): "
                    f"{path.name} contains {entry.filename}"
                )
            return {
                "entry_name": entry.filename,
                "compressed_size": int(entry.compress_size),
                "uncompressed_size": int(entry.file_size),
                "compression_ratio": ratio,
            }

    def _mission_entity_id(self, raw_name: str, kind: str) -> str:
        if raw_name not in self._raw_entity_names.values():
            return raw_name
        return f"{kind}::{raw_name}"

    def _discover_mission_archives(
        self,
        metadata_frame: pd.DataFrame,
        *,
        key_column: str,
        folder_name: str,
        kind: str,
    ) -> tuple[dict[str, Path], list[str], list[str], dict[str, dict[str, Any]]]:
        metadata_by_name = _records_by_key(metadata_frame, key_column)
        allowlisted_names = set(metadata_by_name)
        folder = self.root / folder_name
        actual: dict[str, Path] = {}
        if folder.is_dir():
            for path in folder.glob("*.zip"):
                if path.is_file():
                    actual[path.stem] = path.resolve()
        missing = sorted(allowlisted_names - set(actual), key=_natural_name_key)
        extras = sorted(set(actual) - allowlisted_names, key=_natural_name_key)
        exposed = {
            name: actual[name]
            for name in sorted(set(actual) & allowlisted_names, key=_natural_name_key)
        }
        for raw_name, path in exposed.items():
            entity_id = self._mission_entity_id(raw_name, kind)
            archive = self._inspect_archive(path)
            self._entity_paths[entity_id] = path
            self._entity_kinds[entity_id] = kind
            self._raw_entity_names[entity_id] = raw_name
            row_metadata = dict(metadata_by_name.get(raw_name, {}))
            row_metadata.update(
                {
                    "entity_kind": kind,
                    "raw_name": raw_name,
                    "archive_entry": archive["entry_name"],
                    "archive_compressed_size": archive["compressed_size"],
                    "archive_uncompressed_size": archive["uncompressed_size"],
                    "labels_available": False,
                    "pickle_trust_required": True,
                }
            )
            self._entities[entity_id] = DataEntity(
                entity_id=entity_id,
                relative_path=path.relative_to(self.root).as_posix(),
                format="pickle_zip",
                timestamp_column=_TIMESTAMP_OUTPUT_COLUMN,
                channel_count=1,
                label_columns=[],
                metadata=row_metadata,
            )
        return exposed, missing, extras, metadata_by_name

    def _open_raw_mission(self) -> "ESAADBDataSource":
        if not self._trusted_pickle_acknowledged():
            raise ValueError(
                "raw ESA-ADB mission archives contain pandas pickle objects; loading them "
                "can execute code. Re-run with explicit trusted_pickle acknowledgement only "
                "for a dataset obtained from a trusted ESA source"
            )
        self._load_profile()
        channels = self._read_required_csv(
            self.root / "channels.csv", _MISSION_REQUIRED_FILES["channels.csv"]
        )
        labels = self._read_required_csv(
            self.root / "labels.csv", _MISSION_REQUIRED_FILES["labels.csv"]
        )
        anomaly_types = self._read_required_csv(
            self.root / "anomaly_types.csv", _MISSION_REQUIRED_FILES["anomaly_types.csv"]
        )
        telecommands_path = self.root / "telecommands.csv"
        if telecommands_path.is_file():
            telecommands = self._read_required_csv(
                telecommands_path, _TELECOMMAND_REQUIRED_COLUMNS
            )
        else:
            telecommands = pd.DataFrame(columns=sorted(_TELECOMMAND_REQUIRED_COLUMNS))

        self._mission_tables = {
            "channels": channels,
            "telecommands": telecommands,
            "labels": labels,
            "anomaly_types": anomaly_types,
        }
        self._entities = {}
        self._entity_paths = {}
        self._entity_kinds = {}
        self._raw_entity_names = {}

        _, missing_channels, extra_channels, _ = self._discover_mission_archives(
            channels,
            key_column="Channel",
            folder_name="channels",
            kind="telemetry",
        )
        _, missing_commands, extra_commands, _ = self._discover_mission_archives(
            telecommands,
            key_column="Telecommand",
            folder_name="telecommands",
            kind="telecommand",
        )

        label_counts = labels["Channel"].astype(str).value_counts(dropna=False).to_dict()
        for entity_id, entity in list(self._entities.items()):
            raw_name = self._raw_entity_names[entity_id]
            count = int(label_counts.get(raw_name, 0))
            self._entities[entity_id] = entity.model_copy(
                update={
                    "metadata": {
                        **entity.metadata,
                        "labels_available": count > 0,
                        "label_interval_count": count,
                    }
                }
            )

        self._mission_discovery = {
            "mission_layout_detected": True,
            "telemetry_archive_entity_count": sum(
                item == "telemetry" for item in self._entity_kinds.values()
            ),
            "telecommand_archive_entity_count": sum(
                item == "telecommand" for item in self._entity_kinds.values()
            ),
            "missing_channel_archives": missing_channels,
            "extra_channel_archives_not_allowlisted": extra_channels,
            "missing_telecommand_archives": missing_commands,
            "extra_telecommand_archives_not_allowlisted": extra_commands,
            "label_interval_count": int(len(labels)),
            "anomaly_type_count": int(len(anomaly_types)),
        }
        self._mission_mode = True
        self._opened = True
        return self

    def open(self) -> "ESAADBDataSource":
        if self._opened:
            return self
        if not self.config.read_only:
            raise ValueError("built-in file data sources are read-only")
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root is not a directory: {self.root}")
        if self._is_raw_mission_layout():
            return self._open_raw_mission()
        self._mission_mode = False
        super().open()
        return self

    def close(self) -> None:
        super().close()
        self._entity_kinds.clear()
        self._raw_entity_names.clear()
        self._mission_tables.clear()
        self._mission_discovery.clear()
        self._frame_cache = None
        self._mission_mode = False

    def _candidate_files(self) -> list[Path]:
        return [
            path
            for path in super()._candidate_files()
            if path.name.lower() not in _METADATA_FILENAMES
        ]

    def _load_pickle_frame(self, entity_id: str) -> pd.DataFrame:
        if self._frame_cache is not None and self._frame_cache[0] == entity_id:
            return self._frame_cache[1]
        path = self._entity_paths[entity_id]
        self._inspect_archive(path)
        try:
            payload = pd.read_pickle(path, compression="infer")
        except Exception as exc:
            raise ValueError(
                f"failed to load trusted ESA-ADB pickle archive {path.name}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if isinstance(payload, pd.Series):
            raw_name = self._raw_entity_names[entity_id]
            frame = payload.to_frame(name=str(payload.name or raw_name))
        elif isinstance(payload, pd.DataFrame):
            frame = payload.copy()
        else:
            raise TypeError(
                f"ESA-ADB archive {path.name} contains {type(payload).__name__}, "
                "expected pandas DataFrame or Series"
            )
        frame.columns = [str(item) for item in frame.columns]
        raw_name = self._raw_entity_names[entity_id]

        timestamp = _infer_timestamp_column(
            list(frame.columns), self.config.timestamp_column
        )
        if timestamp is None:
            if isinstance(frame.index, pd.RangeIndex):
                raise ValueError(
                    f"ESA-ADB archive {path.name} has neither a timestamp column nor a "
                    "datetime index"
                )
            parsed = pd.to_datetime(frame.index, errors="coerce", utc=True)
            valid_fraction = float(pd.Series(parsed).notna().mean()) if len(parsed) else 0.0
            if valid_fraction < 0.95:
                raise ValueError(
                    f"ESA-ADB archive {path.name} index is not a usable timestamp axis"
                )
            frame.insert(0, _TIMESTAMP_OUTPUT_COLUMN, parsed)
            timestamp = _TIMESTAMP_OUTPUT_COLUMN
        elif timestamp != _TIMESTAMP_OUTPUT_COLUMN:
            if _TIMESTAMP_OUTPUT_COLUMN in frame.columns:
                raise ValueError(
                    f"ESA-ADB archive {path.name} contains conflicting timestamp columns"
                )
            frame = frame.rename(columns={timestamp: _TIMESTAMP_OUTPUT_COLUMN})
            timestamp = _TIMESTAMP_OUTPUT_COLUMN

        value_columns = [item for item in frame.columns if item != timestamp]
        if raw_name not in value_columns:
            if len(value_columns) == 1:
                frame = frame.rename(columns={value_columns[0]: raw_name})
                value_columns = [raw_name]
            else:
                raise ValueError(
                    f"ESA-ADB archive {path.name} does not expose its allow-listed signal "
                    f"column {raw_name!r}"
                )
        normalized = frame[[timestamp, raw_name]].reset_index(drop=True)
        self._frame_cache = (entity_id, normalized)
        return normalized

    def list_entities(self) -> list[DataEntity]:
        if not self._mission_mode:
            return super().list_entities()
        self._require_open()
        ordered = sorted(
            self._entities,
            key=lambda entity_id: (
                0 if self._entity_kinds.get(entity_id) == "telemetry" else 1,
                _natural_name_key(self._raw_entity_names.get(entity_id, entity_id)),
            ),
        )
        return [self._entities[item].model_copy(deep=True) for item in ordered]

    def list_channels(self, entity_id: str) -> list[str]:
        if not self._mission_mode:
            return super().list_channels(entity_id)
        self._entity(entity_id)
        return [self._raw_entity_names[entity_id]]

    def read_range(
        self,
        entity_id: str,
        start: object | None = None,
        end: object | None = None,
        channels: list[str] | None = None,
        *,
        limit: int | None = None,
    ) -> pd.DataFrame:
        if not self._mission_mode:
            return super().read_range(entity_id, start, end, channels, limit=limit)
        entity = self._entity(entity_id)
        available = self.list_channels(entity_id)
        selected = list(channels) if channels is not None else available
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise KeyError(f"unknown channels for {entity_id}: {unknown}")
        frame = self._load_pickle_frame(entity_id)
        if start is not None or end is not None:
            frame = frame.loc[_time_mask(frame[entity.timestamp_column], start, end)]
        columns = [entity.timestamp_column, *selected]
        result = frame[columns]
        if limit is not None:
            result = result.head(limit)
        return result.reset_index(drop=True).copy()

    def iter_range(
        self,
        entity_id: str,
        start: object | None = None,
        end: object | None = None,
        channels: list[str] | None = None,
        *,
        chunk_size: int = 100_000,
    ) -> Iterator[pd.DataFrame]:
        if not self._mission_mode:
            yield from super().iter_range(
                entity_id, start, end, channels, chunk_size=chunk_size
            )
            return
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        frame = self.read_range(entity_id, start, end, channels)
        for offset in range(0, len(frame), chunk_size):
            yield frame.iloc[offset : offset + chunk_size].copy()

    def read_labels(
        self,
        entity_id: str,
        start: object | None = None,
        end: object | None = None,
        *,
        limit: int | None = None,
    ) -> pd.DataFrame:
        if not self._mission_mode:
            return super().read_labels(entity_id, start, end, limit=limit)
        self._entity(entity_id)
        raw_name = self._raw_entity_names[entity_id]
        labels = self._mission_tables["labels"]
        selected = labels.loc[labels["Channel"].astype(str) == raw_name].copy()
        anomaly_types = self._mission_tables["anomaly_types"]
        selected = selected.merge(
            anomaly_types,
            on="ID",
            how="left",
            validate="many_to_one",
            suffixes=("", "_AnomalyType"),
        )
        selected["StartTime"] = pd.to_datetime(
            selected["StartTime"], errors="coerce", utc=True
        )
        selected["EndTime"] = pd.to_datetime(
            selected["EndTime"], errors="coerce", utc=True
        )
        if start is not None:
            start_ts = pd.to_datetime(start, utc=True)
            selected = selected.loc[selected["EndTime"] >= start_ts]
        if end is not None:
            end_ts = pd.to_datetime(end, utc=True)
            selected = selected.loc[selected["StartTime"] <= end_ts]
        selected = selected.sort_values(["StartTime", "EndTime", "ID"], kind="stable")
        if limit is not None:
            selected = selected.head(limit)
        return selected.reset_index(drop=True).copy()

    def get_metadata(self) -> DataSourceMetadata:
        if not self._mission_mode:
            metadata = super().get_metadata()
            return metadata.model_copy(
                update={
                    "source_type": DataSourceType.ESA_ADB,
                    "root_token": "${ASTROGRAPH_ESA_ADB_ROOT}",
                    "adapter_version": self.adapter_version,
                    "metadata": {
                        **metadata.metadata,
                        "adapter_mode": "profile-driven-structural",
                        "local_acceptance_required": True,
                        "raw_data_exported": False,
                        "mission_layout_detected": False,
                    },
                    "claim_boundary": (
                        "The adapter is structurally capable of reading discovered local "
                        "tabular files. Exact ESA-ADB revision/schema compatibility remains "
                        "unverified until the local probe and acceptance suite pass."
                    ),
                }
            )
        self._require_open()
        return DataSourceMetadata(
            source_type=DataSourceType.ESA_ADB,
            opened=True,
            read_only=True,
            root_token="${ASTROGRAPH_ESA_ADB_ROOT}",
            entity_count=len(self._entities),
            supported_formats=self.supported_formats,
            adapter_version=self.adapter_version,
            metadata={
                "adapter_mode": "raw-esa-ad-mission-pickle-zip",
                "local_acceptance_required": True,
                "raw_data_exported": False,
                "trusted_pickle_acknowledged": True,
                "partial_pickle_reads_supported": False,
                **self._mission_discovery,
            },
            claim_boundary=(
                "Mission mode proves only bounded readability of explicitly trusted local "
                "ESA pickle archives and correct metadata/interval-label association. Pickle "
                "archives are loaded per channel and cannot be streamed partially. No anomaly "
                "detection accuracy, physical validity, or flight qualification is implied."
            ),
        )
