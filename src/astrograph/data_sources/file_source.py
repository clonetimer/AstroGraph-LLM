from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .base import DatasetSource
from .models import DataEntity, DataSourceMetadata, DataSourceType, DatasetSourceConfig


_TIMESTAMP_CANDIDATES = (
    "time_s",
    "timestamp",
    "time",
    "datetime",
    "date",
    "utc",
    "epoch",
    "index",
)
_LABEL_FILE_STEMS = {
    "label",
    "labels",
    "annotation",
    "annotations",
    "event",
    "events",
    "ground_truth",
    "truth",
    "targets",
}
_LABEL_FILE_SUFFIXES = (
    "_label",
    "_labels",
    "_annotation",
    "_annotations",
    "_events",
    "_ground_truth",
)
_SUPPORTED_SUFFIXES = {".csv", ".jsonl", ".ndjson", ".json", ".parquet", ".feather"}


def _safe_entity_id(relative: Path) -> str:
    text = relative.as_posix().strip("/")
    return text or "root"


def _looks_like_label_file(path: Path) -> bool:
    """Identify only dedicated annotation files.

    Fault/anomaly words in a telemetry filename often describe the scenario
    (for example ``reaction_wheel_fault.csv``) rather than a label-only file.
    Broad substring matching therefore drops real entities in mixed archives.
    """

    stem = path.stem.lower()
    return stem in _LABEL_FILE_STEMS or stem.endswith(_LABEL_FILE_SUFFIXES)


def _infer_timestamp_column(columns: list[str], configured: str | None = None) -> str | None:
    if configured:
        return configured if configured in columns else None
    lowered = {str(column).lower(): str(column) for column in columns}
    for candidate in _TIMESTAMP_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    for column in columns:
        name = str(column).lower()
        if any(token in name for token in ("timestamp", "time_", "_time", "datetime")):
            return str(column)
    return None


def _infer_label_columns(columns: list[str], configured: list[str]) -> list[str]:
    if configured:
        return [column for column in configured if column in columns]
    result = []
    for column in columns:
        lowered = str(column).lower()
        if lowered.startswith(("label.", "label_", "truth.", "truth_", "ground_truth.", "ground_truth_")):
            result.append(str(column))
        elif lowered in {
            "label",
            "anomaly",
            "is_anomaly",
            "fault",
            "target",
            "ground_truth",
            "truth",
        }:
            result.append(str(column))
    return sorted(set(result))


def _time_mask(series: pd.Series, start: object | None, end: object | None) -> pd.Series:
    if start is None and end is None:
        return pd.Series(True, index=series.index)
    # pandas converts datetime64 values to integer nanoseconds in to_numeric(),
    # which would misclassify a genuine datetime axis as numeric. Preserve the
    # semantic dtype before attempting the numeric heuristic.
    is_datetime = bool(
        pd.api.types.is_datetime64_any_dtype(series.dtype)
        or isinstance(series.dtype, pd.DatetimeTZDtype)
    )
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_fraction = float(numeric.notna().mean()) if len(numeric) else 0.0
    if not is_datetime and numeric_fraction >= 0.95:
        try:
            start_value = None if start is None else float(start)
            end_value = None if end is None else float(end)
        except (TypeError, ValueError) as exc:
            raise ValueError("numeric timestamp column requires numeric start/end") from exc
        mask = numeric.notna()
        if start_value is not None:
            mask &= numeric >= start_value
        if end_value is not None:
            mask &= numeric <= end_value
        return mask
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    mask = parsed.notna()
    if start is not None:
        start_ts = pd.to_datetime(start, utc=True)
        mask &= parsed >= start_ts
    if end is not None:
        end_ts = pd.to_datetime(end, utc=True)
        mask &= parsed <= end_ts
    return mask


class LocalFileDataSource(DatasetSource):
    """Generic read-only source for local tabular time-series files.

    The adapter intentionally performs structural discovery only. Domain-specific
    semantics must be supplied through a profile or an explicit mapping.
    """

    supported_formats = sorted(_SUPPORTED_SUFFIXES)

    def __init__(self, config: DatasetSourceConfig) -> None:
        super().__init__(config)
        self._entities: dict[str, DataEntity] = {}
        self._entity_paths: dict[str, Path] = {}
        self._label_paths: dict[str, Path] = {}
        self._columns_cache: dict[str, list[str]] = {}
        self._profile: dict[str, Any] = {}

    def _matches_filters(self, relative: Path) -> bool:
        text = relative.as_posix()
        if self.config.include_globs and not any(
            fnmatch.fnmatch(text, pattern) for pattern in self.config.include_globs
        ):
            return False
        if self.config.exclude_globs and any(
            fnmatch.fnmatch(text, pattern) for pattern in self.config.exclude_globs
        ):
            return False
        return True

    def _load_profile(self) -> None:
        if not self.config.profile_path:
            self._profile = {}
            return
        path = Path(self.config.profile_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"data-source profile not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("data-source profile must be a JSON object")
        self._profile = payload

    def _candidate_files(self) -> list[Path]:
        iterator = self.root.rglob("*") if self.config.recursive else self.root.glob("*")
        files = []
        for path in iterator:
            if not path.is_file() or path.suffix.lower() not in _SUPPORTED_SUFFIXES:
                continue
            relative = path.relative_to(self.root)
            if self._matches_filters(relative):
                files.append(path)
        return sorted(files, key=lambda item: item.relative_to(self.root).as_posix())

    @staticmethod
    def _entity_id_for(path: Path, root: Path) -> str:
        relative = path.relative_to(root)
        if path.name.lower() in {"telemetry.csv", "telemetry.parquet", "data.csv", "data.parquet"}:
            return _safe_entity_id(relative.parent)
        return _safe_entity_id(relative.with_suffix(""))

    def open(self) -> "LocalFileDataSource":
        if self._opened:
            return self
        if not self.config.read_only:
            raise ValueError("built-in file data sources are read-only")
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root is not a directory: {self.root}")
        self._load_profile()
        data_candidates: list[Path] = []
        label_candidates: list[Path] = []
        for path in self._candidate_files():
            if _looks_like_label_file(path):
                label_candidates.append(path)
            else:
                data_candidates.append(path)

        for path in data_candidates:
            entity_id = self._entity_id_for(path, self.root)
            # Prefer canonical telemetry/data file over arbitrary siblings.
            previous = self._entity_paths.get(entity_id)
            if previous is not None:
                preferred_names = {"telemetry.csv", "telemetry.parquet", "data.csv", "data.parquet"}
                if previous.name.lower() in preferred_names:
                    continue
                if path.name.lower() not in preferred_names:
                    continue
            self._entity_paths[entity_id] = path

        # If a dataset consists only of label-looking files, keep them visible as
        # entities rather than reporting an empty source.
        if not self._entity_paths:
            for path in label_candidates:
                entity_id = self._entity_id_for(path, self.root)
                self._entity_paths[entity_id] = path

        for path in label_candidates:
            relative = path.relative_to(self.root)
            parent_id = _safe_entity_id(relative.parent)
            if parent_id in self._entity_paths:
                self._label_paths[parent_id] = path
                continue
            stem_id = _safe_entity_id(relative.with_suffix(""))
            for entity_id, data_path in self._entity_paths.items():
                if data_path.parent == path.parent:
                    self._label_paths.setdefault(entity_id, path)
                elif entity_id == stem_id:
                    self._label_paths[entity_id] = path

        self._entities = {}
        for entity_id, path in sorted(self._entity_paths.items()):
            columns = self._read_columns(path)
            timestamp = self._configured_timestamp_for(entity_id, columns)
            labels = self._configured_labels_for(entity_id, columns)
            self._entities[entity_id] = DataEntity(
                entity_id=entity_id,
                relative_path=path.relative_to(self.root).as_posix(),
                format=path.suffix.lower().lstrip("."),
                timestamp_column=timestamp,
                channel_count=len([item for item in columns if item != timestamp and item not in labels]),
                label_columns=labels,
                metadata={"label_file_present": entity_id in self._label_paths},
            )
        self._opened = True
        return self

    def close(self) -> None:
        super().close()
        self._entities.clear()
        self._entity_paths.clear()
        self._label_paths.clear()
        self._columns_cache.clear()

    def _entity_profile(self, entity_id: str) -> dict[str, Any]:
        """Return the profile entry for one discovered entity, if present.

        Probe-generated profiles are entity-scoped because heterogeneous archives
        can use different timestamp/label columns per file. Falling back to the
        top-level summary remains supported for hand-written profiles.
        """

        entities = self._profile.get("entities", [])
        if not isinstance(entities, list):
            return {}
        for item in entities:
            if isinstance(item, dict) and str(item.get("entity_id", "")) == entity_id:
                return item
        return {}

    def _configured_timestamp_for(self, entity_id: str, columns: list[str]) -> str | None:
        entity_profile = self._entity_profile(entity_id)
        profile_timestamp = (
            entity_profile.get("timestamp_column")
            or self._profile.get("timestamp_column")
            or self._profile.get("schema", {}).get("timestamp_column")
        )
        return _infer_timestamp_column(
            columns,
            self.config.timestamp_column or (str(profile_timestamp) if profile_timestamp else None),
        )

    def _configured_labels_for(self, entity_id: str, columns: list[str]) -> list[str]:
        entity_profile = self._entity_profile(entity_id)
        profile_labels = (
            entity_profile.get("label_columns")
            or self._profile.get("label_columns")
            or self._profile.get("schema", {}).get("label_columns")
            or []
        )
        configured = self.config.label_columns or [str(item) for item in profile_labels]
        return _infer_label_columns(columns, configured)

    def _read_columns(self, path: Path) -> list[str]:
        key = str(path)
        if key in self._columns_cache:
            return list(self._columns_cache[key])
        suffix = path.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
        elif suffix in {".jsonl", ".ndjson"}:
            frame = pd.read_json(path, lines=True, nrows=1)
        elif suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(payload, list):
                frame = pd.DataFrame(payload[:1])
            elif isinstance(payload, dict):
                records = payload.get("records") or payload.get("data") or payload.get("rows")
                frame = pd.DataFrame((records or [])[:1]) if isinstance(records, list) else pd.DataFrame([payload])
            else:
                frame = pd.DataFrame()
        elif suffix == ".parquet":
            try:
                frame = pd.read_parquet(path).head(0)
            except ImportError as exc:
                raise RuntimeError("parquet support requires pyarrow or fastparquet") from exc
        elif suffix == ".feather":
            try:
                frame = pd.read_feather(path).head(0)
            except ImportError as exc:
                raise RuntimeError("feather support requires pyarrow") from exc
        else:  # pragma: no cover - filtered during discovery
            raise ValueError(f"unsupported data file format: {suffix}")
        columns = [str(item) for item in frame.columns]
        self._columns_cache[key] = columns
        return list(columns)

    def _read_file(
        self,
        path: Path,
        *,
        usecols: list[str] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path, usecols=usecols, nrows=limit, encoding="utf-8-sig", low_memory=False)
        if suffix in {".jsonl", ".ndjson"}:
            frame = pd.read_json(path, lines=True)
        elif suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(payload, list):
                frame = pd.DataFrame(payload)
            elif isinstance(payload, dict):
                records = payload.get("records") or payload.get("data") or payload.get("rows")
                frame = pd.DataFrame(records) if isinstance(records, list) else pd.DataFrame([payload])
            else:
                frame = pd.DataFrame()
        elif suffix == ".parquet":
            try:
                frame = pd.read_parquet(path, columns=usecols)
            except ImportError as exc:
                raise RuntimeError("parquet support requires pyarrow or fastparquet") from exc
        elif suffix == ".feather":
            try:
                frame = pd.read_feather(path, columns=usecols)
            except ImportError as exc:
                raise RuntimeError("feather support requires pyarrow") from exc
        else:
            raise ValueError(f"unsupported data file format: {suffix}")
        if usecols is not None:
            missing = [item for item in usecols if item not in frame.columns]
            if missing:
                raise KeyError(f"requested columns not found: {missing}")
            frame = frame[usecols]
        if limit is not None:
            frame = frame.head(limit)
        return frame

    def _entity(self, entity_id: str) -> DataEntity:
        self._require_open()
        try:
            return self._entities[entity_id]
        except KeyError as exc:
            raise KeyError(f"unknown dataset entity: {entity_id}") from exc

    def list_entities(self) -> list[DataEntity]:
        self._require_open()
        return [self._entities[key].model_copy(deep=True) for key in sorted(self._entities)]

    def list_channels(self, entity_id: str) -> list[str]:
        entity = self._entity(entity_id)
        columns = self._read_columns(self._entity_paths[entity_id])
        excluded = set(entity.label_columns)
        if entity.timestamp_column:
            excluded.add(entity.timestamp_column)
        return [column for column in columns if column not in excluded]

    def read_range(
        self,
        entity_id: str,
        start: object | None = None,
        end: object | None = None,
        channels: list[str] | None = None,
        *,
        limit: int | None = None,
    ) -> pd.DataFrame:
        entity = self._entity(entity_id)
        path = self._entity_paths[entity_id]
        available_channels = self.list_channels(entity_id)
        if channels is not None:
            unknown = sorted(set(channels) - set(available_channels))
            if unknown:
                raise KeyError(f"unknown channels for {entity_id}: {unknown}")
        selected = list(channels) if channels is not None else available_channels
        usecols = list(selected)
        if entity.timestamp_column and entity.timestamp_column not in usecols:
            usecols.insert(0, entity.timestamp_column)
        # Read labels only through read_labels unless no channel selection was made
        # and the caller explicitly listed them.
        read_limit = limit if start is None and end is None else None
        frame = self._read_file(path, usecols=usecols or None, limit=read_limit)
        if (start is not None or end is not None):
            if not entity.timestamp_column:
                raise ValueError(f"entity {entity_id} has no timestamp column")
            frame = frame.loc[_time_mask(frame[entity.timestamp_column], start, end)]
        if limit is not None:
            frame = frame.head(limit)
        return frame.reset_index(drop=True).copy()

    def iter_range(
        self,
        entity_id: str,
        start: object | None = None,
        end: object | None = None,
        channels: list[str] | None = None,
        *,
        chunk_size: int = 100_000,
    ) -> Iterator[pd.DataFrame]:
        entity = self._entity(entity_id)
        path = self._entity_paths[entity_id]
        if path.suffix.lower() != ".csv":
            yield from super().iter_range(entity_id, start, end, channels, chunk_size=chunk_size)
            return
        available = self.list_channels(entity_id)
        selected = list(channels) if channels is not None else available
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise KeyError(f"unknown channels for {entity_id}: {unknown}")
        usecols = list(selected)
        if entity.timestamp_column and entity.timestamp_column not in usecols:
            usecols.insert(0, entity.timestamp_column)
        for chunk in pd.read_csv(
            path,
            usecols=usecols or None,
            chunksize=chunk_size,
            encoding="utf-8-sig",
            low_memory=False,
        ):
            if start is not None or end is not None:
                if not entity.timestamp_column:
                    raise ValueError(f"entity {entity_id} has no timestamp column")
                chunk = chunk.loc[_time_mask(chunk[entity.timestamp_column], start, end)]
            if not chunk.empty:
                yield chunk.reset_index(drop=True).copy()

    def read_labels(
        self,
        entity_id: str,
        start: object | None = None,
        end: object | None = None,
        *,
        limit: int | None = None,
    ) -> pd.DataFrame:
        entity = self._entity(entity_id)
        if entity_id in self._label_paths:
            path = self._label_paths[entity_id]
            frame = self._read_file(path, limit=limit if start is None and end is None else None)
            timestamp = _infer_timestamp_column([str(item) for item in frame.columns], self.config.timestamp_column)
        else:
            path = self._entity_paths[entity_id]
            columns = self._read_columns(path)
            label_columns = self._configured_labels_for(entity_id, columns)
            timestamp = entity.timestamp_column
            if not label_columns:
                return pd.DataFrame(columns=[timestamp] if timestamp else [])
            usecols = list(label_columns)
            if timestamp and timestamp not in usecols:
                usecols.insert(0, timestamp)
            frame = self._read_file(
                path,
                usecols=usecols,
                limit=limit if start is None and end is None else None,
            )
        if start is not None or end is not None:
            if not timestamp or timestamp not in frame.columns:
                raise ValueError(f"labels for entity {entity_id} have no timestamp column")
            frame = frame.loc[_time_mask(frame[timestamp], start, end)]
        if limit is not None:
            frame = frame.head(limit)
        return frame.reset_index(drop=True).copy()

    def get_metadata(self) -> DataSourceMetadata:
        self._require_open()
        return DataSourceMetadata(
            source_type=self.config.source_type,
            opened=True,
            read_only=self.config.read_only,
            entity_count=len(self._entities),
            supported_formats=self.supported_formats,
            metadata={
                "lazy": self.config.lazy,
                "recursive": self.config.recursive,
                "profile_loaded": bool(self._profile),
            },
        )
