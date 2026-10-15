from __future__ import annotations

from typing import Any

import pandas as pd

from .base import DatasetSource
from .file_source import _infer_label_columns, _infer_timestamp_column, _time_mask
from .models import DataEntity, DataSourceMetadata, DataSourceType, DatasetSourceConfig


class InMemoryDataSource(DatasetSource):
    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        labels: dict[str, pd.DataFrame] | None = None,
        timestamp_column: str | None = None,
    ) -> None:
        super().__init__(
            DatasetSourceConfig(
                source_type=DataSourceType.MEMORY,
                read_only=True,
                lazy=False,
                timestamp_column=timestamp_column,
            )
        )
        self._frames = {key: value.copy(deep=True) for key, value in frames.items()}
        self._labels = {
            key: value.copy(deep=True) for key, value in (labels or {}).items()
        }
        self._entities: dict[str, DataEntity] = {}

    @property
    def supports_file_fingerprints(self) -> bool:
        return False

    def open(self) -> "InMemoryDataSource":
        if self._opened:
            return self
        entities: dict[str, DataEntity] = {}
        for entity_id, frame in sorted(self._frames.items()):
            columns = [str(item) for item in frame.columns]
            timestamp = _infer_timestamp_column(columns, self.config.timestamp_column)
            labels = _infer_label_columns(columns, [])
            entities[entity_id] = DataEntity(
                entity_id=entity_id,
                relative_path=f"memory://{entity_id}",
                format="memory",
                timestamp_column=timestamp,
                row_count=len(frame),
                channel_count=len([c for c in columns if c != timestamp and c not in labels]),
                label_columns=labels,
            )
        self._entities = entities
        self._opened = True
        return self

    def close(self) -> None:
        super().close()
        self._entities.clear()

    def _entity(self, entity_id: str) -> DataEntity:
        self._require_open()
        if entity_id not in self._entities:
            raise KeyError(f"unknown dataset entity: {entity_id}")
        return self._entities[entity_id]

    def list_entities(self) -> list[DataEntity]:
        self._require_open()
        return [self._entities[key].model_copy(deep=True) for key in sorted(self._entities)]

    def list_channels(self, entity_id: str) -> list[str]:
        entity = self._entity(entity_id)
        excluded = set(entity.label_columns)
        if entity.timestamp_column:
            excluded.add(entity.timestamp_column)
        return [str(column) for column in self._frames[entity_id].columns if str(column) not in excluded]

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
        available = self.list_channels(entity_id)
        selected = available if channels is None else list(channels)
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise KeyError(f"unknown channels for {entity_id}: {unknown}")
        usecols = list(selected)
        if entity.timestamp_column and entity.timestamp_column not in usecols:
            usecols.insert(0, entity.timestamp_column)
        frame = self._frames[entity_id][usecols].copy()
        if start is not None or end is not None:
            if not entity.timestamp_column:
                raise ValueError(f"entity {entity_id} has no timestamp column")
            frame = frame.loc[_time_mask(frame[entity.timestamp_column], start, end)]
        if limit is not None:
            frame = frame.head(limit)
        return frame.reset_index(drop=True).copy()

    def read_labels(
        self,
        entity_id: str,
        start: object | None = None,
        end: object | None = None,
        *,
        limit: int | None = None,
    ) -> pd.DataFrame:
        entity = self._entity(entity_id)
        if entity_id in self._labels:
            frame = self._labels[entity_id].copy()
        else:
            columns = ([entity.timestamp_column] if entity.timestamp_column else []) + entity.label_columns
            frame = self._frames[entity_id][[c for c in columns if c]].copy()
        if start is not None or end is not None:
            if not entity.timestamp_column or entity.timestamp_column not in frame.columns:
                raise ValueError(f"labels for entity {entity_id} have no timestamp column")
            frame = frame.loc[_time_mask(frame[entity.timestamp_column], start, end)]
        if limit is not None:
            frame = frame.head(limit)
        return frame.reset_index(drop=True).copy()

    def get_metadata(self) -> DataSourceMetadata:
        self._require_open()
        return DataSourceMetadata(
            source_type=DataSourceType.MEMORY,
            opened=True,
            read_only=True,
            root_token="memory://",
            entity_count=len(self._entities),
            supported_formats=["memory"],
            metadata={"copied_on_input": True},
        )
