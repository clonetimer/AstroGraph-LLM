from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import pandas as pd

from .models import DataEntity, DataSourceMetadata, DatasetSourceConfig


class DatasetSource(ABC):
    """Read-only lifecycle-controlled time-series source.

    Implementations must not mutate the configured root. Calls are rejected before
    ``open`` and after ``close`` so adapters cannot silently continue with stale
    file handles or discovery state.
    """

    adapter_version = "0.5.30-r4f3.2"

    def __init__(self, config: DatasetSourceConfig) -> None:
        self.config = config
        self._opened = False

    @property
    def opened(self) -> bool:
        return self._opened

    @property
    def root(self) -> Path:
        if not self.config.root:
            raise ValueError("file-backed source requires a root path")
        return Path(self.config.root).expanduser().resolve()

    @property
    def supports_file_fingerprints(self) -> bool:
        """Whether entities are backed by files under ``root``.

        Probe code uses this explicit capability instead of catching every file
        access exception and incorrectly treating permission errors or corrupted
        paths as an in-memory source.
        """

        return True

    def _require_open(self) -> None:
        if not self._opened:
            raise RuntimeError("dataset source is not open")

    @staticmethod
    def resolve_under(root: Path, relative: str | Path) -> Path:
        candidate = (root / Path(relative)).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError("dataset path escapes configured root") from exc
        return candidate

    @abstractmethod
    def open(self) -> "DatasetSource":
        ...

    def close(self) -> None:
        self._opened = False

    def __enter__(self) -> "DatasetSource":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @abstractmethod
    def list_entities(self) -> list[DataEntity]:
        ...

    @abstractmethod
    def list_channels(self, entity_id: str) -> list[str]:
        ...

    @abstractmethod
    def read_range(
        self,
        entity_id: str,
        start: object | None = None,
        end: object | None = None,
        channels: list[str] | None = None,
        *,
        limit: int | None = None,
    ) -> pd.DataFrame:
        ...

    def read_window(
        self,
        entity_id: str,
        start: object | None,
        end: object | None,
        channels: list[str] | None = None,
        *,
        limit: int | None = None,
    ) -> pd.DataFrame:
        return self.read_range(entity_id, start, end, channels, limit=limit)

    def iter_range(
        self,
        entity_id: str,
        start: object | None = None,
        end: object | None = None,
        channels: list[str] | None = None,
        *,
        chunk_size: int = 100_000,
    ) -> Iterator[pd.DataFrame]:
        frame = self.read_range(entity_id, start, end, channels)
        for offset in range(0, len(frame), chunk_size):
            yield frame.iloc[offset : offset + chunk_size].copy()

    @abstractmethod
    def read_labels(
        self,
        entity_id: str,
        start: object | None = None,
        end: object | None = None,
        *,
        limit: int | None = None,
    ) -> pd.DataFrame:
        ...

    @abstractmethod
    def get_metadata(self) -> DataSourceMetadata:
        ...
