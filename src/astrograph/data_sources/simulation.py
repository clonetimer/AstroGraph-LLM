from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .file_source import LocalFileDataSource, _safe_entity_id
from .models import DataEntity, DataSourceMetadata, DataSourceType, DatasetSourceConfig


class SimulationDataSource(LocalFileDataSource):
    """Read-only adapter for AstroGraph simulation run directories.

    Each entity is a directory containing ``telemetry.csv`` and, optionally,
    ``run_manifest.json``. Truth/label columns remain excluded from model channels
    and are only accessible through ``read_labels``.
    """

    def __init__(
        self,
        root: str | Path | DatasetSourceConfig,
        *,
        timestamp_column: str = "time_s",
    ) -> None:
        if isinstance(root, DatasetSourceConfig):
            config = root
            if config.source_type != DataSourceType.SIMULATION:
                raise ValueError("SimulationDataSource requires source_type=simulation")
        else:
            config = DatasetSourceConfig(
                source_type=DataSourceType.SIMULATION,
                root=str(root),
                timestamp_column=timestamp_column,
                read_only=True,
                lazy=True,
            )
        super().__init__(config)

    def _candidate_files(self) -> list[Path]:
        root = self.root
        if (root / "telemetry.csv").is_file():
            return [root / "telemetry.csv"]
        return sorted(root.rglob("telemetry.csv"), key=lambda p: p.relative_to(root).as_posix())

    @staticmethod
    def _entity_id_for(path: Path, root: Path) -> str:
        if path.parent == root:
            return "root"
        return _safe_entity_id(path.parent.relative_to(root))

    def open(self) -> "SimulationDataSource":
        super().open()
        enriched: dict[str, DataEntity] = {}
        for entity_id, entity in self._entities.items():
            path = self._entity_paths[entity_id]
            manifest_path = path.parent / "run_manifest.json"
            metadata: dict[str, Any] = dict(entity.metadata)
            if manifest_path.is_file():
                try:
                    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
                    metadata.update(
                        {
                            "manifest_present": True,
                            "scenario": payload.get("scenario"),
                            "duration_s": payload.get("duration_s"),
                            "seed": payload.get("seed"),
                            "manifest_row_count": payload.get("row_count"),
                            "runtime": payload.get("runtime"),
                            "claim_boundary": payload.get("claim_boundary", ""),
                        }
                    )
                    row_count = payload.get("row_count")
                except Exception as exc:
                    metadata.update(
                        {
                            "manifest_present": True,
                            "manifest_error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    row_count = None
            else:
                metadata["manifest_present"] = False
                row_count = None
            enriched[entity_id] = entity.model_copy(
                update={"row_count": row_count, "metadata": metadata}
            )
        self._entities = enriched
        return self

    def list_channels(self, entity_id: str) -> list[str]:
        return [
            column
            for column in super().list_channels(entity_id)
            if not column.lower().startswith(("truth.", "label."))
        ]

    def get_metadata(self) -> DataSourceMetadata:
        metadata = super().get_metadata()
        return metadata.model_copy(
            update={
                "source_type": DataSourceType.SIMULATION,
                "root_token": "${ASTROGRAPH_SIMULATION_DATA_ROOT}",
                "metadata": {
                    **metadata.metadata,
                    "layout": "astrograph-simulation-run-directory",
                    "truth_columns_excluded_from_channels": True,
                },
                "claim_boundary": (
                    "E0 engineering simulation source. Labels and truth channels are "
                    "not flight observations and must not be used as model inputs."
                ),
            }
        )
