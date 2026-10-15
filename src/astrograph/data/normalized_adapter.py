from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

STREAM_FILES = {
    "legacy": "telemetry_raw.csv",
    "adcs": "telemetry_adcs_raw.csv",
    "eps": "telemetry_eps_raw.csv",
    "comm": "telemetry_comm_raw.csv",
    "thermal": "telemetry_thermal_raw.csv",
}


@dataclass(frozen=True)
class D1RunDescriptor:
    run_id: str
    case_type: str
    case_id: str
    repeat_index: int
    run_seed: int
    paired_nominal_run_id: str | None
    split: str


class D1DatasetAdapter:
    """Read-only adapter for the reduced PHM D1 delivery package.

    The uploaded package contains aggregate telemetry and compact per-run audit
    records, not the complete per-run data tree. This adapter never assumes the
    omitted files exist.
    """

    def __init__(self, package_root: str | Path, *, split_override: str | Path | None = None):
        self.package_root = Path(package_root).resolve()
        self.dataset_root = self.package_root / "02_dataset"
        self.aggregate = self.dataset_root / "aggregate"
        self.audit_root = self.package_root / "03_run_audit"
        self.manifest = self._read_json(self.dataset_root / "dataset_manifest.json")
        self.plan = self._read_json(self.dataset_root / "campaign_plan.json")
        self.producer_split = self._read_json(self.dataset_root / "dataset_split_manifest.json")
        self.split_override = None if split_override is None else self._read_json(Path(split_override))
        self._run_index = self._build_run_index()

    @staticmethod
    def _read_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8-sig"))

    @property
    def dataset_id(self) -> str:
        return str(self.manifest["dataset_id"])

    def _split_lookup(self) -> dict[str, str]:
        lookup: dict[str, str] = {}
        if self.split_override and "run_ids" in self.split_override:
            for split, run_ids in self.split_override["run_ids"].items():
                for run_id in run_ids:
                    lookup[str(run_id)] = str(split)
            return lookup
        mapping = {
            "train_run_ids": "train",
            "validation_run_ids": "validation",
            "test_run_ids": "test",
            "zero_shot_validation_run_ids": "zero_shot_validation",
            "compound_fault_validation_run_ids": "compound_fault_validation",
        }
        for key, split in mapping.items():
            for run_id in self.producer_split.get(key, []):
                lookup[str(run_id)] = split
        return lookup

    def _build_run_index(self) -> dict[str, D1RunDescriptor]:
        split_lookup = self._split_lookup()
        index: dict[str, D1RunDescriptor] = {}
        for row in self.plan.get("runs", []):
            run_id = str(row["run_id"])
            index[run_id] = D1RunDescriptor(
                run_id=run_id,
                case_type=str(row["case_type"]),
                case_id=str(row["case_id"]),
                repeat_index=int(row["repeat_index"]),
                run_seed=int(row["run_seed"]),
                paired_nominal_run_id=(
                    None if row.get("paired_nominal_run_id") in {None, ""} else str(row["paired_nominal_run_id"])
                ),
                split=split_lookup.get(run_id, "unassigned"),
            )
        return index

    def iter_runs(self) -> Iterator[D1RunDescriptor]:
        yield from sorted(self._run_index.values(), key=lambda x: (x.case_type, x.case_id, x.repeat_index))

    def get_run(self, run_id: str) -> D1RunDescriptor:
        try:
            return self._run_index[run_id]
        except KeyError as exc:
            raise KeyError(f"Unknown D1 run_id: {run_id}") from exc

    def run_manifest(self, run_id: str) -> dict:
        return self._read_json(self.audit_root / run_id / "run_manifest.json")

    def parameter_sample(self, run_id: str) -> dict:
        return self._read_json(self.audit_root / run_id / "parameter_sample.json")

    def fault_event(self, run_id: str) -> dict | None:
        events = self.parameter_sample(run_id).get("events", [])
        return dict(events[0]) if events else None

    def load_stream(self, run_id: str, stream: str, *, chunksize: int = 200_000) -> pd.DataFrame:
        if stream not in STREAM_FILES:
            raise KeyError(f"Unsupported stream {stream!r}")
        path = self.aggregate / STREAM_FILES[stream]
        parts: list[pd.DataFrame] = []
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            match = chunk[chunk["run_id"].astype(str) == run_id]
            if not match.empty:
                parts.append(match)
        if not parts:
            raise ValueError(f"No rows found for run {run_id} in stream {stream}")
        frame = pd.concat(parts, ignore_index=True)
        frame["time_s"] = pd.to_numeric(frame["time_s"], errors="raise")
        return frame.sort_values("time_s").reset_index(drop=True)

    def load_labels(self, run_id: str, stream: str = "legacy", *, chunksize: int = 250_000) -> pd.DataFrame:
        filename = "telemetry_labels.csv" if stream == "legacy" else "telemetry_multirate_labels.csv"
        path = self.aggregate / filename
        parts: list[pd.DataFrame] = []
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            mask = chunk["run_id"].astype(str) == run_id
            if "telemetry_stream" in chunk.columns:
                mask &= chunk["telemetry_stream"].astype(str) == stream
            match = chunk[mask]
            if not match.empty:
                parts.append(match)
        if not parts:
            raise ValueError(f"No labels for run {run_id}, stream {stream}")
        return pd.concat(parts, ignore_index=True)
