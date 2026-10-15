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
class RunDescriptor:
    run_id: str
    case_type: str
    case_id: str
    repeat_index: int
    run_seed: int
    pair_seed: int
    paired_nominal_run_id: str | None
    split: str


class D0DatasetAdapter:
    """Read-only adapter for the D0 dataset-profile contract."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.aggregate = self.root / "aggregate"
        self.runs_root = self.root / "runs"
        self.manifest = self._read_json("dataset_manifest.json")
        self.plan = self._read_json("campaign_plan.json")
        self.split_manifest = self._read_json("dataset_split_manifest.json")
        self._run_index = self._build_run_index()
        self._aggregate_cache: dict[str, pd.DataFrame] = {}

    def _read_json(self, name: str) -> dict:
        path = self.root / name
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8-sig"))

    @property
    def dataset_id(self) -> str:
        return str(self.manifest["dataset_id"])

    def _split_lookup(self) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for split, key in (
            ("train", "train_run_ids"),
            ("validation", "validation_run_ids"),
            ("test", "test_run_ids"),
            ("zero_shot_validation", "compound_fault_validation_run_ids"),
        ):
            for run_id in self.split_manifest.get(key, []):
                lookup[str(run_id)] = split
        return lookup

    def _build_run_index(self) -> dict[str, RunDescriptor]:
        split_lookup = self._split_lookup()
        index: dict[str, RunDescriptor] = {}
        for row in self.plan.get("runs", []):
            run_id = str(row["run_id"])
            index[run_id] = RunDescriptor(
                run_id=run_id,
                case_type=str(row["case_type"]),
                case_id=str(row["case_id"]),
                repeat_index=int(row["repeat_index"]),
                run_seed=int(row["run_seed"]),
                pair_seed=int(row["pair_seed"]),
                paired_nominal_run_id=(
                    None if row.get("paired_nominal_run_id") in {None, ""} else str(row["paired_nominal_run_id"])
                ),
                split=split_lookup.get(run_id, "unassigned"),
            )
        return index

    def iter_runs(self) -> Iterator[RunDescriptor]:
        yield from sorted(self._run_index.values(), key=lambda item: (item.case_type, item.case_id, item.repeat_index))

    def get_run(self, run_id: str) -> RunDescriptor:
        try:
            return self._run_index[run_id]
        except KeyError as exc:
            raise KeyError(f"Unknown run_id: {run_id}") from exc

    def run_manifest(self, run_id: str) -> dict:
        path = self.runs_root / run_id / "run_manifest.json"
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def resolved_spec(self, run_id: str) -> dict:
        path = self.runs_root / run_id / "input" / "resolved_spec.json"
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def fault_event(self, run_id: str) -> dict | None:
        spec = self.resolved_spec(run_id)
        events = (
            spec.get("fully_resolved_task_spec", {})
            .get("events", {})
            .get("faults", [])
        )
        return dict(events[0]) if events else None

    def load_stream(self, run_id: str, stream: str) -> pd.DataFrame:
        if stream not in STREAM_FILES:
            raise KeyError(f"Unsupported stream {stream!r}; expected one of {sorted(STREAM_FILES)}")
        path = self.aggregate / STREAM_FILES[stream]
        if stream not in self._aggregate_cache:
            self._aggregate_cache[stream] = pd.read_csv(path, encoding="utf-8-sig")
        frame = self._aggregate_cache[stream]
        result = frame[frame["run_id"].astype(str) == run_id].copy()
        if result.empty:
            raise ValueError(f"No rows found for run {run_id} in stream {stream}")
        result["time_s"] = pd.to_numeric(result["time_s"], errors="raise")
        return result.sort_values("time_s").reset_index(drop=True)

    def load_labels(self, run_id: str, stream: str = "legacy") -> pd.DataFrame:
        name = "telemetry_labels.csv" if stream == "legacy" else "telemetry_multirate_labels.csv"
        cache_key = f"labels:{stream}"
        if cache_key not in self._aggregate_cache:
            self._aggregate_cache[cache_key] = pd.read_csv(self.aggregate / name, encoding="utf-8-sig")
        frame = self._aggregate_cache[cache_key]
        result = frame[(frame["run_id"].astype(str) == run_id) & (frame["telemetry_stream"].astype(str) == stream)].copy()
        if result.empty:
            raise ValueError(f"No labels found for run {run_id}, stream {stream}")
        return result.reset_index(drop=True)

    def load_joined_stream(self, run_id: str, stream: str) -> pd.DataFrame:
        raw = self.load_stream(run_id, stream)
        labels = self.load_labels(run_id, stream)
        joined = raw.merge(labels, on=["record_id", "run_id"], how="left", validate="one_to_one")
        if joined.filter(regex=r"^label\.").isna().any().any():
            raise ValueError(f"Label join produced missing values for run {run_id}, stream {stream}")
        return joined.sort_values("time_s").reset_index(drop=True)

    def sample_period_s(self, run_id: str, stream: str) -> float:
        frame = self.load_stream(run_id, stream)
        diffs = frame["time_s"].diff().dropna()
        if diffs.empty:
            return 0.0
        return float(diffs.median())
