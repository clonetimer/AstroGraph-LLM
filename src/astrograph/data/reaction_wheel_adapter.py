from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

D3_STREAM_FILES = {
    "legacy": "telemetry_raw.csv",
    "adcs": "telemetry_adcs_raw.csv",
    "eps": "telemetry_eps_raw.csv",
    "comm": "telemetry_comm_raw.csv",
    "thermal": "telemetry_thermal_raw.csv",
}


@dataclass(frozen=True)
class D3RunDescriptor:
    run_id: str
    case_type: str
    case_id: str
    fault_id: str | None
    repeat_index: int
    run_seed: int
    pair_seed: int
    paired_nominal_run_id: str | None
    split: str


class D3DatasetAdapter:
    """Read-only adapter for the D3 reaction-wheel aggregate + compact-audit delivery."""

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
        }
        for key, split in mapping.items():
            for run_id in self.producer_split.get(key, []):
                lookup[str(run_id)] = split
        return lookup

    def _build_run_index(self) -> dict[str, D3RunDescriptor]:
        split_lookup = self._split_lookup()
        result: dict[str, D3RunDescriptor] = {}
        for row in self.plan.get("runs", []):
            run_id = str(row["run_id"])
            pair = row.get("paired_nominal_run_id")
            result[run_id] = D3RunDescriptor(
                run_id=run_id,
                case_type=str(row["case_type"]),
                case_id=str(row["case_id"]),
                fault_id=None if row.get("fault_id") in {None, ""} else str(row["fault_id"]),
                repeat_index=int(row["repeat_index"]),
                run_seed=int(row["run_seed"]),
                pair_seed=int(row["pair_seed"]),
                paired_nominal_run_id=None if pair in {None, ""} else str(pair),
                split=split_lookup.get(run_id, "unassigned"),
            )
        return result

    def iter_runs(self) -> Iterator[D3RunDescriptor]:
        yield from sorted(self._run_index.values(), key=lambda x: (x.repeat_index, x.case_type, x.case_id))

    def get_run(self, run_id: str) -> D3RunDescriptor:
        try:
            return self._run_index[run_id]
        except KeyError as exc:
            raise KeyError(f"Unknown D3 run_id: {run_id}") from exc

    def run_manifest(self, run_id: str) -> dict:
        return self._read_json(self.audit_root / run_id / "run_manifest.json")

    def parameter_sample(self, run_id: str) -> dict:
        return self._read_json(self.audit_root / run_id / "parameter_sample.json")

    def fault_effect_report(self, run_id: str) -> dict | None:
        path = self.audit_root / run_id / "fault_effect_report.json"
        return self._read_json(path) if path.exists() else None

    def load_stream(self, run_id: str, stream: str, *, chunksize: int = 200_000) -> pd.DataFrame:
        if stream not in D3_STREAM_FILES:
            raise KeyError(f"Unsupported D3 stream {stream!r}")
        path = self.aggregate / D3_STREAM_FILES[stream]
        parts: list[pd.DataFrame] = []
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            match = chunk[chunk["run_id"].astype(str) == run_id]
            if not match.empty:
                parts.append(match)
        if not parts:
            raise ValueError(f"No D3 rows found for run {run_id}, stream {stream}")
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
            raise ValueError(f"No D3 labels for run {run_id}, stream {stream}")
        return pd.concat(parts, ignore_index=True)
