from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .safe_zip import ZipExtractionBudget, extract_selected_member, normalize_zip_member

import pandas as pd

THERMAL_STAR_STREAM_FILES = {
    "legacy": "telemetry_raw.csv",
    "adcs": "telemetry_adcs_raw.csv",
    "eps": "telemetry_eps_raw.csv",
    "comm": "telemetry_comm_raw.csv",
    "thermal": "telemetry_thermal_raw.csv",
}

_FAULT_ID_BY_CASE = {
    "thermal_heater_stuck_on": "THERMAL_HEATER_STUCK_ON",
    "star_tracker_dropout": "ADCS_STAR_TRACKER_DROPOUT",
}


@dataclass(frozen=True)
class D3ThermalStarRunDescriptor:
    run_id: str
    case_type: str
    case_id: str
    fault_id: str | None
    repeat_index: int
    paired_nominal_run_id: str
    split: str
    severity: str
    condition: str


def _locate_dataset_root(root: Path) -> Path:
    if (root / "dataset_manifest.json").exists() and (root / "aggregate").is_dir():
        return root
    matches = [p.parent for p in root.rglob("dataset_manifest.json") if (p.parent / "aggregate").is_dir()]
    if len(matches) != 1:
        raise ValueError(f"cannot uniquely locate D3 thermal/star dataset root under {root}: {len(matches)} candidates")
    return matches[0]


def _wanted_member(relative: str) -> bool:
    if relative in {
        "dataset_manifest.json",
        "dataset_split_manifest.json",
        "validation_report.json",
        "coverage_matrix.csv",
        "aggregate/telemetry_raw.csv",
        "aggregate/telemetry_adcs_raw.csv",
        "aggregate/telemetry_eps_raw.csv",
        "aggregate/telemetry_comm_raw.csv",
        "aggregate/telemetry_thermal_raw.csv",
        "aggregate/telemetry_labels.csv",
        "aggregate/telemetry_multirate_labels.csv",
    }:
        return True
    if not relative.startswith("runs/"):
        return False
    return relative.endswith((
        "/run_manifest.json",
        "/input/parameter_sample.json",
        "/validation/fault_effect_report.json",
    ))


@contextmanager
def materialize_d3_thermal_star_delivery(path: str | Path) -> Iterator[Path]:
    """Yield a directory-shaped thermal/star delivery from a directory or ZIP.

    The producer ZIP may use Windows backslashes. Only aggregate data and compact
    audit JSON required by AstroGraph are materialized.
    """

    source = Path(path).resolve()
    if source.is_dir():
        yield _locate_dataset_root(source)
        return
    if not source.is_file() or source.suffix.lower() != ".zip":
        raise ValueError(f"D3 thermal/star delivery must be a directory or ZIP: {source}")

    temp_root = Path(tempfile.mkdtemp(prefix="astrograph-d3-thermal-star-"))
    try:
        with zipfile.ZipFile(source) as archive:
            normalized = [(info, normalize_zip_member(info)) for info in archive.infolist()]
            candidates = [name for _, name in normalized if name.endswith("/dataset_manifest.json")]
            candidates = [
                name for name in candidates
                if "phm_d3_thermal_star_production" in name
            ] or candidates
            if len(candidates) != 1:
                raise ValueError(f"thermal/star ZIP must contain one dataset manifest, got {len(candidates)}")
            manifest_name = candidates[0]
            prefix = manifest_name[: -len("dataset_manifest.json")].rstrip("/")
            output_root = temp_root / Path(prefix).name
            budget = ZipExtractionBudget()
            for info, name in normalized:
                if info.is_dir() or not name.startswith(prefix + "/"):
                    continue
                relative = name[len(prefix) + 1 :]
                if not _wanted_member(relative):
                    continue
                extract_selected_member(
                    archive,
                    info,
                    output_root=output_root,
                    relative=relative,
                    budget=budget,
                )
        yield _locate_dataset_root(output_root)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


class D3ThermalStarDatasetAdapter:
    """Read-only adapter for the D3 heater/star-tracker production delivery."""

    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = _locate_dataset_root(Path(dataset_root).resolve())
        self.aggregate = self.dataset_root / "aggregate"
        self.runs_root = self.dataset_root / "runs"
        self.manifest = self._read_json(self.dataset_root / "dataset_manifest.json")
        self.producer_split = self._read_json(self.dataset_root / "dataset_split_manifest.json")
        self._run_index = self._build_run_index()

    @staticmethod
    def _read_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8-sig"))

    @property
    def dataset_id(self) -> str:
        return str(self.manifest["dataset_id"])

    def _split_lookup(self) -> dict[str, str]:
        lookup: dict[str, str] = {}
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

    def _build_run_index(self) -> dict[str, D3ThermalStarRunDescriptor]:
        split_lookup = self._split_lookup()
        raw: list[dict] = []
        for manifest_path in sorted(self.runs_root.glob("*/run_manifest.json")):
            row = self._read_json(manifest_path)
            if row.get("status") != "COMPLETE":
                continue
            raw.append(row)
        nominal_by_repeat = {
            int(row["repeat_index"]): str(row["run_id"])
            for row in raw
            if row.get("case_id") == "nominal_heater_off"
        }
        result: dict[str, D3ThermalStarRunDescriptor] = {}
        for row in raw:
            repeat = int(row["repeat_index"])
            run_id = str(row["run_id"])
            pair_id = nominal_by_repeat.get(repeat)
            if pair_id is None:
                raise ValueError(f"missing nominal_heater_off pair for repeat {repeat}")
            case_id = str(row["case_id"])
            result[run_id] = D3ThermalStarRunDescriptor(
                run_id=run_id,
                case_type=str(row["case_type"]),
                case_id=case_id,
                fault_id=_FAULT_ID_BY_CASE.get(case_id),
                repeat_index=repeat,
                paired_nominal_run_id=pair_id,
                split=split_lookup.get(run_id, "unassigned"),
                severity=str(row.get("severity", "unknown")),
                condition=str(row.get("condition", "unknown")),
            )
        return result

    def iter_runs(self) -> Iterator[D3ThermalStarRunDescriptor]:
        yield from sorted(self._run_index.values(), key=lambda x: (x.repeat_index, x.case_id))

    def get_run(self, run_id: str) -> D3ThermalStarRunDescriptor:
        try:
            return self._run_index[str(run_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown D3 thermal/star run_id: {run_id}") from exc

    def parameter_sample(self, run_id: str) -> dict:
        return self._read_json(self.runs_root / run_id / "input" / "parameter_sample.json")

    def fault_effect_report(self, run_id: str) -> dict | None:
        path = self.runs_root / run_id / "validation" / "fault_effect_report.json"
        return self._read_json(path) if path.exists() else None

    def load_stream(self, run_id: str, stream: str, *, chunksize: int = 200_000) -> pd.DataFrame:
        if stream not in THERMAL_STAR_STREAM_FILES:
            raise KeyError(f"Unsupported D3 thermal/star stream {stream!r}")
        path = self.aggregate / THERMAL_STAR_STREAM_FILES[stream]
        parts: list[pd.DataFrame] = []
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            match = chunk[chunk["run_id"].astype(str) == str(run_id)]
            if not match.empty:
                parts.append(match)
        if not parts:
            raise ValueError(f"No thermal/star rows found for run {run_id}, stream {stream}")
        frame = pd.concat(parts, ignore_index=True)
        frame["time_s"] = pd.to_numeric(frame["time_s"], errors="raise")
        return frame.sort_values("time_s").reset_index(drop=True)
