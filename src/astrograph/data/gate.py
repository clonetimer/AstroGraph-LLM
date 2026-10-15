from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from astrograph.schemas import DataGateCheck, DataGateStatus, DataQualityReport


REQUIRED_FILES = {
    "campaign_status.json",
    "campaign_summary.json",
    "dataset_contract.json",
    "telemetry_dictionary.csv",
    "telemetry_raw.csv",
    "telemetry_labels.csv",
    "telemetry_adcs_raw.csv",
    "telemetry_eps_raw.csv",
    "telemetry_comm_raw.csv",
    "telemetry_thermal_raw.csv",
    "telemetry_multirate_labels.csv",
    "degradation_raw.csv",
    "degradation_labels.csv",
    "degradation_adcs_raw.csv",
    "degradation_eps_raw.csv",
    "degradation_comm_raw.csv",
    "degradation_thermal_raw.csv",
    "degradation_multirate_labels.csv",
}

PAIR_FILES = [
    ("telemetry_raw.csv", "telemetry_labels.csv", None),
    ("degradation_raw.csv", "degradation_labels.csv", None),
    ("telemetry_adcs_raw.csv", "telemetry_multirate_labels.csv", "adcs"),
    ("telemetry_eps_raw.csv", "telemetry_multirate_labels.csv", "eps"),
    ("telemetry_comm_raw.csv", "telemetry_multirate_labels.csv", "comm"),
    ("telemetry_thermal_raw.csv", "telemetry_multirate_labels.csv", "thermal"),
    ("degradation_adcs_raw.csv", "degradation_multirate_labels.csv", "adcs"),
    ("degradation_eps_raw.csv", "degradation_multirate_labels.csv", "eps"),
    ("degradation_comm_raw.csv", "degradation_multirate_labels.csv", "comm"),
    ("degradation_thermal_raw.csv", "degradation_multirate_labels.csv", "thermal"),
]


def _check(check_id: str, ok: bool, message: str, **details) -> DataGateCheck:
    return DataGateCheck(
        check_id=check_id,
        status=DataGateStatus.PASS if ok else DataGateStatus.FAIL,
        message=message,
        details=details,
    )


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _select_labels(frame: pd.DataFrame, stream: str | None) -> pd.DataFrame:
    if stream is None:
        return frame[frame["telemetry_stream"].astype(str).isin({"legacy", "base", "whole", "30s"})] if "telemetry_stream" in frame else frame
    return frame[frame["telemetry_stream"].astype(str) == stream]


def validate_dataset(data_dir: str | Path) -> DataQualityReport:
    root = Path(data_dir)
    checks: list[DataGateCheck] = []
    missing = sorted(name for name in REQUIRED_FILES if not (root / name).exists())
    checks.append(_check("required_files", not missing, "Required dataset files are present.", missing=missing))
    if missing:
        return DataQualityReport(status=DataGateStatus.FAIL, checks=checks, claim_boundary="Input package incomplete.")

    status = json.loads((root / "campaign_status.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "campaign_summary.json").read_text(encoding="utf-8"))
    contract = json.loads((root / "dataset_contract.json").read_text(encoding="utf-8"))
    profile = str(status.get("profile", summary.get("execution_profile", "")))
    checks.append(
        _check(
            "campaign_complete",
            status.get("status") == "complete" and status.get("completed_cases") == status.get("total_cases"),
            "Campaign completion status is internally consistent.",
            status=status,
        )
    )
    checks.append(
        _check(
            "quick_disclosure",
            profile == "quick" and float(summary.get("duration_s", 0)) < 5400.0,
            "Bundled data are explicitly quick engineering-acceptance data, not full-orbit production data.",
            profile=profile,
            duration_s=summary.get("duration_s"),
        )
    )
    checks.append(
        _check(
            "schema_version",
            str(contract.get("schema_version", "")).startswith("2."),
            "Dataset protocol schema is supported.",
            schema_version=contract.get("schema_version"),
        )
    )
    checks.append(
        _check(
            "identifier_policy",
            contract.get("identifier_fields_are_model_features") is False
            and contract.get("identifier_policy", "").lower().find("opaque") >= 0,
            "Identifiers are opaque and excluded from model features.",
            policy=contract.get("identifier_policy"),
        )
    )

    raw_feature_fields: set[str] = set(contract.get("raw_telemetry_fields", []))
    dictionary = _load_csv(root / "telemetry_dictionary.csv")
    dictionary_fields = set(dictionary["field"].astype(str))
    checks.append(
        _check(
            "dictionary_coverage",
            raw_feature_fields.issubset(dictionary_fields),
            "Telemetry dictionary covers every declared raw telemetry feature.",
            missing=sorted(raw_feature_fields - dictionary_fields),
        )
    )

    for raw_name, label_name, stream in PAIR_FILES:
        raw = _load_csv(root / raw_name)
        labels = _select_labels(_load_csv(root / label_name), stream)
        raw_keys = set(zip(raw["record_id"].astype(str), raw["run_id"].astype(str)))
        label_keys = set(zip(labels["record_id"].astype(str), labels["run_id"].astype(str)))
        check_suffix = raw_name.replace(".csv", "")
        checks.append(
            _check(
                f"join_{check_suffix}",
                raw_keys == label_keys and len(raw_keys) == len(raw),
                f"Raw/label join keys match for {raw_name}.",
                raw_rows=len(raw),
                label_rows=len(labels),
                raw_only=len(raw_keys - label_keys),
                labels_only=len(label_keys - raw_keys),
            )
        )
        leaked = [c for c in raw.columns if c.startswith("label.") or c.startswith("truth.") or c in {"campaign_case", "case_id"}]
        checks.append(
            _check(
                f"no_label_leak_{check_suffix}",
                not leaked,
                f"No label/truth/scenario columns appear in {raw_name}.",
                leaked_columns=leaked,
            )
        )
        missing_values = int(raw.isna().sum().sum())
        checks.append(
            _check(
                f"no_missing_{check_suffix}",
                missing_values == 0,
                f"No missing values appear in {raw_name}.",
                missing_values=missing_values,
            )
        )

    label_files = ["telemetry_labels.csv", "degradation_labels.csv", "telemetry_multirate_labels.csv", "degradation_multirate_labels.csv"]
    for name in label_files:
        frame = _load_csv(root / name)
        label_cols = [c for c in frame.columns if c.startswith("label.")]
        nulls = int(frame[label_cols].isna().sum().sum()) if label_cols else 0
        checks.append(_check(f"label_completeness_{name}", nulls == 0, f"Labels are complete in {name}.", null_count=nulls))

    degradation_labels = _load_csv(root / "degradation_labels.csv")
    if "label.degradation_active" in degradation_labels and "age_fraction" in degradation_labels:
        active = degradation_labels["label.degradation_active"].astype(bool)
        age = degradation_labels["age_fraction"].astype(float)
        ok = bool(active[age > 0].all() and (~active[age == 0]).all())
        checks.append(_check("degradation_label_semantics", ok, "Degradation labels agree with age_fraction."))

    passed = all(item.status == DataGateStatus.PASS for item in checks)
    return DataQualityReport(
        status=DataGateStatus.PASS if passed else DataGateStatus.FAIL,
        profile=profile,
        schema_version=str(contract.get("schema_version", "")),
        checks=checks,
        claim_boundary=str(summary.get("claim_boundary", "")),
    )


def write_report(report: DataQualityReport, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
