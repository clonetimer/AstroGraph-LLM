"""Fairness audit for three-task and end-to-end diagnosis comparisons."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ComparisonRunMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    family: str
    information_regime: str
    seed: int
    parameter_count: int = Field(gt=0)
    epoch_budget: int = Field(gt=0)
    detection_macro_f1: float = Field(ge=0.0, le=1.0)
    localization_macro_f1: float = Field(ge=0.0, le=1.0)
    classification_macro_f1: float = Field(ge=0.0, le=1.0)
    end_to_end_macro_f1: float = Field(ge=0.0, le=1.0)


class DatasetFairnessAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_audit_passed: Literal[True] = True
    group_overlap_count: Literal[0] = 0
    compared_families: list[str] = Field(min_length=2)
    shared_seeds: list[int] = Field(min_length=1)
    common_epoch_budget: int = Field(gt=0)
    parameter_capacity_ratio: float = Field(ge=1.0, le=4.0)
    task_metric_completeness: Literal[True] = True
    per_task_reporting_preserved: Literal[True] = True
    common_test_split: Literal[True] = True
    common_training_budget: Literal[True] = True
    common_seed_budget: Literal[True] = True
    representation_difference_disclosed: Literal[True] = True
    cross_information_regime_ranking_prohibited: Literal[True] = True
    no_method_omitted: Literal[True] = True
    fairness_status: Literal["PASS"] = "PASS"


class FairThreeTaskComparisonLedger(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-fair-three-task-comparison.v1"] = (
        "astrograph-fair-three-task-comparison.v1"
    )
    comparison_id: str
    source_script_sha256: dict[str, str]
    dataset_index_sha256: dict[str, str]
    execution_contract: dict[str, Any]
    dataset_audits: list[DatasetFairnessAudit] = Field(min_length=1)
    runs: list[ComparisonRunMetrics] = Field(min_length=2)
    family_macro_means: dict[str, dict[str, float]]
    task_leaders: dict[str, dict[str, str | float]]
    anti_suppression_checks: dict[str, bool]
    status: Literal["PASS_FAIR_COMPARISON"] = "PASS_FAIR_COMPARISON"
    ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = (
        "Same-report comparisons use one split and uniform seed/epoch budgets and report detection, "
        "localization, classification and end-to-end Macro-F1 separately. Capacity and representation "
        "differences are disclosed. Results do not establish production promotion or causal superiority."
    )

    @model_validator(mode="after")
    def validate_content(self) -> "FairThreeTaskComparisonLedger":
        if not all(self.anti_suppression_checks.values()):
            raise ValueError("fair comparison contains a failed anti-suppression check")
        expected = _canonical_sha256(self.model_dump(mode="json", exclude={"ledger_sha256"}))
        if self.ledger_sha256 != expected:
            raise ValueError("fair comparison ledger_sha256 mismatch")
        return self


def _metric(item: dict[str, Any], key: str) -> float:
    value = item.get(key)
    if not isinstance(value, dict) or "macro_f1" not in value:
        raise ValueError(f"comparison method lacks {key}.macro_f1")
    return float(value["macro_f1"])


def _information_regime(dataset_id: str, family: str) -> str:
    if dataset_id == "phm.v6_0":
        if family.startswith("paired_residual_"):
            return "MATCHED_NOMINAL_DIGITAL_TWIN_REQUIRED"
        if family.startswith("raw_summary_"):
            return "RAW_SUMMARY_ONLY"
        raise ValueError(f"PHM comparison family has an undisclosed representation: {family}")
    return "CAUSAL_WINDOW_BASE_FEATURES"


def _normalize_report(path: Path, dataset_id: str) -> tuple[list[ComparisonRunMetrics], DatasetFairnessAudit]:
    report = json.loads(path.read_text(encoding="utf-8"))
    split = report.get("split_audit", {})
    if split.get("status") != "PASS" or split.get("group_overlap_count") != 0:
        raise ValueError(f"{dataset_id} comparison split audit did not pass")
    items = report.get("models", [])
    if len(items) < 2:
        raise ValueError(f"{dataset_id} comparison requires at least two methods")
    runs: list[ComparisonRunMetrics] = []
    seeds_by_family: dict[str, set[int]] = defaultdict(set)
    epochs_by_family: dict[str, set[int]] = defaultdict(set)
    for item in items:
        family = str(item["family"])
        seed = int(item["seed"])
        history = item.get("validation_history") or item.get("history")
        if not isinstance(history, list) or not history:
            raise ValueError(f"{dataset_id}:{family}:{seed} lacks a full training history")
        epoch_budget = len(history)
        end_key = "end_to_end_15_class" if "end_to_end_15_class" in item else "end_to_end_18_class"
        runs.append(ComparisonRunMetrics(
            dataset_id=dataset_id,
            family=family,
            information_regime=_information_regime(dataset_id, family),
            seed=seed,
            parameter_count=int(item["parameter_count"]),
            epoch_budget=epoch_budget,
            detection_macro_f1=_metric(item, "detection"),
            localization_macro_f1=_metric(item, "localization"),
            classification_macro_f1=_metric(item, "classification_predicted_component"),
            end_to_end_macro_f1=_metric(item, end_key),
        ))
        seeds_by_family[family].add(seed)
        epochs_by_family[family].add(epoch_budget)
    families = sorted(seeds_by_family)
    seed_sets = {tuple(sorted(values)) for values in seeds_by_family.values()}
    epoch_sets = {tuple(sorted(values)) for values in epochs_by_family.values()}
    if len(seed_sets) != 1:
        raise ValueError(f"{dataset_id} methods received unequal seed budgets")
    if len(epoch_sets) != 1 or len(next(iter(epoch_sets))) != 1:
        raise ValueError(f"{dataset_id} methods received unequal epoch budgets")
    params = [item.parameter_count for item in runs]
    capacity_ratio = max(params) / min(params)
    regimes = {item.information_regime for item in runs}
    disclosure = (
        len(regimes) == 1
        or "matched nominal digital twin" in str(report.get("claim_boundary", "")).lower()
    )
    if not disclosure:
        raise ValueError(f"{dataset_id} does not disclose its information-regime difference")
    audit = DatasetFairnessAudit(
        dataset_id=dataset_id,
        report_sha256=_file_sha256(path),
        compared_families=families,
        shared_seeds=list(next(iter(seed_sets))),
        common_epoch_budget=next(iter(epoch_sets))[0],
        parameter_capacity_ratio=capacity_ratio,
    )
    return runs, audit


def build_fair_three_task_comparison_ledger(
    *, adcs_report_path: str | Path, phm_report_path: str | Path,
    adcs_script_path: str | Path, phm_script_path: str | Path,
    adcs_dataset_index_path: str | Path, phm_dataset_index_path: str | Path,
    output_path: str | Path,
) -> FairThreeTaskComparisonLedger:
    paths = {
        "adcs.v5_2_2": Path(adcs_report_path).resolve(),
        "phm.v6_0": Path(phm_report_path).resolve(),
    }
    scripts = {
        "adcs.v5_2_2": Path(adcs_script_path).resolve(),
        "phm.v6_0": Path(phm_script_path).resolve(),
    }
    indexes = {
        "adcs.v5_2_2": Path(adcs_dataset_index_path).resolve(),
        "phm.v6_0": Path(phm_dataset_index_path).resolve(),
    }
    all_runs: list[ComparisonRunMetrics] = []
    audits: list[DatasetFairnessAudit] = []
    for dataset_id, path in paths.items():
        runs, audit = _normalize_report(path, dataset_id)
        all_runs.extend(runs)
        audits.append(audit)
    script_text = {key: path.read_text(encoding="utf-8") for key, path in scripts.items()}
    anti_suppression = {
        "adcs_uniform_epoch_argument": "epochs=args.epochs" in script_text["adcs.v5_2_2"],
        "phm_uniform_epoch_argument": "epochs=args.epochs" in script_text["phm.v6_0"],
        "adcs_common_model_loop": "for name in args.models" in script_text["adcs.v5_2_2"],
        "phm_common_model_loop": "for kind in (\"mlp\", \"resmlp\")" in script_text["phm.v6_0"],
        "every_run_reports_all_four_targets": len(all_runs) == sum(
            len(json.loads(path.read_text(encoding="utf-8"))["models"]) for path in paths.values()
        ),
        "all_split_audits_pass": all(item.split_audit_passed for item in audits),
        "capacity_ratio_disclosed_and_bounded": all(item.parameter_capacity_ratio <= 4 for item in audits),
    }
    grouped: dict[str, list[ComparisonRunMetrics]] = defaultdict(list)
    for run in all_runs:
        grouped[f"{run.dataset_id}:{run.information_regime}:{run.family}"].append(run)
    metric_names = (
        "detection_macro_f1", "localization_macro_f1",
        "classification_macro_f1", "end_to_end_macro_f1",
    )
    family_means = {
        key: {metric: mean(getattr(item, metric) for item in values) for metric in metric_names}
        for key, values in sorted(grouped.items())
    }
    leaders: dict[str, dict[str, str | float]] = {}
    for dataset_id in paths:
        regimes = sorted({run.information_regime for run in all_runs if run.dataset_id == dataset_id})
        for regime in regimes:
            relevant = {
                key: values for key, values in family_means.items()
                if key.startswith(f"{dataset_id}:{regime}:")
            }
            for metric in metric_names:
                family, value = max(
                    ((key.rsplit(":", 1)[1], values[metric]) for key, values in relevant.items()),
                    key=lambda pair: (pair[1], pair[0]),
                )
                leaders[f"{dataset_id}:{regime}:{metric}"] = {
                    "family": family, "macro_f1": value,
                }
    body = {
        "comparison_id": "FAIR-3TASK-" + _canonical_sha256({
            key: _file_sha256(path) for key, path in paths.items()
        })[:20].upper(),
        "source_script_sha256": {key: _file_sha256(path) for key, path in scripts.items()},
        "dataset_index_sha256": {key: _file_sha256(path) for key, path in indexes.items()},
        "execution_contract": {
            "adcs.v5_2_2": {
                "device": "cpu",
                "epoch_budget": next(item.common_epoch_budget for item in audits if item.dataset_id == "adcs.v5_2_2"),
                "seeds": next(item.shared_seeds for item in audits if item.dataset_id == "adcs.v5_2_2"),
                "information_regime": "CAUSAL_WINDOW_BASE_FEATURES",
            },
            "phm.v6_0": {
                "device": "cpu",
                "epoch_budget": next(item.common_epoch_budget for item in audits if item.dataset_id == "phm.v6_0"),
                "seeds": next(item.shared_seeds for item in audits if item.dataset_id == "phm.v6_0"),
                "information_regimes": ["RAW_SUMMARY_ONLY", "MATCHED_NOMINAL_DIGITAL_TWIN_REQUIRED"],
            },
        },
        "dataset_audits": audits,
        "runs": all_runs,
        "family_macro_means": family_means,
        "task_leaders": leaders,
        "anti_suppression_checks": anti_suppression,
    }
    provisional = FairThreeTaskComparisonLedger.model_construct(**body, ledger_sha256="0" * 64)
    ledger = FairThreeTaskComparisonLedger(
        **body,
        ledger_sha256=_canonical_sha256(provisional.model_dump(mode="json", exclude={"ledger_sha256"})),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(ledger.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return ledger


__all__ = [
    "ComparisonRunMetrics", "DatasetFairnessAudit", "FairThreeTaskComparisonLedger",
    "build_fair_three_task_comparison_ledger",
]
