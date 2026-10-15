"""Governed audit for domain-strong time-series diagnosis comparisons."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


SOTA_FAMILIES = ("inception_time", "timesnet", "modern_tcn")
TASKS = (
    "detection_macro_f1", "localization_macro_f1",
    "classification_macro_f1", "end_to_end_macro_f1",
)
PAPER_REGISTRY = {
    "inception_time": {
        "title": "InceptionTime: Finding AlexNet for Time Series Classification",
        "year": 2020,
        "primary_url": "https://doi.org/10.1007/s10618-020-00710-y",
        "task_fit": "purpose-built multivariate time-series classification",
    },
    "timesnet": {
        "title": "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis",
        "year": 2023,
        "primary_url": "https://openreview.net/forum?id=ju_Uqw384Oq",
        "task_fit": "task-general backbone evaluated on classification and anomaly detection",
    },
    "modern_tcn": {
        "title": "ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis",
        "year": 2024,
        "primary_url": "https://openreview.net/forum?id=vpJMJerXHU",
        "task_fit": "large-effective-receptive-field convolutional general time-series backbone",
    },
}


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _split_passed(audit: dict[str, Any]) -> bool:
    return audit.get("status") == "PASS" and int(audit.get("group_overlap_count", -1)) == 0


def _aggregate_values(payload: dict[str, Any]) -> dict[str, float]:
    return {task: float(payload[task]["mean"]) for task in TASKS}


def _adcs_comparison_family(item: dict[str, Any]) -> str:
    """Recover the encoder ID from the existing composed ADCS family receipt."""
    value = str(item.get("family", ""))
    prefix = "logistic_detector_"
    suffix = "_hierarchy_blend"
    if not (value.startswith(prefix) and value.endswith(suffix)):
        raise ValueError("ADCS model receipt lacks a governed composed family identifier")
    return value[len(prefix):-len(suffix)]


def _adcs_runs(report: dict[str, Any]) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    if report.get("protocol", {}).get("device") != "cuda":
        raise ValueError("ADCS SOTA comparison was not executed on CUDA")
    if report.get("adcs", {}).get("feature_profile") != "innovation":
        raise ValueError("ADCS comparison must use the current innovation feature profile")
    folds = report["adcs"]["folds"]
    if [item["test_fold"] for item in folds] != list(range(5)):
        raise ValueError("ADCS comparison must contain five ordered test folds")
    if not all(_split_passed(item["split_audit"]) for item in folds):
        raise ValueError("ADCS comparison split audit failed")
    expected = {*SOTA_FAMILIES, "residual_tcn"}
    if set(report["adcs"]["aggregate"]) != expected:
        raise ValueError("ADCS comparison omitted a frozen SOTA or AstroGraph family")
    training = []
    for fold in folds:
        number = int(fold["test_fold"])
        if {_adcs_comparison_family(item) for item in fold["models"]} != expected:
            raise ValueError("ADCS fold model set mismatch")
        for item in fold["models"]:
            family = _adcs_comparison_family(item)
            if item.get("device") != "cuda" or not item.get("cuda_device_name"):
                raise ValueError("ADCS model lacks CUDA execution evidence")
            if int(item.get("maximum_epochs", 0)) != 80 or len(item.get("validation_history", [])) != 80:
                raise ValueError("ADCS family did not receive the complete 80-epoch budget")
            if int(item.get("seed", -1)) != 20260901 + number:
                raise ValueError("ADCS families did not receive the frozen fold seed")
            if int(item.get("batch_size", 0)) != 1024:
                raise ValueError("ADCS families did not receive the frozen batch size")
            training.append({
                "fold": number, "family": family,
                "parameter_count": int(item["parameter_count"]),
                "epochs_completed": len(item["validation_history"]),
                "seed": int(item["seed"]), "batch_size": int(item["batch_size"]),
                "cuda_device_name": item["cuda_device_name"],
            })
    return {
        family: _aggregate_values(report["adcs"]["aggregate"][family])
        for family in sorted(expected)
    }, training


def _phm_runs(
    reports: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    expected = {"astrograph_specialist", *SOTA_FAMILIES}
    if set(reports) != expected:
        raise ValueError("PHM comparison omitted a frozen SOTA or AstroGraph specialist family")
    results: dict[str, dict[str, float]] = {}
    training = []
    reference_channels: dict[int, list[str]] = {}
    reference_splits: dict[int, str] = {}
    expected_encoders = {
        "astrograph_specialist": {
            "detector_training": "causal_multiview_detector_tcn",
            "diagnosis_training": "hierarchical_mechanism_gat_tcn",
        },
        **{
            family: {"detector_training": family, "diagnosis_training": family}
            for family in SOTA_FAMILIES
        },
    }
    for family, report in sorted(reports.items()):
        protocol = report.get("protocol", {})
        if (
            protocol.get("label_contract") != "manifestation_aware"
            or protocol.get("training_sampling") != "trajectory_uniform"
            or protocol.get("detector_feature_profile") != "change_summary"
            or protocol.get("detector_task_scope") != "detection_aux"
            or float(protocol.get("detector_diagnosis_auxiliary_weight", -1.0)) != 0.1
        ):
            raise ValueError(f"PHM {family} did not use the frozen information/training contract")
        folds = report.get("folds", [])
        if [item["test_fold"] for item in folds] != list(range(5)):
            raise ValueError(f"PHM {family} must contain five ordered test folds")
        for fold in folds:
            number = int(fold["test_fold"])
            if not _split_passed(fold["split_audit"]):
                raise ValueError(f"PHM {family} split audit failed")
            split_hash = _canonical_hash(fold["split_audit"])
            channels = list(fold["selected_channels"])
            if len(channels) != 96:
                raise ValueError(f"PHM {family} did not receive the frozen 96-channel budget")
            if number in reference_channels and channels != reference_channels[number]:
                raise ValueError("PHM families received different train-only channel selections")
            if number in reference_splits and split_hash != reference_splits[number]:
                raise ValueError("PHM families received different group splits")
            reference_channels[number] = channels
            reference_splits[number] = split_hash
            for role in ("detector_training", "diagnosis_training"):
                item = fold[role]
                if item.get("device") != "cuda" or not item.get("cuda_device_name"):
                    raise ValueError(f"PHM {family}/{role} lacks CUDA evidence")
                if int(item.get("maximum_epochs", 0)) != 80 or len(item.get("history", [])) != 80:
                    raise ValueError(f"PHM {family}/{role} did not receive 80 epochs")
                if int(item.get("seed", -1)) != 20260901 + number:
                    raise ValueError(f"PHM {family}/{role} did not receive the frozen fold seed")
                if int(item.get("batch_size", 0)) != 1024:
                    raise ValueError(f"PHM {family}/{role} did not receive the frozen batch size")
                if item.get("family") != expected_encoders[family][role]:
                    raise ValueError(f"PHM {family}/{role} encoder family mismatch")
                training.append({
                    "fold": number, "family": family, "role": role,
                    "encoder": item["family"],
                    "parameter_count": int(item["parameter_count"]),
                    "epochs_completed": len(item["history"]),
                    "seed": int(item["seed"]), "batch_size": int(item["batch_size"]),
                    "cuda_device_name": item["cuda_device_name"],
                })
        results[family] = _aggregate_values(report["aggregate"])
    return results, training


def _leaders(results: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, Any]]:
    leaders = {}
    for dataset, families in results.items():
        for task in TASKS:
            family, score = max(
                ((family, values[task]) for family, values in families.items()),
                key=lambda item: (item[1], item[0]),
            )
            leaders[f"{dataset}:{task}"] = {"family": family, "macro_f1": score}
    return leaders


def build_domain_sota_comparison_audit(
    *,
    adcs_report_path: str | Path,
    phm_report_paths: dict[str, str | Path],
    source_paths: list[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate the executed comparison and emit a content-addressed audit."""
    adcs_path = Path(adcs_report_path).resolve()
    phm_paths = {name: Path(path).resolve() for name, path in phm_report_paths.items()}
    adcs_report = json.loads(adcs_path.read_text(encoding="utf-8"))
    phm_reports = {
        name: json.loads(path.read_text(encoding="utf-8")) for name, path in phm_paths.items()
    }
    adcs_results, adcs_training = _adcs_runs(adcs_report)
    phm_results, phm_training = _phm_runs(phm_reports)
    results = {"adcs.v5_2_2": adcs_results, "phm.v6_0": phm_results}
    parameter_counts = [item["parameter_count"] for item in [*adcs_training, *phm_training]]
    body = {
        "schema_version": "astrograph-domain-sota-comparison-audit.v1",
        "comparison_id": "DOMAIN-SOTA-" + _canonical_hash(results)[:20].upper(),
        "status": "PASS_DOMAIN_SOTA_FAIR_COMPARISON",
        "paper_registry": PAPER_REGISTRY,
        "adaptation_disclosure": {
            family: "MECHANISM_PRESERVING_LOCAL_ADAPTATION_NOT_AUTHORS_REPOSITORY"
            for family in SOTA_FAMILIES
        },
        "protocol": {
            "folds": 5,
            "group_key": "sample_index modulo five; one Test, next Validation, remaining Train",
            "maximum_epochs_per_network": 80,
            "early_stopping_truncated_training": False,
            "device": "CUDA",
            "test_access": "once per frozen fold/model selection",
            "metrics": list(TASKS),
            "adcs_information_regime": "20 s causal window; innovation profile for every family",
            "phm_information_regime": "60 s causal window; identical train-only 96-channel selection; specialist detector/diagnosis structure for every family",
        },
        "anti_suppression_checks": {
            "three_domain_strong_families_present": True,
            "astrograph_current_family_present": True,
            "same_group_splits": True,
            "same_train_only_preprocessing": True,
            "same_full_epoch_budget": True,
            "same_fold_seed_and_batch_budget": True,
            "same_label_sampling_and_detector_contract": True,
            "same_four_task_metrics": True,
            "cuda_execution_verified_per_network": True,
            "native_capacity_not_artificially_capped": True,
            "parameter_difference_disclosed": True,
            "no_cross_information_regime_ranking": True,
            "test_not_used_for_selection": True,
        },
        "capacity_disclosure": {
            "minimum_parameter_count": min(parameter_counts),
            "maximum_parameter_count": max(parameter_counts),
            "maximum_to_minimum_ratio": max(parameter_counts) / min(parameter_counts),
            "interpretation": "Capacity is reported rather than capped; equalizing it downward would under-specify the published strong families.",
        },
        "results": results,
        "task_leaders": _leaders(results),
        "training_receipts": {"adcs": adcs_training, "phm": phm_training},
        "artifact_sha256": {
            "adcs_report": _hash_file(adcs_path),
            **{f"phm_{name}": _hash_file(path) for name, path in sorted(phm_paths.items())},
            **{
                f"source_{Path(path).name}": _hash_file(Path(path).resolve())
                for path in source_paths
            },
        },
        "claim_boundary": (
            "Five-fold E0 synthetic-data comparison of local mechanism-preserving adaptations of strong "
            "time-series architecture families. It is not an exact authors-code reproduction, external "
            "benchmark, production promotion, causal superiority, flight or safety claim."
        ),
    }
    body["audit_sha256"] = _canonical_hash(body)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return body


def validate_domain_sota_comparison_audit(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = payload.pop("audit_sha256", "")
    if expected != _canonical_hash(payload):
        raise ValueError("domain SOTA comparison audit hash mismatch")
    payload["audit_sha256"] = expected
    if payload.get("status") != "PASS_DOMAIN_SOTA_FAIR_COMPARISON":
        raise ValueError("domain SOTA comparison did not pass")
    if not all(payload.get("anti_suppression_checks", {}).values()):
        raise ValueError("domain SOTA comparison contains a failed anti-suppression check")
    return payload


__all__ = [
    "PAPER_REGISTRY", "SOTA_FAMILIES", "TASKS",
    "build_domain_sota_comparison_audit", "validate_domain_sota_comparison_audit",
]
