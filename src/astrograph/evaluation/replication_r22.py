"""R22 preregistered new-holdout admission and frozen replication executors.

R22 is deliberately incapable of tuning on R20/R21 seen evaluation data.  It binds
new source bytes before truth inspection, verifies identity novelty, and runs only the
frozen OPS remediation selected in R21 or the unchanged ESA transport baseline.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .binary_anomaly_metrics import binary_metrics
from .esa_event_metrics import evaluate_events
from .esa_statistical_predictor import _score_core as _esa_score_core
from .external_dataset_contract import digest, file_identity, seal, verify_seal
from .ops_sat_adapter import _raw_segment_score_core, validate_online_fields

CONFIG = Path(__file__).resolve().parents[3] / "configs/evaluation/r22_replication_preregistration.v1.json"
R21_REGISTRY = Path(__file__).resolve().parents[3] / "configs/evaluation/r21_data_exposure_registry.v1.json"
SOURCE_ROOT = Path(__file__).resolve().parents[3]

OPS_FAMILY = "OPS-SAT"
ESA_FAMILY = "ESA"
ALLOWED_FAMILIES = {OPS_FAMILY, ESA_FAMILY}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_source_binding(row: Mapping[str, Any]) -> None:
    relative = str(row.get("relative_path", ""))
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise ValueError("R22_IMPLEMENTATION_SOURCE_PATH_INVALID")
    path = SOURCE_ROOT / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError("R22_IMPLEMENTATION_SOURCE_MISSING:" + relative)
    identity = file_identity(path)
    if identity != {"size": int(row.get("size", -1)), "sha256": row.get("sha256")}:
        raise ValueError("R22_IMPLEMENTATION_SOURCE_DRIFT:" + relative)


def _esa_implementation_payload(spec: Mapping[str, Any]) -> dict:
    return {
        "comparison_semantics": spec["comparison_semantics"],
        "channels": spec["channels"],
        "required_preprocessing_transform": spec["required_preprocessing_transform"],
        "aggregation": spec["aggregation"],
        "threshold": spec["threshold"],
        "confirmation_duration_s": spec["confirmation_duration_s"],
        "reference": spec["reference"],
        "implementation_sources": spec["implementation_sources"],
    }


def _verify_implementation_bindings(plan: Mapping[str, Any]) -> None:
    ops = plan["families"][OPS_FAMILY]
    baseline = dict(ops["baseline"])
    baseline_identity = baseline.pop("implementation_identity_sha256")
    _verify_source_binding(baseline["implementation_source"])
    if digest(baseline) != baseline_identity:
        raise ValueError("R22_OPS_BASELINE_IMPLEMENTATION_IDENTITY_DRIFT")
    remediation = dict(ops["remediation"])
    remediation_identity = remediation.pop("implementation_identity_sha256")
    for row in remediation["implementation_sources"]:
        _verify_source_binding(row)
    if digest(remediation) != remediation_identity:
        raise ValueError("R22_OPS_REMEDIATION_IMPLEMENTATION_IDENTITY_DRIFT")

    esa = plan["families"][ESA_FAMILY]
    for row in esa["implementation_sources"]:
        _verify_source_binding(row)
    if digest(_esa_implementation_payload(esa)) != esa["implementation_identity_sha256"]:
        raise ValueError("R22_ESA_IMPLEMENTATION_IDENTITY_DRIFT")


def load_replication_plan(path: str | Path = CONFIG) -> dict:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_seal(plan)
    if plan.get("schema_version") != "astrograph-r22-replication-preregistration.v1":
        raise ValueError("R22_REPLICATION_PLAN_SCHEMA_INVALID")
    if not plan.get("frozen_before_new_holdout"):
        raise ValueError("R22_REPLICATION_PLAN_NOT_FROZEN")
    source_id = plan.get("implementation_source") or {}
    if source_id.get("relative_path") != "src/astrograph/evaluation/replication_r22.py":
        raise ValueError("R22_EXECUTOR_SOURCE_BINDING_INVALID")
    if source_id.get("sha256") != _sha256_file(Path(__file__)):
        raise ValueError("R22_EXECUTOR_SOURCE_SHA256_DRIFT")
    _verify_implementation_bindings(plan)
    return plan


def _artifact_identity(path: Path) -> dict:
    path = path.resolve()
    if not path.exists():
        raise ValueError("R22_HOLDOUT_SOURCE_NOT_FOUND")
    if path.is_symlink():
        raise ValueError("R22_HOLDOUT_SYMLINK_FORBIDDEN")
    if path.is_file():
        row = {"relative_path": path.name, **file_identity(path)}
        return {
            "kind": "FILE_BYTES",
            "files": [row],
            "source_artifact_size_bytes": row["size"],
            "source_artifact_sha256": row["sha256"],
        }
    if not path.is_dir():
        raise ValueError("R22_HOLDOUT_SOURCE_TYPE_UNSUPPORTED")
    rows: list[dict[str, Any]] = []
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError("R22_HOLDOUT_SYMLINK_FORBIDDEN")
        if item.is_file():
            rows.append({"relative_path": item.relative_to(path).as_posix(), **file_identity(item)})
    if not rows:
        raise ValueError("R22_HOLDOUT_SOURCE_EMPTY")
    return {
        "kind": "DIRECTORY_CONTENT_MANIFEST",
        "files": rows,
        "source_artifact_size_bytes": sum(row["size"] for row in rows),
        "source_artifact_sha256": digest(rows),
    }


def build_new_holdout_intake_receipt(
    source_path: str | Path,
    *,
    dataset_id: str,
    dataset_family: str,
    operator_attests_pre_truth_inspection: bool,
    plan: Mapping[str, Any] | None = None,
) -> dict:
    """Freeze new source bytes before truth/prediction work.

    The attestation is procedural metadata, not cryptographic proof that no human has
    previously inspected labels.  The machine-enforced portion is byte/dataset identity
    novelty versus all frozen R20/R21 artifacts listed in the plan.
    """
    frozen = load_replication_plan() if plan is None else dict(plan)
    verify_seal(frozen)
    family = str(dataset_family)
    if family not in ALLOWED_FAMILIES:
        raise ValueError("R22_REPLICATION_FAMILY_UNREGISTERED")
    dataset_id = str(dataset_id).strip()
    if not dataset_id:
        raise ValueError("R22_NEW_DATASET_ID_REQUIRED")
    if dataset_id in set(frozen["known_dataset_ids"]):
        raise ValueError("R22_KNOWN_DATASET_ID_REUSE_BLOCKED")
    if not operator_attests_pre_truth_inspection:
        raise ValueError("R22_PRE_TRUTH_INTAKE_ATTESTATION_REQUIRED")
    source = Path(source_path).resolve()
    if source == SOURCE_ROOT or source.is_relative_to(SOURCE_ROOT):
        raise ValueError("R22_EXTERNAL_HOLDOUT_INSIDE_SOURCE_TREE")
    identity = _artifact_identity(source)
    known_sources = set(frozen["known_source_artifact_sha256s"])
    if identity["source_artifact_sha256"] in known_sources:
        raise ValueError("R22_KNOWN_SOURCE_ARTIFACT_REUSE_BLOCKED")
    known_member_hashes = set(frozen["known_member_sha256s_by_family"].get(family, []))
    member_collisions = sorted({row["sha256"] for row in identity["files"]} & known_member_hashes)
    if member_collisions:
        raise ValueError("R22_KNOWN_SOURCE_MEMBER_REUSE_BLOCKED")
    return seal({
        "schema_version": "astrograph-r22-new-holdout-intake-receipt.v1",
        "status": "INTAKE_FROZEN_PRE_TRUTH",
        "dataset_id": dataset_id,
        "dataset_family": family,
        "source_artifact_kind": identity["kind"],
        "source_artifact_size_bytes": identity["source_artifact_size_bytes"],
        "source_artifact_sha256": identity["source_artifact_sha256"],
        "source_files": identity["files"],
        "operator_attests_pre_truth_inspection": True,
        "attestation_is_not_independent_proof": True,
        "known_source_collision_count": 0,
        "known_member_collision_count": 0,
        "r21_source_sha256": frozen["r21_source_sha256"],
        "replication_plan_sha256": frozen["receipt_sha256"],
    })


def _load_r21_registry() -> dict:
    registry = json.loads(R21_REGISTRY.read_text(encoding="utf-8"))
    verify_seal(registry)
    return registry


def admit_new_holdout_groups(
    intake_receipt: Mapping[str, Any],
    *,
    raw_group_ids: Sequence[str],
    evaluation_task_id: str,
    label_semantics_sha256: str,
    target_universe_sha256: str,
    plan: Mapping[str, Any] | None = None,
) -> dict:
    """Admit group identities without using truth labels.

    Identity novelty is source-content + raw-group identity novelty.  It does not prove
    statistical/physical independence of trajectories, spacecraft, or operating modes.
    """
    intake = dict(intake_receipt)
    verify_seal(intake)
    if intake.get("status") != "INTAKE_FROZEN_PRE_TRUTH":
        raise ValueError("R22_HOLDOUT_INTAKE_NOT_FROZEN")
    frozen = load_replication_plan() if plan is None else dict(plan)
    verify_seal(frozen)
    if intake.get("replication_plan_sha256") != frozen["receipt_sha256"]:
        raise ValueError("R22_INTAKE_PLAN_BINDING_DRIFT")
    family = intake["dataset_family"]
    expected = frozen["families"][family]
    if evaluation_task_id != expected["evaluation_task_id"]:
        raise ValueError("R22_TASK_IDENTITY_DRIFT")
    if label_semantics_sha256 != expected["label_semantics_sha256"]:
        raise ValueError("R22_LABEL_SEMANTICS_DRIFT")
    if target_universe_sha256 != expected["target_universe_sha256"]:
        raise ValueError("R22_TARGET_UNIVERSE_DRIFT")
    ids = [str(value) for value in raw_group_ids]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("R22_HOLDOUT_GROUP_IDENTITY_INVALID")
    canonical = [digest({"source_artifact_sha256": intake["source_artifact_sha256"], "raw_group_id": gid}) for gid in sorted(ids)]
    if len(canonical) != len(set(canonical)):
        raise ValueError("R22_HOLDOUT_CANONICAL_GROUP_COLLISION")
    # Raw ID reuse is reported but is not itself proof of contamination because an
    # independent source may legitimately restart local segment/event numbering.
    registry = _load_r21_registry()
    known_raw: set[str] = set()
    for dataset_name in expected["r21_registry_datasets"]:
        entry = registry["datasets"][dataset_name]
        for group_list in entry.get("group_ids_by_partition", {}).values():
            known_raw.update(map(str, group_list))
    raw_overlap = sorted(set(ids) & known_raw)
    return seal({
        "schema_version": "astrograph-r22-new-holdout-group-admission.v1",
        "status": "ADMITTED_IDENTITY_NOVELTY_NOT_INDEPENDENCE_PROOF",
        "dataset_id": intake["dataset_id"],
        "dataset_family": family,
        "intake_receipt_sha256": intake["receipt_sha256"],
        "source_artifact_sha256": intake["source_artifact_sha256"],
        "evaluation_task_id": evaluation_task_id,
        "label_semantics_sha256": label_semantics_sha256,
        "target_universe_sha256": target_universe_sha256,
        "raw_group_ids": sorted(ids),
        "raw_group_id_overlap_with_prior": raw_overlap,
        "canonical_group_identity_sha256": digest(canonical),
        "group_count": len(ids),
        "statistical_independence_proven": False,
        "independence_claim_boundary": "Content/group identity novelty only; physical/statistical independence requires study-design provenance.",
        "replication_plan_sha256": frozen["receipt_sha256"],
    })


def verify_new_holdout_admission(source_path: str | Path, intake_receipt: Mapping[str, Any], group_receipt: Mapping[str, Any]) -> None:
    intake = dict(intake_receipt); groups = dict(group_receipt)
    verify_seal(intake); verify_seal(groups)
    identity = _artifact_identity(Path(source_path))
    if identity["source_artifact_sha256"] != intake["source_artifact_sha256"] or identity["source_artifact_size_bytes"] != intake["source_artifact_size_bytes"]:
        raise ValueError("R22_NEW_HOLDOUT_SOURCE_DRIFT")
    if groups["intake_receipt_sha256"] != intake["receipt_sha256"] or groups["source_artifact_sha256"] != intake["source_artifact_sha256"]:
        raise ValueError("R22_INTAKE_GROUP_BINDING_DRIFT")
    if groups.get("replication_plan_sha256") != intake.get("replication_plan_sha256"):
        raise ValueError("R22_INTAKE_GROUP_PLAN_BINDING_DRIFT")
    canonical = [digest({"source_artifact_sha256": intake["source_artifact_sha256"], "raw_group_id": gid}) for gid in groups["raw_group_ids"]]
    if groups["canonical_group_identity_sha256"] != digest(canonical):
        raise ValueError("R22_GROUP_IDENTITY_DRIFT")


def _slope_by_index(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    xbar = (len(values) - 1) / 2.0
    ybar = statistics.fmean(values)
    denom = sum((index - xbar) ** 2 for index in range(len(values)))
    if denom <= 0:
        return 0.0
    return sum((index - xbar) * (value - ybar) for index, value in enumerate(values)) / denom


def _ops_short_fallback_score(points: Sequence[Mapping[str, Any]], reference: Mapping[str, Any], minimums: Mapping[str, int]) -> float:
    if not points:
        raise ValueError("R22_OPS_EMPTY_SEGMENT")
    for point in points:
        validate_online_fields(dict(point), track="raw")
    channel_set = {str(point["channel"]) for point in points}
    segment_set = {str(point["segment"]) for point in points}
    if len(channel_set) != 1 or len(segment_set) != 1:
        raise ValueError("R22_OPS_SEGMENT_IDENTITY_DRIFT")
    channel = next(iter(channel_set))
    if channel not in reference:
        raise ValueError("R22_OPS_REFERENCE_CHANNEL_UNAVAILABLE")
    values = [float(point["value"]) for point in points]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("R22_OPS_NONFINITE_OBSERVATION")
    feature_values: dict[str, float] = {}
    if len(values) >= int(minimums["mean"]):
        feature_values["mean"] = statistics.fmean(values)
    if len(values) >= int(minimums["span"]):
        feature_values["span"] = max(values) - min(values)
    if len(values) >= int(minimums["std"]):
        mean = statistics.fmean(values)
        feature_values["std"] = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    if len(values) >= int(minimums["trend"]):
        feature_values["trend"] = _slope_by_index(values)
    if not feature_values:
        raise ValueError("R22_OPS_FALLBACK_FEATURES_UNAVAILABLE")
    scores: list[float] = []
    for name, value in feature_values.items():
        ref = reference[channel][name]
        center = float(ref["center"]); scale = float(ref["scale"])
        if not (math.isfinite(center) and math.isfinite(scale) and scale > 0):
            raise ValueError("R22_OPS_REFERENCE_IDENTITY_DRIFT")
        scores.append(abs(value - center) / scale)
    return max(scores)


def ops_r21_frozen_remediation_prediction(points: Sequence[Mapping[str, Any]], plan: Mapping[str, Any] | None = None) -> dict:
    frozen = load_replication_plan() if plan is None else dict(plan)
    verify_seal(frozen)
    spec = frozen["families"][OPS_FAMILY]["remediation"]
    try:
        score = float(_raw_segment_score_core(list(points)))
        kind = "long"
        threshold = float(spec["long_path_threshold"])
    except ValueError as exc:
        if str(exc) not in set(spec["fallback_on_errors"]):
            raise
        score = _ops_short_fallback_score(points, spec["reference"], spec["operator_minimum_samples"])
        kind = "short"
        threshold = float(spec["short_fallback_threshold"])
    return {
        "status": "SCORED",
        "kind": kind,
        "score": score,
        "threshold": threshold,
        "predicted_anomaly": int(score >= threshold),
        "score_semantics": "UNCALIBRATED_ANOMALY_SCORE",
    }


def ops_r20_frozen_baseline_prediction(points: Sequence[Mapping[str, Any]], plan: Mapping[str, Any] | None = None) -> dict:
    frozen = load_replication_plan() if plan is None else dict(plan)
    verify_seal(frozen)
    threshold = float(frozen["families"][OPS_FAMILY]["baseline"]["threshold"])
    try:
        score = float(_raw_segment_score_core(list(points)))
        return {"status": "SCORED", "score": score, "threshold": threshold, "predicted_anomaly": int(score >= threshold),
                "score_semantics": "UNCALIBRATED_ANOMALY_SCORE"}
    except ValueError as exc:
        if str(exc) in set(frozen["families"][OPS_FAMILY]["baseline"]["expected_block_errors"]):
            return {"status": "BLOCKED", "reason": str(exc), "score_semantics": "UNCALIBRATED_ANOMALY_SCORE"}
        raise


def generate_ops_replication_predictions(
    source_path: str | Path,
    *,
    raw_segments: Mapping[str, Sequence[Mapping[str, Any]]],
    intake_receipt: Mapping[str, Any],
    group_receipt: Mapping[str, Any],
    plan: Mapping[str, Any] | None = None,
) -> dict:
    frozen = load_replication_plan() if plan is None else dict(plan)
    verify_seal(frozen)
    verify_new_holdout_admission(source_path, intake_receipt, group_receipt)
    groups = dict(group_receipt)
    if groups["dataset_family"] != OPS_FAMILY:
        raise ValueError("R22_OPS_REPLICATION_FAMILY_MISMATCH")
    expected_ids = set(groups["raw_group_ids"])
    if set(map(str, raw_segments)) != expected_ids:
        raise ValueError("R22_OPS_PREDICTION_GROUP_SET_DRIFT")
    baseline: dict[str, Any] = {}; remediation: dict[str, Any] = {}
    for sid in sorted(expected_ids):
        points = list(raw_segments[sid])
        if not points or {str(point["segment"]) for point in points} != {sid}:
            raise ValueError("R22_OPS_SEGMENT_PAYLOAD_IDENTITY_DRIFT")
        baseline[sid] = ops_r20_frozen_baseline_prediction(points, frozen)
        remediation[sid] = ops_r21_frozen_remediation_prediction(points, frozen)
    return seal({
        "schema_version": "astrograph-r22-ops-replication-predictions.v1",
        "status": "FROZEN_BEFORE_TRUTH_JOIN",
        "dataset_id": groups["dataset_id"],
        "group_admission_sha256": groups["receipt_sha256"],
        "source_artifact_sha256": groups["source_artifact_sha256"],
        "replication_plan_sha256": frozen["receipt_sha256"],
        "baseline_implementation_sha256": frozen["families"][OPS_FAMILY]["baseline"]["implementation_identity_sha256"],
        "remediation_implementation_sha256": frozen["families"][OPS_FAMILY]["remediation"]["implementation_identity_sha256"],
        "predictions": {"baseline": baseline, "remediation": remediation},
        "truth_joined": False,
        "external_validation_receipt_generated": False,
    })


def _ops_summary(predictions: Mapping[str, Any], truth: Mapping[str, int]) -> dict:
    eligible = sorted(truth)
    scored = [sid for sid in eligible if predictions[sid]["status"] == "SCORED"]
    blocked = [sid for sid in eligible if predictions[sid]["status"] != "SCORED"]
    if scored:
        labels = [int(truth[sid]) for sid in scored]
        scores01 = [float(predictions[sid]["predicted_anomaly"]) for sid in scored]
        metrics = binary_metrics(labels, scores01, 0.5)
        f05 = metrics.get("F0.5") or 0.0
    else:
        metrics = {}; f05 = 0.0
    coverage = len(scored) / len(eligible) if eligible else 0.0
    return {
        "eligible_count": len(eligible),
        "scored_count": len(scored),
        "blocked_count": len(blocked),
        "coverage": coverage,
        "covered_metrics": metrics,
        "coverage_weighted_F0.5": coverage * f05,
        "blocked_group_ids": blocked,
    }


def evaluate_ops_replication(prediction_receipt: Mapping[str, Any], truth: Mapping[str, int], group_receipt: Mapping[str, Any]) -> dict:
    predictions = dict(prediction_receipt); groups = dict(group_receipt)
    verify_seal(predictions); verify_seal(groups)
    if predictions.get("truth_joined") is not False:
        raise ValueError("R22_PREDICTIONS_NOT_PRETRUTH_FROZEN")
    if predictions["group_admission_sha256"] != groups["receipt_sha256"]:
        raise ValueError("R22_PREDICTION_GROUP_ADMISSION_DRIFT")
    expected = set(groups["raw_group_ids"])
    if set(map(str, truth)) != expected or any(int(value) not in (0, 1) for value in truth.values()):
        raise ValueError("R22_OPS_TRUTH_GROUP_OR_LABEL_DRIFT")
    baseline = predictions["predictions"]["baseline"]
    remediation = predictions["predictions"]["remediation"]
    bsum = _ops_summary(baseline, truth); rsum = _ops_summary(remediation, truth)
    paired = {"both_scored": 0, "corrected": 0, "induced_error": 0, "coverage_gain_correct": 0, "coverage_gain_wrong": 0}
    for sid in sorted(expected):
        bp, rp, y = baseline[sid], remediation[sid], int(truth[sid])
        if bp["status"] == "SCORED" and rp["status"] == "SCORED":
            paired["both_scored"] += 1
            bc = int(bp["predicted_anomaly"]) == y; rc = int(rp["predicted_anomaly"]) == y
            paired["corrected"] += int((not bc) and rc)
            paired["induced_error"] += int(bc and (not rc))
        elif bp["status"] != "SCORED" and rp["status"] == "SCORED":
            if int(rp["predicted_anomaly"]) == y:
                paired["coverage_gain_correct"] += 1
            else:
                paired["coverage_gain_wrong"] += 1
    return seal({
        "schema_version": "astrograph-r22-new-holdout-replication-report.v1",
        "family": OPS_FAMILY,
        "status": "NEW_HOLDOUT_REPLICATION_DESCRIPTIVE",
        "group_admission_sha256": groups["receipt_sha256"],
        "prediction_receipt_sha256": predictions["receipt_sha256"],
        "primary_endpoint": "COVERAGE_WEIGHTED_F0.5_DELTA",
        "baseline": bsum,
        "remediation": rsum,
        "primary_delta": rsum["coverage_weighted_F0.5"] - bsum["coverage_weighted_F0.5"],
        "coverage_delta": rsum["coverage"] - bsum["coverage"],
        "paired_descriptive": paired,
        "inferential_p_value": None,
        "independence_note": groups["independence_claim_boundary"],
        "retuning_performed": False,
        "external_validation_receipt_generated": False,
        "qualification": "NOT_FLIGHT_QUALIFIED",
    })


def build_esa_preprocessed_input_receipt(
    *,
    intake_receipt: Mapping[str, Any],
    group_receipt: Mapping[str, Any],
    channel_artifacts: Mapping[str, Mapping[str, Any]],
    preprocessor_source_sha256: str,
    transform: str,
    evaluation_range_ns: Sequence[int],
    plan: Mapping[str, Any] | None = None,
) -> dict:
    """Bind a future upstream preprocessing result; does not certify raw equivalence."""
    intake = dict(intake_receipt); groups = dict(group_receipt)
    verify_seal(intake); verify_seal(groups)
    frozen = load_replication_plan() if plan is None else dict(plan); verify_seal(frozen)
    spec = frozen["families"][ESA_FAMILY]
    if intake["dataset_family"] != ESA_FAMILY or groups["dataset_family"] != ESA_FAMILY:
        raise ValueError("R22_ESA_REPLICATION_FAMILY_MISMATCH")
    if groups.get("intake_receipt_sha256") != intake["receipt_sha256"] or groups.get("source_artifact_sha256") != intake["source_artifact_sha256"]:
        raise ValueError("R22_ESA_INTAKE_GROUP_BINDING_DRIFT")
    if intake.get("replication_plan_sha256") != frozen["receipt_sha256"] or groups.get("replication_plan_sha256") != frozen["receipt_sha256"]:
        raise ValueError("R22_ESA_PLAN_BINDING_DRIFT")
    if transform != spec["required_preprocessing_transform"]:
        raise ValueError("R22_ESA_PREPROCESSING_TRANSFORM_DRIFT")
    if len(evaluation_range_ns) != 2:
        raise ValueError("R22_ESA_EVALUATION_RANGE_INVALID")
    lower, upper = int(evaluation_range_ns[0]), int(evaluation_range_ns[1])
    if upper <= lower:
        raise ValueError("R22_ESA_EVALUATION_RANGE_INVALID")
    if not isinstance(preprocessor_source_sha256, str) or len(preprocessor_source_sha256) != 64 or any(c not in "0123456789abcdef" for c in preprocessor_source_sha256):
        raise ValueError("R22_ESA_PREPROCESSOR_SOURCE_IDENTITY_INVALID")
    channels = sorted(channel_artifacts)
    if channels != sorted(spec["channels"]):
        raise ValueError("R22_ESA_CHANNEL_SET_DRIFT")
    for channel, identity in channel_artifacts.items():
        if not isinstance(identity.get("sha256"), str) or len(identity["sha256"]) != 64 or int(identity.get("size", 0)) <= 0:
            raise ValueError("R22_ESA_PREPROCESSED_ARTIFACT_IDENTITY_INVALID")
    return seal({
        "schema_version": "astrograph-r22-esa-preprocessed-input-receipt.v1",
        "status": "PREPROCESSED_INPUT_BOUND_UPSTREAM_PROVENANCE_NOT_INDEPENDENTLY_REEXECUTED",
        "intake_receipt_sha256": intake["receipt_sha256"],
        "group_admission_sha256": groups["receipt_sha256"],
        "source_artifact_sha256": intake["source_artifact_sha256"],
        "preprocessor_source_sha256": preprocessor_source_sha256,
        "transform": transform,
        "evaluation_range_ns": [lower, upper],
        "channels": {name: dict(channel_artifacts[name]) for name in channels},
        "channel_manifest_sha256": digest([{"channel": name, **dict(channel_artifacts[name])} for name in channels]),
        "raw_to_preprocessed_equivalence_proven": False,
        "replication_plan_sha256": frozen["receipt_sha256"],
    })


def esa_channel_payload_identity(payload: Mapping[str, Any]) -> dict:
    """Canonical content identity for one admitted preprocessed ESA channel.

    The identity is over the arrays actually consumed by the frozen scorer, not
    merely the container/NPZ bytes.  A caller may additionally bind the file-byte
    identity in ``build_esa_preprocessed_input_receipt``.
    """
    channel = str(payload.get("channel", "")).strip()
    if not channel:
        raise ValueError("R22_ESA_CHANNEL_REQUIRED")
    times = np.asarray(payload.get("time_ns"), dtype=np.int64)
    values = np.asarray(payload.get("value"), dtype=np.float64)
    observed = np.asarray(payload.get("observed"), dtype=np.bool_)
    if times.ndim != 1 or values.ndim != 1 or observed.ndim != 1 or not (len(times) == len(values) == len(observed)):
        raise ValueError("R22_ESA_PREPROCESSED_ARRAY_SHAPE_INVALID")
    if len(times) == 0 or not np.all(np.isfinite(values[observed])):
        raise ValueError("R22_ESA_PREPROCESSED_ARRAY_INVALID")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("R22_ESA_TIME_GRID_NOT_STRICTLY_INCREASING")
    content = {
        "channel": channel,
        "decision_count": int(len(times)),
        "time_ns_sha256": hashlib.sha256(np.ascontiguousarray(times).tobytes(order="C")).hexdigest(),
        "value_sha256": hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest(),
        "observed_sha256": hashlib.sha256(np.ascontiguousarray(observed).tobytes(order="C")).hexdigest(),
    }
    return {**content, "array_payload_sha256": digest(content)}


def _alarm_events_from_decisions(time_ns: np.ndarray, alarm: np.ndarray, *, range_end_ns: int) -> list[dict[str, Any]]:
    if len(time_ns) != len(alarm):
        raise ValueError("R22_ESA_ALARM_GRID_SHAPE_INVALID")
    events: list[dict[str, Any]] = []
    run_start: int | None = None
    for index, flag in enumerate(alarm):
        if bool(flag) and run_start is None:
            run_start = int(time_ns[index])
        elif not bool(flag) and run_start is not None:
            events.append({"start_ns": run_start, "end_ns": int(time_ns[index]), "end_closed": False})
            run_start = None
    if run_start is not None:
        events.append({"start_ns": run_start, "end_ns": int(range_end_ns), "end_closed": True})
    return events


def generate_esa_transport_predictions(
    *,
    channel_payloads: Mapping[str, Mapping[str, Any]],
    group_receipt: Mapping[str, Any],
    preprocessed_receipt: Mapping[str, Any],
    plan: Mapping[str, Any] | None = None,
) -> dict:
    """Run the unchanged R20/R21 5-minute-score algorithm on governed preprocessed input."""
    groups = dict(group_receipt); pre = dict(preprocessed_receipt)
    verify_seal(groups); verify_seal(pre)
    frozen = load_replication_plan() if plan is None else dict(plan); verify_seal(frozen)
    spec = frozen["families"][ESA_FAMILY]
    if spec["comparison_semantics"] != "TRANSPORT_ONLY_IDENTICAL_BASELINE_AND_R21_SELECTION":
        raise ValueError("R22_ESA_COMPARISON_SEMANTICS_DRIFT")
    if pre["group_admission_sha256"] != groups["receipt_sha256"]:
        raise ValueError("R22_ESA_PREPROCESSING_GROUP_BINDING_DRIFT")
    if groups.get("replication_plan_sha256") != frozen["receipt_sha256"] or pre.get("replication_plan_sha256") != frozen["receipt_sha256"]:
        raise ValueError("R22_ESA_PLAN_BINDING_DRIFT")
    if sorted(channel_payloads) != sorted(spec["channels"]):
        raise ValueError("R22_ESA_CHANNEL_SET_DRIFT")
    common_times = None; combined = None; observed_any = None
    per_channel: dict[str, Any] = {}
    threshold = float(spec["threshold"])
    for channel in spec["channels"]:
        payload = dict(channel_payloads[channel])
        if payload.get("channel") != channel:
            raise ValueError("R22_ESA_CHANNEL_PAYLOAD_DRIFT")
        array_identity = esa_channel_payload_identity(payload)
        expected_channel_identity = pre["channels"].get(channel, {})
        expected_array_sha = expected_channel_identity.get("array_payload_sha256")
        if expected_array_sha is not None and expected_array_sha != array_identity["array_payload_sha256"]:
            raise ValueError("R22_ESA_PREPROCESSED_ARRAY_IDENTITY_DRIFT:" + channel)
        scores = _esa_score_core(payload, spec["reference"][channel])
        times = np.asarray(payload["time_ns"]); observed = np.asarray(payload["observed"], dtype=bool)
        alarm = (scores >= threshold) & observed
        if common_times is None:
            common_times = times.copy(); combined = alarm.copy(); observed_any = observed.copy()
        else:
            if not np.array_equal(common_times, times):
                raise ValueError("R22_ESA_CHANNEL_GRID_MISMATCH")
            combined |= alarm; observed_any |= observed
        per_channel[channel] = {
            "array_payload_sha256": array_identity["array_payload_sha256"],
            "score_sha256": hashlib.sha256(np.asarray(scores, dtype=np.float64).tobytes(order="C")).hexdigest(),
            "alarm_count": int(alarm.sum()),
            "observed_count": int(observed.sum()),
        }
    assert common_times is not None and combined is not None and observed_any is not None
    range_lower, range_upper = map(int, pre["evaluation_range_ns"])
    if not (range_lower <= int(common_times[0]) <= int(common_times[-1]) <= range_upper):
        raise ValueError("R22_ESA_DECISIONS_OUTSIDE_PREPROCESSED_RANGE")
    alarm_events = _alarm_events_from_decisions(common_times, combined, range_end_ns=range_upper)
    return seal({
        "schema_version": "astrograph-r22-esa-transport-predictions.v1",
        "status": "FROZEN_BEFORE_TRUTH_JOIN",
        "group_admission_sha256": groups["receipt_sha256"],
        "preprocessed_receipt_sha256": pre["receipt_sha256"],
        "comparison_semantics": spec["comparison_semantics"],
        "baseline_and_remediation_are_identical": True,
        "superiority_contrast_allowed": False,
        "threshold": threshold,
        "replication_plan_sha256": frozen["receipt_sha256"],
        "implementation_identity_sha256": spec["implementation_identity_sha256"],
        "time_ns_sha256": hashlib.sha256(np.asarray(common_times, dtype=np.int64).tobytes(order="C")).hexdigest(),
        "alarm_sha256": hashlib.sha256(np.asarray(combined, dtype=np.bool_).tobytes(order="C")).hexdigest(),
        "observed_sha256": hashlib.sha256(np.asarray(observed_any, dtype=np.bool_).tobytes(order="C")).hexdigest(),
        "decision_count": int(len(common_times)),
        "alarm_count": int(combined.sum()),
        "observed_count": int(observed_any.sum()),
        "decision_time_range_ns": [int(common_times[0]), int(common_times[-1])],
        "evaluation_range_ns": [range_lower, range_upper],
        "alarm_events": alarm_events,
        "per_channel": per_channel,
        "truth_joined": False,
        "external_validation_receipt_generated": False,
    })


def evaluate_esa_transport_replication(
    prediction_receipt: Mapping[str, Any],
    *,
    truth_events: Sequence[Mapping[str, Any]],
    full_range_ns: Sequence[int],
    group_receipt: Mapping[str, Any],
    plan: Mapping[str, Any] | None = None,
) -> dict:
    """Join truth only after the ESA prediction receipt has been frozen.

    R21 selected the same 5-minute/threshold-6 implementation as the R20
    baseline, so this function intentionally has no baseline-vs-remediation
    delta or superiority endpoint.  It is an independent transport replication
    of one unchanged detector.
    """
    predictions = dict(prediction_receipt); groups = dict(group_receipt)
    verify_seal(predictions); verify_seal(groups)
    frozen = load_replication_plan() if plan is None else dict(plan); verify_seal(frozen)
    spec = frozen["families"][ESA_FAMILY]
    if predictions.get("truth_joined") is not False or predictions.get("superiority_contrast_allowed") is not False:
        raise ValueError("R22_ESA_PREDICTION_RECEIPT_SEMANTICS_INVALID")
    if predictions.get("group_admission_sha256") != groups.get("receipt_sha256"):
        raise ValueError("R22_ESA_PREDICTION_GROUP_ADMISSION_DRIFT")
    if predictions.get("replication_plan_sha256") != frozen["receipt_sha256"]:
        raise ValueError("R22_ESA_REPLICATION_PLAN_DRIFT")
    if predictions.get("implementation_identity_sha256") != spec["implementation_identity_sha256"]:
        raise ValueError("R22_ESA_IMPLEMENTATION_IDENTITY_DRIFT")
    if len(full_range_ns) != 2:
        raise ValueError("R22_ESA_FULL_RANGE_INVALID")
    lower, upper = int(full_range_ns[0]), int(full_range_ns[1])
    if upper <= lower:
        raise ValueError("R22_ESA_FULL_RANGE_INVALID")
    decision_lower, decision_upper = map(int, predictions["decision_time_range_ns"])
    if not (lower <= decision_lower <= decision_upper <= upper):
        raise ValueError("R22_ESA_DECISIONS_OUTSIDE_EVALUATION_RANGE")
    if [lower, upper] != list(map(int, predictions.get("evaluation_range_ns", []))):
        raise ValueError("R22_ESA_EVALUATION_RANGE_DRIFT")
    admitted_truth: list[dict[str, Any]] = []
    truth_event_ids: set[str] = set()
    for row in truth_events:
        event = {
            "event_id": str(row["event_id"]),
            "category": str(row["category"]),
            "start_ns": int(row["start_ns"]),
            "end_ns": int(row["end_ns"]),
        }
        if not event["event_id"]:
            raise ValueError("R22_ESA_TRUTH_EVENT_ID_REQUIRED")
        truth_event_ids.add(event["event_id"])
        admitted_truth.append(event)
    metrics = evaluate_events(
        admitted_truth,
        list(predictions["alarm_events"]),
        full_range=(lower, upper),
        confirmation_duration_s=float(spec["confirmation_duration_s"]),
    )
    return seal({
        "schema_version": "astrograph-r22-new-holdout-replication-report.v1",
        "family": ESA_FAMILY,
        "status": "NEW_HOLDOUT_TRANSPORT_REPLICATION_DESCRIPTIVE",
        "comparison_semantics": spec["comparison_semantics"],
        "superiority_contrast_allowed": False,
        "group_admission_sha256": groups["receipt_sha256"],
        "prediction_receipt_sha256": predictions["receipt_sha256"],
        "primary_endpoint": spec["primary_endpoint"],
        "metrics": metrics,
        "truth_event_count": len(truth_event_ids),
        "retuning_performed": False,
        "baseline_vs_remediation_delta": None,
        "external_validation_receipt_generated": False,
        "qualification": "NOT_FLIGHT_QUALIFIED",
    })


def require_esa_no_superiority_contrast(report_or_request: Mapping[str, Any]) -> None:
    if report_or_request.get("superiority_contrast") or report_or_request.get("baseline_vs_remediation_delta") is not None:
        raise ValueError("R22_ESA_IDENTICAL_IMPLEMENTATION_SUPERIORITY_CLAIM_FORBIDDEN")


def build_r22_readiness(*, holdout_admissions: Mapping[str, Mapping[str, Any]] | None = None,
                        required_families: Sequence[str] | None = None,
                        plan: Mapping[str, Any] | None = None) -> dict:
    frozen = load_replication_plan() if plan is None else dict(plan); verify_seal(frozen)
    admissions = dict(holdout_admissions or {})
    required = list(required_families) if required_families is not None else [OPS_FAMILY, ESA_FAMILY]
    if not required or any(family not in ALLOWED_FAMILIES for family in required) or len(required) != len(set(required)):
        raise ValueError("R22_REQUIRED_FAMILY_SET_INVALID")
    ready: dict[str, bool] = {}
    blockers: list[str] = []
    for family in required:
        receipt = admissions.get(family)
        ok = False
        if receipt is not None:
            try:
                verify_seal(dict(receipt))
                ok = receipt.get("dataset_family") == family and receipt.get("status") == "ADMITTED_IDENTITY_NOVELTY_NOT_INDEPENDENCE_PROOF"
            except Exception:
                ok = False
        ready[family] = ok
        if not ok:
            blockers.append("NEW_HOLDOUT_NOT_ADMITTED:" + family)
    return seal({
        "schema_version": "astrograph-r22-new-replication-readiness.v1",
        "status": "READY" if all(ready.values()) else "READY_FOR_NEW_HOLDOUT_BLOCKED_DATASET",
        "required_families": required,
        "family_readiness": ready,
        "blockers": blockers,
        "r21_source_sha256": frozen["r21_source_sha256"],
        "replication_plan_sha256": frozen["receipt_sha256"],
        "performance_result_generated": False,
        "new_external_validation_receipt_generated": False,
        "qualification": "NOT_FLIGHT_QUALIFIED",
    })
