"""Leakage-controlled three-stage baselines for the synthetic PHM v6.0 corpus."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


SPLIT_INDICES = {
    "train": frozenset({0, 1, 2, 5, 6, 7}),
    "validation": frozenset({3, 8}),
    "test": frozenset({4, 9}),
}
EXCLUDED_EXACT = {"time_s", "packet_receive_time_s", "packet_sequence", "packet_valid"}
EXCLUDED_TOKENS = (
    "fault", "condition", "label", "active_effect", "event.", "sample_id",
    "pair_id", "remaining_ratio", "capacity_j", "capacity_bits", "leak_rate",
    "load_multiplier", "effective_", "truth",
)
STATS = ("mean", "std", "min", "max", "delta", "rms")


@dataclass(frozen=True)
class PHMRecord:
    sample_id: str
    sample_index: int
    subsystem: str
    condition_id: str
    activation_time_s: float
    event_start_s: float
    fault_manifestation_time_s: float
    fault_injection_phase: str
    telemetry_path: Path
    split: str


@dataclass
class PHMDataset:
    features: pd.DataFrame
    records: list[PHMRecord]
    components: np.ndarray


def split_for_index(sample_index: int) -> str:
    matches = [name for name, indices in SPLIT_INDICES.items() if sample_index in indices]
    if len(matches) != 1:
        raise ValueError(f"sample_index {sample_index} does not belong to exactly one split")
    return matches[0]


def load_records(dataset_root: str | Path) -> list[PHMRecord]:
    root = Path(dataset_root)
    with (root / "sample_index.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        PHMRecord(
            sample_id=row["sample_id"],
            sample_index=int(row["sample_index"]),
            subsystem=row["subsystem"],
            condition_id=row["condition_id"],
            activation_time_s=float(row["activation_time_s"]),
            event_start_s=(float(row["event_start_s"]) if row["event_start_s"] else math.nan),
            fault_manifestation_time_s=(
                float(row["fault_manifestation_time_s"])
                if row["fault_manifestation_time_s"] else math.nan
            ),
            fault_injection_phase=row["fault_injection_phase"],
            telemetry_path=root / row["telemetry_file"],
            split=split_for_index(int(row["sample_index"])),
        )
        for row in rows
    ]


def audit_split(records: Sequence[PHMRecord]) -> dict[str, Any]:
    by_split = {name: [row for row in records if row.split == name] for name in SPLIT_INDICES}
    groups = {name: {row.sample_index for row in values} for name, values in by_split.items()}
    overlaps = {
        "train_validation": sorted(groups["train"] & groups["validation"]),
        "train_test": sorted(groups["train"] & groups["test"]),
        "validation_test": sorted(groups["validation"] & groups["test"]),
    }
    result = {
        "schema_version": "astrograph-phm-v60-split-audit.v1",
        "primary_group_key": "sample_index shared across every subsystem and operating phase",
        "group_overlap": overlaps,
        "group_overlap_count": sum(len(item) for item in overlaps.values()),
        "splits": {},
    }
    for name, values in by_split.items():
        result["splits"][name] = {
            "sample_count": len(values),
            "fault_count": sum(row.condition_id != "nominal" for row in values),
            "nominal_count": sum(row.condition_id == "nominal" for row in values),
            "group_count": len(groups[name]),
            "subsystem_counts": dict(sorted(Counter(row.subsystem for row in values).items())),
        }
    expected = {
        "train": (336, 306, 30, 6),
        "validation": (112, 102, 10, 2),
        "test": (112, 102, 10, 2),
    }
    result["frozen_counts_match"] = all(
        (
            item["sample_count"], item["fault_count"], item["nominal_count"], item["group_count"]
        ) == expected[name]
        for name, item in result["splits"].items()
    )
    result["status"] = (
        "PASS" if result["group_overlap_count"] == 0 and result["frozen_counts_match"] else "FAIL"
    )
    return result


def is_allowed_channel(name: str) -> bool:
    lowered = name.lower()
    return name not in EXCLUDED_EXACT and not any(token in lowered for token in EXCLUDED_TOKENS)


def summarize_telemetry(path: Path, activation_time_s: float) -> dict[str, float]:
    frame = pd.read_csv(path, low_memory=False)
    if "time_s" not in frame:
        raise ValueError(f"telemetry lacks time_s: {path}")
    time_s = pd.to_numeric(frame["time_s"], errors="coerce")
    active = time_s.ge(activation_time_s)
    result: dict[str, float] = {}
    for name in frame.columns:
        if not is_allowed_channel(str(name)):
            continue
        series = pd.to_numeric(frame.loc[active, name], errors="coerce")
        values = series.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        for stat in STATS:
            key = f"{name}__{stat}"
            if not len(values):
                result[key] = math.nan
            elif stat == "mean":
                result[key] = float(np.mean(values))
            elif stat == "std":
                result[key] = float(np.std(values))
            elif stat == "min":
                result[key] = float(np.min(values))
            elif stat == "max":
                result[key] = float(np.max(values))
            elif stat == "delta":
                result[key] = float(values[-1] - values[0])
            else:
                result[key] = float(np.sqrt(np.mean(np.square(values))))
    return result


def build_dataset(
    dataset_root: str | Path,
    fault_to_component: dict[str, str],
    *,
    progress: bool = False,
) -> PHMDataset:
    records = load_records(dataset_root)
    rows: list[dict[str, float]] = []
    components: list[str] = []
    for index, record in enumerate(records, start=1):
        rows.append(summarize_telemetry(record.telemetry_path, record.activation_time_s))
        components.append(
            "nominal" if record.condition_id == "nominal" else fault_to_component[record.condition_id]
        )
        if progress and index % 50 == 0:
            print(f"[features] {index}/{len(records)} traces", flush=True)
    features = pd.DataFrame(rows)
    forbidden = [name for name in features if not is_allowed_channel(name.rsplit("__", 1)[0])]
    if forbidden:
        raise RuntimeError(f"forbidden feature leakage detected: {forbidden[:10]}")
    return PHMDataset(features=features, records=records, components=np.asarray(components, dtype=str))


def make_pipeline(frame: pd.DataFrame, *, random_state: int = 20260813):
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    usable = [
        name for name in frame.columns
        if frame[name].notna().sum() >= 2 and frame[name].nunique(dropna=True) > 1
    ]
    if not usable:
        raise ValueError("no usable telemetry features")
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=min(128, len(usable)))),
        ("classify", LogisticRegression(
            max_iter=4000, class_weight="balanced", random_state=random_state,
        )),
    ])
    return pipeline, usable




def _finite_f_classif(values: Any, labels: Any):
    """ANOVA scores with an explicit finite policy for constant-within-class channels."""
    import warnings
    from sklearn.feature_selection import f_classif

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scores, pvalues = f_classif(values, labels)
    scores = np.asarray(scores, dtype=float)
    pvalues = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(scores)
    ceiling = float(np.max(scores[finite])) + 1.0 if np.any(finite) else 1.0
    scores = np.nan_to_num(scores, nan=0.0, posinf=ceiling, neginf=0.0)
    pvalues = np.nan_to_num(pvalues, nan=1.0, posinf=1.0, neginf=1.0)
    return scores, pvalues

def _adaptive_candidate_pipeline(
    frame: pd.DataFrame,
    spec: dict[str, Any],
    *,
    random_state: int = 20260813,
):
    """Build a train-only candidate used by the PHM validation-selected cascade.

    Candidate families are deliberately small and deterministic.  The validation
    split chooses the family/threshold; the test split is not consulted until the
    selected pipeline is frozen.
    """
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_selection import SelectKBest
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    usable = [
        name for name in frame.columns
        if frame[name].notna().sum() >= 2 and frame[name].nunique(dropna=True) > 1
    ]
    if not usable:
        raise ValueError("no usable telemetry features")
    family = str(spec["family"])
    if family == "logistic":
        pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=_finite_f_classif, k=min(int(spec["k"]), len(usable)))),
            ("classify", LogisticRegression(
                max_iter=4000, class_weight="balanced", C=float(spec.get("C", 1.0)),
                random_state=random_state,
            )),
        ])
    elif family == "svc":
        pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=_finite_f_classif, k=min(int(spec["k"]), len(usable)))),
            ("classify", CalibratedClassifierCV(
                estimator=SVC(
                    C=float(spec.get("C", 10.0)), class_weight="balanced",
                    random_state=random_state,
                ),
                method="sigmoid",
                cv=5,
                ensemble=False,
            )),
        ])
    elif family == "extra_trees":
        pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("classify", ExtraTreesClassifier(
                n_estimators=int(spec.get("n_estimators", 500)),
                max_features=float(spec.get("max_features", 0.3)),
                min_samples_leaf=int(spec.get("min_samples_leaf", 2)),
                class_weight="balanced", random_state=random_state, n_jobs=-1,
            )),
        ])
    else:
        raise ValueError(f"unsupported adaptive PHM family: {family}")
    return pipeline, usable


def _adaptive_specs() -> list[dict[str, Any]]:
    # Order is a deterministic tie break after validation Macro-F1.
    return [
        {"id": "lr-k128", "family": "logistic", "k": 128, "C": 1.0},
        {"id": "lr-k256", "family": "logistic", "k": 256, "C": 1.0},
        {"id": "svc-c10-k256", "family": "svc", "k": 256, "C": 10.0},
        {"id": "extra-trees-500", "family": "extra_trees", "n_estimators": 500,
         "max_features": 0.3, "min_samples_leaf": 2},
    ]


def _route_class_probabilities(
    class_probabilities: np.ndarray,
    component_probabilities: np.ndarray,
    class_labels: list[str],
    component_labels: list[str],
    fault_to_component: dict[str, str],
    mode: str,
) -> np.ndarray:
    """Return class labels after an explicitly validation-selected hierarchy route."""
    if mode == "raw":
        return np.asarray(class_labels)[class_probabilities.argmax(1)]
    if mode == "hard":
        component_pred = np.asarray(component_labels)[component_probabilities.argmax(1)]
        output: list[str] = []
        for row_index, component in enumerate(component_pred):
            allowed = [
                index for index, fault in enumerate(class_labels)
                if fault_to_component[fault] == component
            ]
            values = class_probabilities[row_index, allowed]
            output.append(class_labels[allowed[int(np.argmax(values))]])
        return np.asarray(output, dtype=str)
    if not mode.startswith("soft:"):
        raise ValueError(f"unsupported class route: {mode}")
    alpha = float(mode.split(":", 1)[1])
    weighted = class_probabilities.copy()
    for class_index, fault in enumerate(class_labels):
        component = fault_to_component[fault]
        component_index = component_labels.index(component)
        weighted[:, class_index] *= np.maximum(
            component_probabilities[:, component_index], 1e-12
        ) ** alpha
    return np.asarray(class_labels)[weighted.argmax(1)]


def train_three_stage_adaptive(
    dataset: PHMDataset,
    fault_to_component: dict[str, str],
    *,
    random_state: int = 20260813,
    candidate_specs: list[dict[str, Any]] | None = None,
    classification_routes: list[str] | None = None,
    strategy_id: str = "adaptive_classical_validation_selected",
    model_description: str | None = None,
) -> dict[str, Any]:
    """Validation-selected low-capacity PHM cascade for retrospective trace screening.

    This keeps the same post-activation trace-summary information boundary as the
    historical LR regression.  Train fits candidate models, Validation selects the
    family/threshold/routing policy, and Test is evaluated only after those choices
    are frozen.  It is not the unknown-onset causal-window PHM runtime.
    """
    from astrograph.evaluation.data_exposure import guard_development
    guard_development('OPTIMIZE')
    records = dataset.records
    split = np.asarray([row.split for row in records])
    conditions = np.asarray([row.condition_id for row in records], dtype=str)
    detection_truth = np.where(conditions == "nominal", "nominal", "fault")
    fault_mask = conditions != "nominal"
    train = split == "train"
    validation = split == "validation"
    test = split == "test"
    train_fault = train & fault_mask
    validation_fault = validation & fault_mask
    test_fault = test & fault_mask

    specs = list(candidate_specs) if candidate_specs is not None else _adaptive_specs()
    if not specs:
        raise ValueError("at least one PHM candidate spec is required")
    selection_rows: dict[str, list[dict[str, Any]]] = {
        "detection": [], "localization": [], "classification": []
    }

    # Detection: model family and threshold are both Validation-only choices.
    detection_candidates: list[tuple[float, int, float, Any, list[str], dict[str, Any]]] = []
    for spec_index, spec in enumerate(specs):
        model, columns = _adaptive_candidate_pipeline(
            dataset.features.loc[train], spec, random_state=random_state
        )
        model.fit(dataset.features.loc[train, columns], detection_truth[train])
        probabilities = _aligned_probabilities(
            model, dataset.features.loc[validation], columns, ["nominal", "fault"]
        )[:, 1]
        best_score = -1.0
        best_threshold = 0.5
        for threshold in np.linspace(0.1, 0.9, 17):
            predicted = np.where(probabilities >= threshold, "fault", "nominal")
            score = metric_bundle(
                detection_truth[validation], predicted, ["nominal", "fault"]
            )["macro_f1"]
            key = (float(score), -abs(float(threshold) - 0.5))
            current = (best_score, -abs(best_threshold - 0.5))
            if key > current:
                best_score = float(score)
                best_threshold = float(threshold)
        row = {
            "candidate": spec, "validation_macro_f1": best_score,
            "threshold": best_threshold, "feature_count": len(columns),
        }
        selection_rows["detection"].append(row)
        detection_candidates.append(
            (best_score, -spec_index, best_threshold, model, columns, spec)
        )
    detection_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    detection_validation_f1, _, detection_threshold, detection_model, detection_columns, detection_spec = detection_candidates[0]

    # Localization: Validation Macro-F1 with deterministic candidate-order tie break.
    component_labels = sorted(set(dataset.components[train_fault]))
    localization_candidates: list[tuple[float, int, Any, list[str], dict[str, Any]]] = []
    for spec_index, spec in enumerate(specs):
        model, columns = _adaptive_candidate_pipeline(
            dataset.features.loc[train_fault], spec, random_state=random_state
        )
        model.fit(dataset.features.loc[train_fault, columns], dataset.components[train_fault])
        predicted = model.predict(dataset.features.loc[validation_fault, columns])
        score = metric_bundle(
            dataset.components[validation_fault], predicted, component_labels
        )["macro_f1"]
        selection_rows["localization"].append({
            "candidate": spec, "validation_macro_f1": float(score),
            "feature_count": len(columns),
        })
        localization_candidates.append((float(score), -spec_index, model, columns, spec))
    localization_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    localization_validation_f1, _, localization_model, localization_columns, localization_spec = localization_candidates[0]

    # Classification model and hierarchy route are selected jointly on Validation.
    class_labels = sorted(set(conditions[train_fault]))
    validation_component_probabilities = _aligned_probabilities(
        localization_model, dataset.features.loc[validation], localization_columns, component_labels
    )
    route_order = list(classification_routes) if classification_routes is not None else ["raw", "soft:0.25", "soft:0.5", "soft:1", "hard"]
    if not route_order:
        raise ValueError("at least one classification route is required")
    classification_candidates: list[tuple[float, int, int, Any, list[str], dict[str, Any], str]] = []
    for spec_index, spec in enumerate(specs):
        model, columns = _adaptive_candidate_pipeline(
            dataset.features.loc[train_fault], spec, random_state=random_state
        )
        model.fit(dataset.features.loc[train_fault, columns], conditions[train_fault])
        class_probabilities = _aligned_probabilities(
            model, dataset.features.loc[validation], columns, class_labels
        )
        candidate_routes = []
        for route_index, route in enumerate(route_order):
            routed = _route_class_probabilities(
                class_probabilities, validation_component_probabilities,
                class_labels, component_labels, fault_to_component, route,
            )
            score = metric_bundle(
                conditions[validation_fault], routed[fault_mask[validation]], class_labels
            )["macro_f1"]
            candidate_routes.append({"route": route, "validation_macro_f1": float(score)})
            classification_candidates.append(
                (float(score), -spec_index, -route_index, model, columns, spec, route)
            )
        selection_rows["classification"].append({
            "candidate": spec, "feature_count": len(columns),
            "routes": candidate_routes,
        })
    classification_candidates.sort(
        # Prefer a less coupled route (raw before soft/hard) when Validation scores tie;
        # only then use candidate order as the final deterministic tie break.
        key=lambda item: (item[0], item[2], item[1]), reverse=True
    )
    (classification_validation_f1, _, _, classification_model, classification_columns,
     classification_spec, classification_route) = classification_candidates[0]

    # Frozen Test evaluation.
    detection_score = _aligned_probabilities(
        detection_model, dataset.features.loc[test], detection_columns, ["nominal", "fault"]
    )[:, 1]
    detection_pred = np.where(detection_score >= detection_threshold, "fault", "nominal")
    component_prob_all = _aligned_probabilities(
        localization_model, dataset.features.loc[test], localization_columns, component_labels
    )
    component_pred_all = np.asarray(component_labels)[component_prob_all.argmax(1)]
    class_prob_all = _aligned_probabilities(
        classification_model, dataset.features.loc[test], classification_columns, class_labels
    )
    class_pred_all = _route_class_probabilities(
        class_prob_all, component_prob_all, class_labels, component_labels,
        fault_to_component, classification_route,
    )
    oracle_class = np.asarray(class_labels)[class_prob_all[test_fault[test]].argmax(1)]

    test_conditions = conditions[test]
    test_components = dataset.components[test]
    active_test = test_conditions != "nominal"
    cascade_pred = np.where(detection_pred == "nominal", "nominal", class_pred_all)
    cascade_labels = ["nominal", *class_labels]
    result = {
        "schema_version": "astrograph-phm-v60-three-stage-result.v2",
        "model": model_description or (
            "validation-selected classical cascade: train-only imputation/preprocessing; "
            "candidate pool {LogisticRegression, SVC, ExtraTrees}; Validation selects detector "
            "threshold, localization model, classification model and hierarchy route"
        ),
        "strategy": strategy_id,
        "feature_scope": "post_activation_trace_summary",
        "feature_count": int(dataset.features.shape[1]),
        "validation_sample_count": int(validation.sum()),
        "selection": {
            "detection": {
                "candidate": detection_spec, "threshold": float(detection_threshold),
                "validation_macro_f1": float(detection_validation_f1),
            },
            "localization": {
                "candidate": localization_spec,
                "validation_macro_f1": float(localization_validation_f1),
            },
            "classification": {
                "candidate": classification_spec, "route": classification_route,
                "validation_macro_f1": float(classification_validation_f1),
            },
            "candidate_results": selection_rows,
            "selection_boundary": "Train fits candidates; Validation selects; Test is evaluated after freeze.",
        },
        "detection": metric_bundle(
            detection_truth[test], detection_pred, ["nominal", "fault"]
        ),
        "localization": metric_bundle(
            test_components[active_test], component_pred_all[active_test], component_labels
        ),
        "classification_oracle_unconditioned": metric_bundle(
            test_conditions[active_test], oracle_class, class_labels
        ),
        "classification_predicted_component": metric_bundle(
            test_conditions[active_test], class_pred_all[active_test], class_labels
        ),
        "end_to_end_18_class": metric_bundle(test_conditions, cascade_pred, cascade_labels),
        "failure_attribution": {
            "fault_traces": int(active_test.sum()),
            "first_wrong_stage_detection": int(np.sum(active_test & (detection_pred == "nominal"))),
            "first_wrong_stage_localization": int(np.sum(
                active_test & (detection_pred == "fault") & (component_pred_all != test_components)
            )),
            "fully_correct_fault_traces": int(np.sum(
                active_test & (detection_pred == "fault") & (class_pred_all == test_conditions)
            )),
        },
        "test_predictions": [
            {
                "sample_id": record.sample_id, "truth": str(test_conditions[index]),
                "detection_score": float(detection_score[index]),
                "detection_threshold": float(detection_threshold),
                "predicted_component": str(component_pred_all[index]),
                "predicted_class": str(class_pred_all[index]),
                "class_probabilities": {
                    name: float(class_prob_all[index, class_index])
                    for class_index, name in enumerate(class_labels)
                },
                "component_probabilities": {
                    name: float(component_prob_all[index, component_index])
                    for component_index, name in enumerate(component_labels)
                },
            }
            for index, record in enumerate([row for row in records if row.split == "test"])
        ],
        "models": {
            "detection": detection_model,
            "localization": localization_model,
            "classification": classification_model,
        },
        "feature_columns": {
            "detection": detection_columns,
            "localization": localization_columns,
            "classification": classification_columns,
        },
        "claim_boundary": (
            "Single fixed grouped split on synthetic post-activation traces; retrospective trace-level "
            "screening only. The activation boundary is part of this experiment and the result is not "
            "an unknown-onset online or flight/safety claim."
        ),
    }
    return result

PROMOTED_EXTRA_TREES_SPEC: dict[str, Any] = {
    "id": "extra-trees-500",
    "family": "extra_trees",
    "n_estimators": 500,
    "max_features": 0.3,
    "min_samples_leaf": 2,
}


def train_three_stage_promoted_extra_trees(
    dataset: PHMDataset,
    fault_to_component: dict[str, str],
    *,
    random_state: int = 20260813,
) -> dict[str, Any]:
    """Promoted PHM v6 retrospective screening cascade.

    All three heads use the frozen ExtraTrees-500 family selected from the dev12
    comparison and grouped-CV study. Train fits the estimators, Validation selects
    only the detector threshold, classification uses the uncoupled raw route, and
    Test is evaluated after those choices are frozen. This promotion is limited to
    the known-activation post-activation trace-summary screening protocol; it is
    not the unknown-onset causal-window or flight/safety model.
    """
    from astrograph.evaluation.data_exposure import guard_development
    guard_development('FIT')
    return train_three_stage_adaptive(
        dataset,
        fault_to_component,
        random_state=random_state,
        candidate_specs=[dict(PROMOTED_EXTRA_TREES_SPEC)],
        classification_routes=["raw"],
        strategy_id="promoted_extra_trees_trace_screening_v1",
        model_description=(
            "promoted retrospective PHM cascade: ExtraTreesClassifier(500) for detection, "
            "localization and classification; train-only imputation; Validation selects only "
            "the detector threshold; classification uses the raw route"
        ),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_three_stage_promoted_bundle(
    dataset: PHMDataset,
    bundle_dir: str | Path,
    *,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load and evaluate the frozen dev13 PHM retrospective screening bundle.

    The bundle is fail-closed: model hashes, feature columns and (when supplied)
    the frozen ``sample_index.csv`` identity are verified before evaluation.
    It remains scoped to the known-activation retrospective protocol.
    """
    import joblib

    root = Path(bundle_dir)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "astrograph-phm-v60-promoted-model-bundle.v1":
        raise ValueError("unsupported PHM promoted model bundle schema")
    if manifest.get("strategy") != "promoted_extra_trees_trace_screening_v1":
        raise ValueError("unexpected PHM promoted strategy")
    if manifest.get("feature_scope") != "post_activation_trace_summary":
        raise ValueError("unexpected PHM promoted feature scope")
    if manifest.get("classification_route") != "raw":
        raise ValueError("unexpected PHM promoted classification route")
    if dataset_root is not None:
        sample_index = Path(dataset_root) / "sample_index.csv"
        expected = manifest.get("dataset_contract", {}).get("sample_index_sha256")
        if not sample_index.is_file() or (expected and _sha256_file(sample_index) != expected):
            raise ValueError("PHM promoted bundle dataset identity mismatch")

    models: dict[str, Any] = {}
    feature_columns = {str(k): list(v) for k, v in manifest["feature_columns"].items()}
    for head in ("detection", "localization", "classification"):
        model_entry = manifest["models"][head]
        model_path = root / model_entry["path"]
        if _sha256_file(model_path) != model_entry["sha256"]:
            raise ValueError(f"PHM promoted model hash mismatch: {head}")
        missing = [name for name in feature_columns[head] if name not in dataset.features.columns]
        if missing:
            raise ValueError(f"PHM promoted model missing input features for {head}: {missing[:5]}")
        models[head] = joblib.load(model_path)

    records = dataset.records
    split = np.asarray([row.split for row in records])
    conditions = np.asarray([row.condition_id for row in records], dtype=str)
    detection_truth = np.where(conditions == "nominal", "nominal", "fault")
    test = split == "test"
    test_conditions = conditions[test]
    active_test = test_conditions != "nominal"
    test_components = dataset.components[test]

    detection_threshold = float(manifest["detection_threshold"])
    detection_score = _aligned_probabilities(
        models["detection"], dataset.features.loc[test], feature_columns["detection"],
        ["nominal", "fault"],
    )[:, 1]
    detection_pred = np.where(detection_score >= detection_threshold, "fault", "nominal")

    component_labels = [str(x) for x in models["localization"].named_steps["classify"].classes_]
    component_prob_all = _aligned_probabilities(
        models["localization"], dataset.features.loc[test], feature_columns["localization"],
        component_labels,
    )
    component_pred_all = np.asarray(component_labels)[component_prob_all.argmax(1)]

    class_labels = [str(x) for x in models["classification"].named_steps["classify"].classes_]
    class_prob_all = _aligned_probabilities(
        models["classification"], dataset.features.loc[test], feature_columns["classification"],
        class_labels,
    )
    class_pred_all = np.asarray(class_labels)[class_prob_all.argmax(1)]
    oracle_class = np.asarray(class_labels)[class_prob_all[active_test].argmax(1)]
    cascade_pred = np.where(detection_pred == "nominal", "nominal", class_pred_all)
    cascade_labels = ["nominal", *class_labels]
    test_records = [row for row in records if row.split == "test"]

    return {
        "schema_version": "astrograph-phm-v60-three-stage-result.v2",
        "model": "frozen promoted ExtraTrees-500 retrospective PHM model bundle",
        "strategy": "promoted_extra_trees_trace_screening_v1",
        "model_bundle": {
            "bundle_id": manifest["bundle_id"],
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": _sha256_file(manifest_path),
            "loaded_frozen_artifacts": True,
        },
        "feature_scope": manifest["feature_scope"],
        "feature_count": int(dataset.features.shape[1]),
        "validation_sample_count": int(np.sum(split == "validation")),
        "selection": {
            "detection": {
                "candidate": manifest["model_spec"],
                "threshold": detection_threshold,
                "validation_macro_f1": None,
            },
            "localization": {"candidate": manifest["model_spec"], "validation_macro_f1": None},
            "classification": {
                "candidate": manifest["model_spec"],
                "route": manifest["classification_route"],
                "validation_macro_f1": None,
            },
            "selection_boundary": "Frozen dev13 promoted bundle; selection evidence is stored in the bundle governance record.",
        },
        "detection": metric_bundle(detection_truth[test], detection_pred, ["nominal", "fault"]),
        "localization": metric_bundle(
            test_components[active_test], component_pred_all[active_test], component_labels
        ),
        "classification_oracle_unconditioned": metric_bundle(
            test_conditions[active_test], oracle_class, class_labels
        ),
        "classification_predicted_component": metric_bundle(
            test_conditions[active_test], class_pred_all[active_test], class_labels
        ),
        "end_to_end_18_class": metric_bundle(test_conditions, cascade_pred, cascade_labels),
        "failure_attribution": {
            "fault_traces": int(active_test.sum()),
            "first_wrong_stage_detection": int(np.sum(active_test & (detection_pred == "nominal"))),
            "first_wrong_stage_localization": int(np.sum(
                active_test & (detection_pred == "fault") & (component_pred_all != test_components)
            )),
            "fully_correct_fault_traces": int(np.sum(
                active_test & (detection_pred == "fault") & (class_pred_all == test_conditions)
            )),
        },
        "test_predictions": [
            {
                "sample_id": record.sample_id,
                "truth": str(test_conditions[index]),
                "detection_score": float(detection_score[index]),
                "detection_threshold": detection_threshold,
                "predicted_component": str(component_pred_all[index]),
                "predicted_class": str(class_pred_all[index]),
                "class_probabilities": {
                    name: float(class_prob_all[index, class_index])
                    for class_index, name in enumerate(class_labels)
                },
                "component_probabilities": {
                    name: float(component_prob_all[index, component_index])
                    for component_index, name in enumerate(component_labels)
                },
            }
            for index, record in enumerate(test_records)
        ],
        "models": models,
        "feature_columns": feature_columns,
        "claim_boundary": manifest["claim_boundary"],
    }


def metric_bundle(truth: Sequence[str], predicted: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

    recalls = recall_score(truth, predicted, labels=list(labels), average=None, zero_division=0)
    support = Counter(truth)
    return {
        "sample_count": len(truth),
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=list(labels), average="macro", zero_division=0)),
        "per_class_recall": {name: float(value) for name, value in zip(labels, recalls, strict=True)},
        "support": {name: int(support[name]) for name in labels},
    }


def _aligned_probabilities(model: Any, frame: pd.DataFrame, columns: list[str], labels: list[str]) -> np.ndarray:
    raw = model.predict_proba(frame[columns])
    classes = list(model.named_steps["classify"].classes_)
    aligned = np.zeros((len(frame), len(labels)), dtype=float)
    for source_index, name in enumerate(classes):
        aligned[:, labels.index(str(name))] = raw[:, source_index]
    return aligned


def train_three_stage(dataset: PHMDataset, fault_to_component: dict[str, str]) -> dict[str, Any]:
    from astrograph.evaluation.data_exposure import guard_development
    guard_development('FIT')
    records = dataset.records
    split = np.asarray([row.split for row in records])
    conditions = np.asarray([row.condition_id for row in records], dtype=str)
    detection_truth = np.where(conditions == "nominal", "nominal", "fault")
    fault_mask = conditions != "nominal"
    train = split == "train"
    validation = split == "validation"
    test = split == "test"
    train_fault = train & fault_mask
    test_fault = test & fault_mask

    detection_model, detection_columns = make_pipeline(dataset.features.loc[train])
    detection_model.fit(dataset.features.loc[train, detection_columns], detection_truth[train])
    detection_score = _aligned_probabilities(
        detection_model, dataset.features.loc[test], detection_columns, ["nominal", "fault"]
    )[:, 1]
    detection_pred = np.where(detection_score >= 0.5, "fault", "nominal")

    component_labels = sorted(set(dataset.components[train_fault]))
    localization_model, localization_columns = make_pipeline(dataset.features.loc[train_fault])
    localization_model.fit(
        dataset.features.loc[train_fault, localization_columns], dataset.components[train_fault]
    )
    component_prob_all = _aligned_probabilities(
        localization_model, dataset.features.loc[test], localization_columns, component_labels
    )
    component_pred_all = np.asarray(component_labels)[component_prob_all.argmax(1)]

    class_labels = sorted(set(conditions[train_fault]))
    classification_model, classification_columns = make_pipeline(dataset.features.loc[train_fault])
    classification_model.fit(
        dataset.features.loc[train_fault, classification_columns], conditions[train_fault]
    )
    class_prob_all = _aligned_probabilities(
        classification_model, dataset.features.loc[test], classification_columns, class_labels
    )
    conditioned = np.zeros_like(class_prob_all)
    for row_index, component in enumerate(component_pred_all):
        allowed = [
            index for index, fault in enumerate(class_labels)
            if fault_to_component[fault] == component
        ]
        values = class_prob_all[row_index, allowed]
        conditioned[row_index, allowed] = values / values.sum() if values.sum() else 1.0 / len(allowed)
    class_pred_all = np.asarray(class_labels)[conditioned.argmax(1)]

    test_conditions = conditions[test]
    test_components = dataset.components[test]
    active_test = test_conditions != "nominal"
    cascade_pred = np.where(detection_pred == "nominal", "nominal", class_pred_all)
    cascade_labels = ["nominal", *class_labels]
    oracle_class = np.asarray(class_labels)[class_prob_all[active_test].argmax(1)]
    result = {
        "schema_version": "astrograph-phm-v60-three-stage-result.v1",
        "model": "train-only median imputation + standardization + univariate top-128 selection + class-balanced logistic regression per stage",
        "feature_count": int(dataset.features.shape[1]),
        "selected_feature_limit": 128,
        "validation_sample_count": int(validation.sum()),
        "detection": metric_bundle(
            detection_truth[test], detection_pred, ["nominal", "fault"]
        ),
        "localization": metric_bundle(
            test_components[active_test], component_pred_all[active_test], component_labels
        ),
        "classification_oracle_unconditioned": metric_bundle(
            test_conditions[active_test], oracle_class, class_labels
        ),
        "classification_predicted_component": metric_bundle(
            test_conditions[active_test], class_pred_all[active_test], class_labels
        ),
        "end_to_end_18_class": metric_bundle(test_conditions, cascade_pred, cascade_labels),
        "failure_attribution": {
            "fault_traces": int(active_test.sum()),
            "first_wrong_stage_detection": int(np.sum(active_test & (detection_pred == "nominal"))),
            "first_wrong_stage_localization": int(np.sum(
                active_test & (detection_pred == "fault") & (component_pred_all != test_components)
            )),
            "fully_correct_fault_traces": int(np.sum(
                active_test & (detection_pred == "fault") & (class_pred_all == test_conditions)
            )),
        },
        "test_predictions": [
            {
                "sample_id": record.sample_id,
                "truth": str(test_conditions[index]),
                "detection_score": float(detection_score[index]),
                "predicted_component": str(component_pred_all[index]),
                "predicted_class": str(class_pred_all[index]),
                "class_probabilities": {
                    name: float(conditioned[index, class_index])
                    for class_index, name in enumerate(class_labels)
                },
                "component_probabilities": {
                    name: float(component_prob_all[index, component_index])
                    for component_index, name in enumerate(component_labels)
                },
            }
            for index, record in enumerate([row for row in records if row.split == "test"])
        ],
        "models": {
            "detection": detection_model,
            "localization": localization_model,
            "classification": classification_model,
        },
        "feature_columns": {
            "detection": detection_columns,
            "localization": localization_columns,
            "classification": classification_columns,
        },
        "claim_boundary": "Single fixed grouped split on synthetic traces; trace-level screening, not online unknown-onset certification.",
    }
    return result


__all__ = [
    "PHMDataset", "PHMRecord", "SPLIT_INDICES", "audit_split", "build_dataset",
    "is_allowed_channel", "load_records", "metric_bundle", "split_for_index",
    "summarize_telemetry", "train_three_stage", "train_three_stage_adaptive",
    "train_three_stage_promoted_extra_trees", "load_three_stage_promoted_bundle",
    "PROMOTED_EXTRA_TREES_SPEC",
]
