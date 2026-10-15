"""Truth-joined, abstention-aware three-task scoring for frozen replay records.

This module is evaluator-only: callers finish inference and persist raw head
outputs BEFORE joining labels. No prediction is reconstructed from ground truth.
A derived system component is named separately from the independent model head.
"""
from __future__ import annotations

from collections import Counter
import math
from typing import Iterable, Mapping, Sequence

ABSTAIN = "__ABSTAIN__"
UNAVAILABLE = "__UNAVAILABLE__"
NORMAL = "__NORMAL__"
RESERVED = frozenset({ABSTAIN, UNAVAILABLE, NORMAL})
SINGLETON_KINDS = frozenset({"CLASS", "MODEL_ONLY", "PREFERRED_UNCONFIRMED"})


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str], *,
                           labels: Sequence[str]) -> dict:
    """Macro-average fixed labels; abstentions count as FN, never vanish.

    Confusion rows are the declared true labels. Columns additionally contain
    normal, abstention and unavailable. Zero-denominator class F1 is zero.
    """
    labels = list(labels)
    if len(y_true) != len(y_pred) or not labels or len(set(labels)) != len(labels):
        raise ValueError("aligned arrays and a unique fixed label universe required")
    if set(labels) & {ABSTAIN, UNAVAILABLE}:
        raise ValueError("abstention/unavailable cannot be target classes")
    if not set(y_true) <= set(labels):
        raise ValueError("truth contains an undeclared class")
    if not set(y_pred) <= set(labels) | RESERVED:
        raise ValueError("prediction contains an undeclared class")
    columns = labels + [k for k in (NORMAL, ABSTAIN, UNAVAILABLE) if k not in labels]
    counts = Counter(zip(y_true, y_pred))
    per_class = {}
    for label in labels:
        tp = counts[label, label]
        fp = sum(v for (t, p), v in counts.items() if p == label and t != label)
        fn = sum(v for (t, p), v in counts.items() if t == label and p != label)
        per_class[label] = {
            "tp": tp, "fp": fp, "fn": fn, "support": tp + fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
    n = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    output = sum(p not in {ABSTAIN, UNAVAILABLE} for p in y_pred)
    unavailable = sum(p == UNAVAILABLE for p in y_pred)
    return {
        "n": n, "correct": correct, "wrong_single_output": output - correct,
        "abstained": sum(p == ABSTAIN for p in y_pred), "unavailable": unavailable,
        "accuracy_all": correct / n if n else None,
        "execution_coverage": (n - unavailable) / n if n else None,
        "single_output_coverage": output / n if n else None,
        "conditional_accuracy": correct / output if output else None,
        "macro_f1": sum(v["f1"] for v in per_class.values()) / len(labels) if n else None,
        "macro_recall": sum(v["recall"] for v in per_class.values()) / len(labels) if n else None,
        "per_class": per_class,
        "confusion": {"rows": labels, "columns": columns,
                      "counts": [[counts[t, p] for p in columns] for t in labels]},
        "semantics": "fixed-label macro F1; abstentions and unavailable count as class FN; no selective deletion",
    }


def _fault_prediction(value: Mapping) -> str:
    kind = value.get("kind")
    ids = value.get("ids", [])
    if kind == "UNAVAILABLE":
        return UNAVAILABLE
    if kind == "NORMAL":
        if ids:
            raise ValueError("normal output cannot contain fault IDs")
        return NORMAL
    if kind in SINGLETON_KINDS:
        if len(ids) != 1:
            raise ValueError("singleton output must contain exactly one class")
        return ids[0]
    if kind in {"SET", "ABSTAIN"}:
        return ABSTAIN
    raise ValueError("unknown prediction kind")


def _detection_prediction(value: Mapping) -> str:
    if value.get("kind") == "SET" and value.get("ids"):
        return "ANOMALY"  # Type undecided does not erase an observed anomaly.
    pred = _fault_prediction(value)
    return pred if pred in RESERVED else "ANOMALY"


def _argmax(values: Mapping[str, float]) -> str:
    if not values:
        return UNAVAILABLE
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()) or abs(sum(values.values()) - 1) > .001:
        raise ValueError("invalid model head probability distribution")
    # Preserve declared class order for an exact probability tie, like np.argmax.
    return max(values, key=values.get)


def evaluate_three_task_replay(rows: Iterable[dict], *, fault_to_component: Mapping[str, str],
                               dataset_class_to_fault: Mapping[str, str],
                               branches: Sequence[str]) -> dict:
    rows = list(rows)
    faults = sorted(fault_to_component)
    components = sorted(set(fault_to_component.values()))
    if set(dataset_class_to_fault.values()) != set(faults):
        raise ValueError("class map and component hierarchy must cover the same faults")
    ids = [(r["case_id"], r["endpoint_s"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate replay windows")
    active = [r for r in rows if r["target"] is not None]
    for r in active:
        if r["target"] not in fault_to_component:
            raise ValueError("unmapped evaluation truth")
        if r.get("target_component") != fault_to_component[r["target"]]:
            raise ValueError("component truth differs from the versioned evaluator hierarchy")
    head_rows = [r for r in active if r.get("heads")]
    head_output = {}
    for label, field in (("localization_direct_head", "direct_component_probabilities"),
                         ("localization_class_derived", "derived_component_probabilities"),
                         ("localization_deployed_blend", "component_probabilities")):
        preds = [(_argmax(r["heads"][field]) if r.get("heads") else UNAVAILABLE) for r in active]
        head_output[label] = classification_metrics(
            [r["target_component"] for r in active], preds, labels=components)
    for label, field in (("classification_raw_head", "raw_class_probabilities"),
                         ("classification_deployed_policy", "dataset_class_probabilities")):
        preds = [(dataset_class_to_fault[_argmax(r["heads"][field])] if r.get("heads") else UNAVAILABLE)
                 for r in active]
        head_output[label] = classification_metrics([r["target"] for r in active], preds, labels=faults)
    head_output["scope"] = "all active-fault windows, before detector gating; direct, class-derived and deployed localization are distinct"
    report = {
        "schema_version": "astrograph-three-task-replay-metrics.v1",
        "windows": len(rows), "active_fault_windows": len(active),
        "normal_windows": len(rows)-len(active),
        "active_head_outputs_available": len(head_rows),
        "model_heads": head_output, "branches": {},
        "label_universe": {"faults": faults, "components": components},
        "claim_boundary": "retrospective same-simulator replay; working hypotheses are not confirmed diagnoses; derived system localization is not the independent localization head",
    }
    for branch in branches:
        if any(branch not in r for r in rows):
            raise ValueError(f"branch {branch} missing a planned window")
        if any(not set(r[branch].get("ids", [])) <= set(faults) for r in rows):
            raise ValueError("branch prediction contains an undeclared fault")
        truth_d = ["ANOMALY" if r["target"] else NORMAL for r in rows]
        pred_d = [_detection_prediction(r[branch]) for r in rows]
        executable = [i for i, p in enumerate(pred_d) if p != UNAVAILABLE]
        pred_f = [_fault_prediction(r[branch]) for r in active]
        pred_c = [fault_to_component[p] if p in fault_to_component else p for p in pred_f]
        # A class set can identify a component only when every candidate agrees.
        for index, r in enumerate(active):
            if r[branch]["kind"] == "SET":
                candidate_components = {fault_to_component[f] for f in r[branch]["ids"]}
                if len(candidate_components) == 1:
                    pred_c[index] = next(iter(candidate_components))
        detection = classification_metrics(truth_d, pred_d, labels=[NORMAL, "ANOMALY"])
        valid_detection = classification_metrics([truth_d[i] for i in executable],
                                                 [pred_d[i] for i in executable], labels=[NORMAL, "ANOMALY"])
        report["branches"][branch] = {
            "detection_all": detection,
            "detection_executable": valid_detection,
            "classification_system": classification_metrics([r["target"] for r in active], pred_f, labels=faults),
            "localization_system_derived": classification_metrics([r["target_component"] for r in active], pred_c, labels=components),
            "localization_semantics": "component inferred from system fault output or unanimous component of an unresolved candidate set; NOT a learned localization head",
        }
    return report
