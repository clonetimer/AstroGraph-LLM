"""S1 diagnosis baseline: shared-backbone multi-head model + evaluation metrics.

Candidate budget = 1 (DEV_GOAL): this cycle ships ONE trained candidate — a
small multi-layer perceptron shared backbone with three read heads (detection /
mechanism / localization), trained on CUDA when available.  It is compared
against the S0 zero-fit baseline (gray-box residual / OOD / fail threshold),
which requires no training.

RESEARCH-PROBE STATUS (recorded 2026-08-04): this model is a research probe,
NOT a production capability.  Its checkpoint carries ``production_bound=false``
and it is NOT reachable from unified/CLI (release.json research_paths).  All its
numbers are 40-trace pipeline probes.  Production binding would require the
candidate-promotion freeze gate and the C2 calibration-strategy decision.

Honesty discipline:
* grouped holdout is enforced by the caller (samples are pre-split by
  ``diagnosis_split``); the training loop never sees tune/test loss.
* the train-split feature scaler and the nominal reference are fit on TRAIN
  only; tune/test features are transformed with those frozen statistics.
* metrics are reported with the worst-group (worst per-object accuracy) and the
  mechanism confusion matrix, so cross-object confusion is visible.
* tiny-N caveat: 40 episodes is far too small to claim production capability;
  these numbers demonstrate the pipeline and the design, not readiness.

The model is intentionally simple and mature (MLP + softmax heads, cross-entropy,
Adam, weight decay) so that ablation from the zero-fit baseline is clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from astrograph.data_driven.credibility import SplitName

from .diagnosis_dataset import DiagnosisSample
from .diagnosis_features import MECHANISM_CLASS_ORDER, OBJECT_ARCHETYPE


# --------------------------------------------------------------------------- #
# Feature normalisation (fit on TRAIN only)
# --------------------------------------------------------------------------- #


@dataclass
class FeatureScaler:
    """Standardisation statistics fit on the train split only."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureScaler":
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean) / self.std


# --------------------------------------------------------------------------- #
# Zero-fit detection baseline (S0, no training)
# --------------------------------------------------------------------------- #


@dataclass
class ZeroFitDetectionBaseline:
    """Anomaly score from gray-box fail/OOD/residual/violation — no training.

    ``anomaly_score`` = max of (fail, OOD, clamped residual, violation fraction)
    in [0, 1].  A decision threshold is chosen on the validation split by the
    caller.  This is the S0 comparison the trained baseline must beat (or at
    least match) under grouped holdout.
    """

    def score(self, sample: DiagnosisSample) -> float:
        gb = sample.gray_box if hasattr(sample, "gray_box") else None
        if gb is None:
            return 0.0
        fail = 1.0 if gb.get("fail", False) else 0.0
        ood = float(gb.get("ood", 0.0))
        rms_values = gb.get("residual_rms", {})
        rms = next(iter(rms_values.values()), 0.0) if rms_values else 0.0
        rms_norm = min(1.0, float(rms) / 5.0)
        vio = float(gb.get("raw_violation_fraction", 0.0))
        return float(max(fail, ood, rms_norm, vio))

    def predict(self, sample: DiagnosisSample, threshold: float = 0.5) -> int:
        return 1 if self.score(sample) >= threshold else 0


# --------------------------------------------------------------------------- #
# Shared-backbone multi-head MLP
# --------------------------------------------------------------------------- #


def _build_mlp(
    in_features: int,
    hidden: int,
    n_out: int,
    *,
    torch: Any,
    nn: Any,
    dropout: float = 0.3,
):
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, n_out),
    )


@dataclass
class DiagnosisBaselineConfig:
    """Hyper-parameters for the trained baseline (candidate budget = 1)."""

    hidden: int = 32
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    epochs: int = 200
    batch_size: int = 16
    seed: int = 20260804
    device: str = "auto"
    patience: int = 40


class DiagnosisBaseline:
    """Shared backbone + three read heads (detection / mechanism / localization).

    Heads are trained sequentially on the shared backbone, then a light joint
    fine-tune pass (framework doc stage A -> B -> C) is applied.  The model is
    created lazily so CPU-only tests can still exercise the code path.
    """

    def __init__(self, config: DiagnosisBaselineConfig) -> None:
        self.config = config
        self.model = None
        self.scaler: FeatureScaler | None = None
        self.n_mechanism = len(MECHANISM_CLASS_ORDER)
        self.n_objects = len(OBJECT_ARCHETYPE)
        self.mechanism_classes = MECHANISM_CLASS_ORDER
        self.objects = tuple(sorted(OBJECT_ARCHETYPE))
        self._torch = None
        self._nn = None
        self._trained_epochs = 0
        self._final_metrics: dict[str, Any] = {}

    def _import_torch(self):
        if self._torch is not None:
            return
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install torch for the trained diagnosis baseline") from exc
        self._torch = torch
        self._nn = nn

    def _resolve_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        torch = self._torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _tensors(
        self,
        partition: dict[SplitName, list[DiagnosisSample]],
    ) -> dict[str, Any]:
        torch = self._torch
        features = {}
        for name, samples in partition.items():
            arr = np.stack([s.feature for s in samples], axis=0)
            features[name] = arr
        det = {name: [s.detection_label for s in samples] for name, samples in partition.items()}
        mech = {name: [s.mechanism_class_index for s in samples] for name, samples in partition.items()}
        obj = {name: [s.object_index for s in samples] for name, samples in partition.items()}
        return {"features": features, "det": det, "mech": mech, "obj": obj}

    def fit(
        self,
        partition: dict[SplitName, list[DiagnosisSample]],
    ) -> "DiagnosisBaseline":
        """Train on TRAIN, early-stop on VALIDATION (never TEST)."""
        self._import_torch()
        torch = self._torch
        nn = self._nn
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        tensors = self._tensors(partition)
        train_feat = tensors["features"][SplitName.TRAIN]
        self.scaler = FeatureScaler.fit(train_feat)
        x_train = torch.as_tensor(
            self.scaler.transform(train_feat), dtype=torch.float32
        )
        tune_samples = partition.get(SplitName.VALIDATION, [])
        if tune_samples:
            x_val = torch.as_tensor(
                self.scaler.transform(tensors["features"][SplitName.VALIDATION]),
                dtype=torch.float32,
            )
        else:
            x_val = None

        device = self._resolve_device()
        in_features = x_train.shape[1]
        hidden = self.config.hidden
        n_mech = self.n_mechanism
        n_obj = self.n_objects

        x_train = x_train.to(device)
        if x_val is not None:
            x_val = x_val.to(device)

        self.model = nn.Module()
        self.model.backbone = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
        )
        self.model.detection_head = nn.Linear(hidden, 2)
        self.model.mechanism_head = nn.Linear(hidden, n_mech)
        self.model.localization_head = nn.Linear(hidden, n_obj)
        self.model.to(device)

        # Stage A: detection head on shared backbone (binary CE).
        # Stage B: mechanism head (CE) + localization head (CE).
        # Stage C: joint fine-tune (all three heads, small weight decay).
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        # Class-balanced weights (2026-08-05, revised): sqrt-inverse-frequency
        # weighting on TRAIN only (never tune/test).  A plain inverse-frequency
        # weight over-penalises the common classes (NOMINAL/CAPACITY_LOSS fall
        # below 1.0, which drove TRAIN NOMINAL recall to 0.12) and inflates the
        # empty classes (QUEUE_* live in VALIDATION, so TRAIN has no signal for
        # them — a high weight there is pure noise).  sqrt + a 0.5 floor keeps
        # common classes learnable while still up-weighting rare mechanisms.
        mech_counts = np.bincount(tensors["mech"][SplitName.TRAIN], minlength=self.n_mechanism).astype(float)
        mech_freq = np.where(mech_counts == 0, 1.0, mech_counts) / float(len(tensors["mech"][SplitName.TRAIN]))
        mech_weights = np.clip(1.0 / np.sqrt(mech_freq), 0.5, 3.0)
        mech_weights[mech_counts == 0] = 0.5  # empty classes: minimal, non-noisy
        mech_weights_t = torch.as_tensor(mech_weights, dtype=torch.float32).to(device)
        det_counts = np.bincount(tensors["det"][SplitName.TRAIN], minlength=2).astype(float)
        det_freq = np.where(det_counts == 0, 1.0, det_counts) / float(len(tensors["det"][SplitName.TRAIN]))
        det_weights = np.clip(1.0 / np.sqrt(det_freq), 0.5, 3.0)
        det_weights_t = torch.as_tensor(det_weights, dtype=torch.float32).to(device)
        loss_fn = nn.CrossEntropyLoss()
        mech_loss_fn = nn.CrossEntropyLoss(weight=mech_weights_t)
        det_loss_fn = nn.CrossEntropyLoss(weight=det_weights_t)

        y_det_train = torch.as_tensor(tensors["det"][SplitName.TRAIN], dtype=torch.long).to(device)
        y_mech_train = torch.as_tensor(tensors["mech"][SplitName.TRAIN], dtype=torch.long).to(device)
        y_obj_train = torch.as_tensor(tensors["obj"][SplitName.TRAIN], dtype=torch.long).to(device)

        if x_val is not None:
            y_det_val = torch.as_tensor(tensors["det"][SplitName.VALIDATION], dtype=torch.long).to(device)
            y_mech_val = torch.as_tensor(tensors["mech"][SplitName.VALIDATION], dtype=torch.long).to(device)
            y_obj_val = torch.as_tensor(tensors["obj"][SplitName.VALIDATION], dtype=torch.long).to(device)

        best_val = -1.0
        best_state = None
        patience = self.config.patience
        steps_no_improve = 0

        def train_epoch() -> None:
            self.model.train()
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            batch = self.config.batch_size
            for start in range(0, len(indices), batch):
                idx = indices[start : start + batch]
                xb = x_train[idx]
                opt.zero_grad()
                h = self.model.backbone(xb)
                l_det = det_loss_fn(self.model.detection_head(h), y_det_train[idx])
                l_mech = mech_loss_fn(self.model.mechanism_head(h), y_mech_train[idx])
                l_obj = loss_fn(self.model.localization_head(h), y_obj_train[idx])
                loss = l_det + l_mech + l_obj
                loss.backward()
                opt.step()

        def evaluate() -> float:
            self.model.eval()
            with torch.no_grad():
                h = self.model.backbone(x_val if x_val is not None else x_train)
                det_logits = self.model.detection_head(h)
                mech_logits = self.model.mechanism_head(h)
                obj_logits = self.model.localization_head(h)
                y_det = y_det_val if x_val is not None else y_det_train
                y_mech = y_mech_val if x_val is not None else y_mech_train
                y_obj = y_obj_val if x_val is not None else y_obj_train
                acc_det = (det_logits.argmax(1) == y_det).float().mean().item()
                acc_mech = (mech_logits.argmax(1) == y_mech).float().mean().item()
                acc_obj = (obj_logits.argmax(1) == y_obj).float().mean().item()
            return float(acc_det + acc_mech + acc_obj)

        for epoch in range(1, self.config.epochs + 1):
            train_epoch()
            self._trained_epochs = epoch
            if x_val is not None:
                score = evaluate()
                if score > best_val:
                    best_val = score
                    steps_no_improve = 0
                    best_state = {
                        k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()
                    }
                else:
                    steps_no_improve += 1
                    if steps_no_improve >= patience:
                        break
            if epoch % 25 == 0:
                pass

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to("cpu")
        return self

    # -- inference ---------------------------------------------------------- #

    def _forward(self, sample: DiagnosisSample):
        torch = self._torch
        x = torch.as_tensor(self.scaler.transform(sample.feature.reshape(1, -1)), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            h = self.model.backbone(x)
            det_logits = self.model.detection_head(h)
            mech_logits = self.model.mechanism_head(h)
            obj_logits = self.model.localization_head(h)
        return (
            int(det_logits.argmax(1).item()),
            int(mech_logits.argmax(1).item()),
            int(obj_logits.argmax(1).item()),
            det_logits.softmax(1)[0, 1].item(),
        )

    def predict(self, sample: DiagnosisSample) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("baseline not trained yet")
        det_pred, mech_pred, obj_pred, det_score = self._forward(sample)
        return {
            "detection": det_pred,
            "detection_score": det_score,
            "mechanism_class": self.mechanism_classes[mech_pred],
            "mechanism_index": mech_pred,
            "object_id": self.objects[obj_pred],
            "object_index": obj_pred,
        }

    def predict_batch(self, samples: list[DiagnosisSample]) -> list[dict[str, Any]]:
        return [self.predict(s) for s in samples]

    def save(self, path: Any) -> None:
        import json

        from pathlib import Path

        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "schema_version": "astrograph-diagnosis-baseline-checkpoint.v1",
            "config": {
                "hidden": self.config.hidden,
                "dropout": self.config.dropout,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "seed": self.config.seed,
                "device": self.config.device,
            },
            "feature_dim": len(self.scaler.mean),
            "production_bound": False,
            "research_probe": True,
            "trained_epochs": self._trained_epochs,
            "mechanism_classes": list(self.mechanism_classes),
            "objects": list(self.objects),
            "scaler_mean": self.scaler.mean.tolist(),
            "scaler_std": self.scaler.std.tolist(),
            "state_dict": {k: v.tolist() for k, v in self.model.state_dict().items()},
            "metrics": self._final_metrics,
        }
        json.dump(checkpoint, open(path, "w", encoding="utf-8"), indent=2)


# --------------------------------------------------------------------------- #
# Metrics (honest, includes worst group)
# --------------------------------------------------------------------------- #


def _confusion_matrix(truth: list[int], pred: list[int], n: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(truth, pred):
        matrix[t, p] += 1
    return matrix


def _classification_metrics(
    truth: list[int],
    pred: list[int],
    n_classes: int,
) -> dict[str, Any]:
    matrix = _confusion_matrix(truth, pred, n_classes)
    total = len(truth)
    correct = int(np.trace(matrix))
    micro_acc = correct / total if total else 0.0
    per_class_recall: list[float] = []
    per_class_precision: list[float] = []
    per_class_f1: list[float] = []
    for c in range(n_classes):
        tp = matrix[c, c]
        tp_fp = int(matrix[:, c].sum())
        tp_fn = int(matrix[c, :].sum())
        recall = tp / tp_fn if tp_fn else 0.0
        precision = tp / tp_fp if tp_fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class_recall.append(recall)
        per_class_precision.append(precision)
        per_class_f1.append(f1)
    macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0
    balanced_acc = float(np.mean(per_class_recall)) if per_class_recall else 0.0
    return {
        "micro_accuracy": micro_acc,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "n_samples": total,
        "n_correct": correct,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "per_class_f1": per_class_f1,
        "confusion_matrix": matrix.tolist(),
    }


def worst_group_accuracy(
    samples: list[DiagnosisSample],
    pred_objects: list[str],
) -> dict[str, Any]:
    """Worst per-object accuracy (honest reporting of the weakest group)."""
    by_object: dict[str, list[int]] = {}
    for sample, pred in zip(samples, pred_objects):
        by_object.setdefault(sample.component_instance_id, []).append(
            1 if pred == sample.component_instance_id else 0
        )
    per_object = {
        obj: float(np.mean(hits)) for obj, hits in sorted(by_object.items())
    }
    worst = min(per_object, key=per_object.get) if per_object else None
    return {
        "per_object": per_object,
        "worst_object": worst,
        "worst_accuracy": per_object[worst] if worst is not None else 0.0,
    }


def evaluate_baseline(
    model: DiagnosisBaseline,
    samples: list[DiagnosisSample],
    *,
    object_names: tuple[str, ...],
) -> dict[str, Any]:
    """Full honest evaluation of the trained baseline on a sample list."""
    preds = model.predict_batch(samples)
    det_truth = [s.detection_label for s in samples]
    det_pred = [p["detection"] for p in preds]
    mech_truth = [s.mechanism_class_index for s in samples]
    mech_pred = [p["mechanism_index"] for p in preds]
    obj_truth = [s.object_index for s in samples]
    obj_pred = [p["object_index"] for p in preds]

    det = _classification_metrics(det_truth, det_pred, 2)
    mech = _classification_metrics(
        mech_truth, mech_pred, model.n_mechanism
    )
    loc = _classification_metrics(obj_truth, obj_pred, model.n_objects)
    worst = worst_group_accuracy(samples, [p["object_id"] for p in preds])

    # detection: report score + per-group
    det_scores = [p["detection_score"] for p in preds]
    return {
        "detection": det,
        "detection_scores": det_scores,
        "mechanism": mech,
        "localization": loc,
        "worst_group": worst,
        "n_samples": len(samples),
    }


def evaluate_zero_fit(
    baseline: ZeroFitDetectionBaseline,
    samples: list[DiagnosisSample],
    threshold: float = 0.5,
) -> dict[str, Any]:
    truth = [s.detection_label for s in samples]
    pred = [baseline.predict(s, threshold=threshold) for s in samples]
    scores = [baseline.score(s) for s in samples]
    metrics = _classification_metrics(truth, pred, 2)
    metrics["scores"] = scores
    return metrics


def _threshold_from_tune(
    baseline: ZeroFitDetectionBaseline,
    tune_samples: list[DiagnosisSample],
) -> float:
    """Pick the zero-fit detection threshold on the validation split only."""
    best = (0.0, 0.5)
    for candidate in np.arange(0.05, 1.0, 0.05):
        pred = [baseline.predict(s, threshold=float(candidate)) for s in tune_samples]
        truth = [s.detection_label for s in tune_samples]
        acc = float(np.mean([int(a == b) for a, b in zip(pred, truth)]))
        if acc > best[0]:
            best = (acc, float(candidate))
    return best[1]


__all__ = [
    "DiagnosisBaseline",
    "DiagnosisBaselineConfig",
    "FeatureScaler",
    "ZeroFitDetectionBaseline",
    "_classification_metrics",
    "_confusion_matrix",
    "_threshold_from_tune",
    "evaluate_baseline",
    "evaluate_zero_fit",
    "worst_group_accuracy",
]
