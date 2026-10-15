#!/usr/bin/env python3
"""Deep multi-task screening on raw and paired-residual PHM v6.0 summaries."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astrograph.data_driven.phm_v60 import audit_split, build_dataset, load_records, metric_bundle

DEFAULT_DATASET = Path(os.environ.get("ASTROGRAPH_PHM_V60_DATA_ROOT", str(ROOT / "data" / "external" / "whole_spacecraft_phm_dataset_v6_0")))
HIERARCHY = ROOT / "configs/datasets/phm_v6_0/diagnosis_hierarchy.json"


def make_residual_features(dataset):
    residual = dataset.features.copy()
    nominal_by_group = {
        (row.sample_index, row.subsystem): dataset.features.iloc[index]
        for index, row in enumerate(dataset.records)
        if row.condition_id == "nominal"
    }
    for index, row in enumerate(dataset.records):
        residual.iloc[index] = dataset.features.iloc[index] - nominal_by_group[(row.sample_index, row.subsystem)]
    return residual


def prepare(frame, conditions: np.ndarray, split: np.ndarray, limit: int = 256):
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    train = split == "train"
    usable = [
        name for name in frame
        if frame.loc[train, name].notna().sum() >= 2
        and frame.loc[train, name].nunique(dropna=True) > 1
    ]
    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(frame.loc[train, usable])
    selector = SelectKBest(f_classif, k=min(limit, train_imputed.shape[1]))
    selector.fit(train_imputed, conditions[train])
    selected = [name for name, keep in zip(usable, selector.get_support(), strict=True) if keep]
    scaler = StandardScaler()
    train_selected = imputer.transform(frame[usable])[:, selector.get_support()]
    scaler.fit(train_selected[train])
    return scaler.transform(train_selected).astype(np.float32), selected


def make_model(kind: str, width_in: int, component_count: int, class_count: int):
    from torch import nn

    class ResidualBlock(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(),
                nn.Dropout(0.15), nn.Linear(width * 2, width), nn.Dropout(0.1),
            )

        def forward(self, x):
            return x + self.net(x)

    if kind == "mlp":
        encoder = nn.Sequential(
            nn.Linear(width_in, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(192, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(0.1),
        )
        width = 96
    else:
        encoder = nn.Sequential(
            nn.Linear(width_in, 160), nn.GELU(),
            ResidualBlock(160), ResidualBlock(160), nn.LayerNorm(160),
        )
        width = 160

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.detector = nn.Linear(width, 2)
            self.localizer = nn.Linear(width, component_count)
            self.classifier = nn.Linear(width, class_count)

        def forward(self, x):
            h = self.encoder(x)
            return self.detector(h), self.localizer(h), self.classifier(h)

    return Model()


def weights(values: np.ndarray, count: int):
    import torch

    frequencies = np.bincount(values[values >= 0], minlength=count).astype(float)
    result = np.sqrt(frequencies.sum() / np.maximum(frequencies * count, 1.0))
    return torch.tensor(result / result.mean(), dtype=torch.float32)


def infer(model: Any, x: np.ndarray):
    import torch

    model.eval()
    with torch.no_grad():
        return tuple(torch.softmax(value, 1).numpy() for value in model(torch.from_numpy(x)))


def evaluate(
    model: Any, x: np.ndarray, indices: np.ndarray, conditions: np.ndarray,
    components: np.ndarray, class_labels: list[str], component_labels: list[str],
    fault_to_component: dict[str, str],
) -> tuple[dict[str, Any], float]:
    dprob, lprob, cprob = infer(model, x[indices])
    truth = conditions[indices]
    component_truth = components[indices]
    active = truth != "nominal"
    detection_truth = np.where(active, "fault", "nominal")
    detection_pred = np.asarray(["nominal", "fault"])[dprob.argmax(1)]
    component_pred = np.asarray(component_labels)[lprob.argmax(1)]
    conditioned = np.zeros_like(cprob)
    for row_index, component in enumerate(component_pred):
        allowed = [i for i, fault in enumerate(class_labels) if fault_to_component[fault] == component]
        values = cprob[row_index, allowed]
        conditioned[row_index, allowed] = values / values.sum() if values.sum() else 1.0 / len(allowed)
    class_pred = np.asarray(class_labels)[conditioned.argmax(1)]
    cascade = np.where(detection_pred == "nominal", "nominal", class_pred)
    result = {
        "detection": metric_bundle(detection_truth, detection_pred, ["nominal", "fault"]),
        "localization": metric_bundle(component_truth[active], component_pred[active], component_labels),
        "classification_predicted_component": metric_bundle(truth[active], class_pred[active], class_labels),
        "classification_unconditioned": metric_bundle(
            truth[active], np.asarray(class_labels)[cprob[active].argmax(1)], class_labels
        ),
        "end_to_end_18_class": metric_bundle(truth, cascade, ["nominal", *class_labels]),
    }
    score = (
        0.2 * result["detection"]["macro_f1"]
        + 0.3 * result["localization"]["macro_f1"]
        + 0.5 * result["classification_predicted_component"]["macro_f1"]
    )
    return result, score


def train_one(
    *, kind: str, representation: str, x: np.ndarray, conditions: np.ndarray,
    components: np.ndarray, split: np.ndarray, class_labels: list[str],
    component_labels: list[str], fault_to_component: dict[str, str], epochs: int, seed: int,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_idx = np.flatnonzero(split == "train")
    validation_idx = np.flatnonzero(split == "validation")
    test_idx = np.flatnonzero(split == "test")
    condition_index = {name: index for index, name in enumerate(class_labels)}
    component_index = {name: index for index, name in enumerate(component_labels)}
    detection = np.asarray([0 if name == "nominal" else 1 for name in conditions], dtype=np.int64)
    class_truth = np.asarray([condition_index.get(name, -1) for name in conditions], dtype=np.int64)
    component_truth = np.asarray([component_index.get(name, -1) for name in components], dtype=np.int64)
    model = make_model(kind, x.shape[1], len(component_labels), len(class_labels))
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x[train_idx]), torch.from_numpy(detection[train_idx]),
            torch.from_numpy(component_truth[train_idx]), torch.from_numpy(class_truth[train_idx]),
        ),
        batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    criteria = (
        nn.CrossEntropyLoss(weight=weights(detection[train_idx], 2)),
        nn.CrossEntropyLoss(weight=weights(component_truth[train_idx], len(component_labels))),
        nn.CrossEntropyLoss(weight=weights(class_truth[train_idx], len(class_labels))),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=5e-3)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch, best_score = 0, -1.0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch_x, batch_d, batch_l, batch_c in loader:
            optimizer.zero_grad(set_to_none=True)
            dlogits, llogits, clogits = model(batch_x)
            active = batch_c >= 0
            loss = criteria[0](dlogits, batch_d)
            if active.any():
                loss = loss + criteria[1](llogits[active], batch_l[active])
                loss = loss + criteria[2](clogits[active], batch_c[active])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running += float(loss.detach()) * len(batch_x)
        validation, score = evaluate(
            model, x, validation_idx, conditions, components, class_labels,
            component_labels, fault_to_component,
        )
        history.append({"epoch": epoch, "train_loss": running / len(train_idx), "selection_score": score})
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{representation}/{kind}] epoch={epoch} loss={running / len(train_idx):.4f} val={score:.4f}", flush=True)
    model.load_state_dict(best_state)
    test, _ = evaluate(
        model, x, test_idx, conditions, components, class_labels, component_labels, fault_to_component
    )
    return {
        "family": f"{representation}_{kind}",
        "representation": representation,
        "encoder": kind,
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "training_seconds": time.perf_counter() - started,
        "history": history,
        **test,
    }


def render(report: dict[str, Any]) -> str:
    rows = []
    for item in report["models"]:
        rows.append(
            f"| {item['family']} / {item['seed']} | {item['parameter_count']} | {item['best_epoch']} | "
            f"{item['detection']['macro_f1']:.4f} | {item['localization']['macro_f1']:.4f} | "
            f"{item['classification_predicted_component']['macro_f1']:.4f} | "
            f"{item['end_to_end_18_class']['macro_f1']:.4f} |"
        )
    return """# PHM v6.0 深度多任务筛选

Validation 选择 checkpoint；Test 每模型只评估一次。paired residual 依赖同 Monte-Carlo 初值的正常数字孪生，不属于无条件在线可用信息。

| 模型 | 参数 | best epoch | Detection F1 | Localization F1 | Cascade class F1 | E2E F1 |
|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "scripts/integration/out/phm_v60_deep_comparison")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260814, 20260815, 20260816])
    args = parser.parse_args()
    hierarchy = json.loads(HIERARCHY.read_text())
    records = load_records(args.dataset_root)
    audit = audit_split(records)
    if audit["status"] != "PASS":
        raise RuntimeError(audit)
    dataset = build_dataset(args.dataset_root, hierarchy["fault_to_component"], progress=True)
    conditions = np.asarray([row.condition_id for row in records], dtype=str)
    split = np.asarray([row.split for row in records], dtype=str)
    class_labels = sorted(set(conditions) - {"nominal"})
    component_labels = sorted(set(dataset.components) - {"nominal"})
    representations = {"raw_summary": dataset.features, "paired_residual": make_residual_features(dataset)}
    models = []
    selected_features = {}
    for representation, frame in representations.items():
        x, selected = prepare(frame, conditions, split)
        selected_features[representation] = selected
        for kind in ("mlp", "resmlp"):
            for seed in args.seeds:
                models.append(train_one(
                    kind=kind, representation=representation, x=x, conditions=conditions,
                    components=dataset.components, split=split, class_labels=class_labels,
                    component_labels=component_labels, fault_to_component=hierarchy["fault_to_component"],
                    epochs=args.epochs, seed=seed,
                ))
                models[-1]["seed"] = seed
    report = {
        "schema_version": "astrograph-phm-v60-deep-comparison.v1",
        "split_audit": audit,
        "feature_count": {name: len(value) for name, value in selected_features.items()},
        "selected_features": selected_features,
        "models": models,
        "claim_boundary": (
            "Multi-seed E0 architecture screening with a known activation boundary; paired residual "
            "additionally requires a matched nominal digital twin. This is not an online or production claim."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(render(report))
    print(args.output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
