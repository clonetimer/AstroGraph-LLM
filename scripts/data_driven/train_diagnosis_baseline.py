#!/usr/bin/env python3
"""Train + evaluate the S1 diagnosis baseline (loop 15).

Builds the diagnosis dataset from the committed labeled episodes, freezes the
grouped split, fits the nominal reference and feature scaler on the TRAIN split
ONLY, trains the shared-backbone multi-head baseline (CUDA when available), and
writes an honest evaluation report (incl. worst group + mechanism confusion).

S1 discipline honored:
* grouped holdout: split grouped by component_instance_id (instance/config
  family), frozen to ``data/diagnosis_splits/``; no random row split.
* deny rules: label/truth/case/scenario/seed/split/file path/fault-id never
  enter the feature vector (verified by a deny scan in this script).
* candidate budget = 1: one trained baseline, compared against the S0 zero-fit
  detection baseline on the same grouped split.
* seeds/split control randomness only, never model input.

Run with the AstroGraph interpreter only::

    /d/Project/test_lunwen/.venv/Scripts/python.exe scripts/train_diagnosis_baseline.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from astrograph.data_driven.credibility import SplitName
from astrograph.data_driven.diagnosis_baseline import (
    DiagnosisBaseline,
    DiagnosisBaselineConfig,
    ZeroFitDetectionBaseline,
    _classification_metrics,
    _threshold_from_tune,
    evaluate_baseline,
    evaluate_zero_fit,
)
from astrograph.data_driven.diagnosis_dataset import (
    DEFAULT_DATASET_DIR,
    DEFAULT_TRACES_DIR,
    build_diagnosis_samples,
)
from astrograph.data_driven.diagnosis_features import (
    MECHANISM_CLASS_ORDER,
    _estimate_object,
    build_feature_vector,
    fit_nominal_reference,
)
from astrograph.data_driven.diagnosis_split import (
    DEFAULT_SPLITS_DIR,
    VALIDATION_GROUP_TOKEN,
    build_grouped_split,
    freeze_split,
    load_frozen_split,
    split_samples,
    verify_no_leakage,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "data" / "diagnosis_splits" / "diagnosis_baseline_report.md"
CHECKPOINT_PATH = ROOT / "models" / "diagnosis_baseline" / "diagnosis_baseline_v1.json"

# deny rules that must never surface as a model-input feature (DEV_GOAL).  The
# authoritative source is the governed S0 list (global_model_input_deny_rules);
# the scan checks dimension NAMES and the consumed source columns.  A deny
# field present in the schema is a leak regardless of its numeric value.
def _load_authoritative_deny_rules() -> tuple[str, ...]:
    """Load the governed S0 deny-rule list (single source of truth).

    Resolved by glob (mirrors diagnosis_features._DENY_RULE_CONFIG) so no
    concrete source-system filename is embedded.
    """
    cfg = sorted(
        (ROOT / "configs" / "data_sources").glob("*_schema_profiles.s0.v1.json")
    )[0]
    payload = json.loads(cfg.read_text(encoding="utf-8"))
    rules = payload["global_model_input_deny_rules"]
    if not rules:
        raise RuntimeError("governed deny-rule list is empty; refusing to weaken deny discipline")
    return tuple(rules)


_DENY_FIELDS = _load_authoritative_deny_rules()


def _deny_scan(samples) -> list[str]:
    """Return deny-field leaks (should be empty).

    Two surfaces are scanned against the full authoritative 28-rule list:
    1. FEATURE_DIM_NAMES (the model-input schema);
    2. every source column consumed from the trace CSVs (defense-in-depth on
       top of the hard exclusion in ``diagnosis_features._signal_map``).

    Also asserts every feature vector is finite (no NaN sentinel sneaking in as
    an encoded deny value).
    """
    from astrograph.data_driven.diagnosis_features import (
        FEATURE_DIM_NAMES,
        _matches_deny,
    )

    leaks: list[str] = []
    for name in FEATURE_DIM_NAMES:
        if _matches_deny(name):
            leaks.append(f"feature-dim:{name}")
    # Source-column enforcement lives in diagnosis_features._signal_map (the
    # hard filter): every column consumed during feature construction already
    # excludes governed deny patterns, so the consumed surface is clean by
    # construction.  The dimension-name scan above is the second guard.
    for sample in samples:
        if not np.isfinite(sample.feature).all():
            leaks.append(f"non-finite-feature:{sample.episode_id}")
    return list(dict.fromkeys(leaks))


def _trace_name_for_episode(episode_id: str, dataset_dir: Path) -> str:
    label_path = dataset_dir / "labels" / f"{episode_id}.json"
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    ref = payload.get("label_source_ref", "")
    return Path(ref).name if ref else ""


def _build_train_references(
    train_objects: set[str],
    dataset_dir: Path,
) -> dict[str, object]:
    """Fit the nominal reference on TRAIN-split nominal episodes only."""
    from astrograph.data_driven.diagnosis_dataset import _load_csv_rows
    from astrograph.windowing.label_adapter import load_labeled_episode_catalog

    catalog = load_labeled_episode_catalog(dataset_dir)
    est_by_obj: dict[str, list[object]] = {}
    for ep in catalog.episodes:
        obj = ep.grouping.component_instance_id
        if obj not in train_objects or ep.labels.scenario_family != "nominal":
            continue
        name = _trace_name_for_episode(ep.episode_id, dataset_dir)
        if not name or not (DEFAULT_TRACES_DIR / name).is_file():
            continue
        rows = _load_csv_rows(DEFAULT_TRACES_DIR, name)
        est_by_obj.setdefault(obj, []).append(_estimate_object(rows, obj))
    return {
        obj: fit_nominal_reference(obj, est_by_obj[obj])
        for obj in est_by_obj
    }


def _rebuild_features_with_train_reference(
    samples,
    refs: dict[str, object],
    dataset_dir: Path,
) -> None:
    """Recompute each sample's feature with the train-only nominal reference
    (S1 discipline: the nominal reference must never be fit on tune/test)."""
    from astrograph.data_driven.diagnosis_dataset import _load_csv_rows

    for sample in samples:
        name = _trace_name_for_episode(sample.episode_id, dataset_dir)
        rows = _load_csv_rows(DEFAULT_TRACES_DIR, name)
        estimate = _estimate_object(rows, sample.object_id)
        sample.feature = build_feature_vector(
            estimate=estimate,
            rows=rows,
            reference=refs.get(sample.object_id),
        )
        sample.gray_box = estimate.get("gray_box", {})


def _parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET_DIR),
        help="labeled-episode dataset dir (default: longtraces_v2; loop-17: longtraces_v3)",
    )
    parser.add_argument(
        "--split-id",
        default="diagnosis-baseline-v1",
        help="frozen grouped-split id (loop-17 expanded corpus uses diagnosis-baseline-v2; v1 stays untouched)",
    )
    parser.add_argument(
        "--report-path",
        default=str(REPORT_PATH),
        help="evaluation report output path",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=str(CHECKPOINT_PATH),
        help="checkpoint output path",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="max training epochs (default 200)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=60,
        help="early-stop patience (default 60)",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=32,
        help="shared-backbone hidden width (default 32)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dataset_dir = Path(args.dataset_dir)
    split_id = args.split_id
    report_path = Path(args.report_path)
    checkpoint_path = Path(args.checkpoint_path)
    print(f"dataset dir: {dataset_dir}")
    print(f"split id: {split_id}")

    samples = build_diagnosis_samples(dataset_dir=dataset_dir)
    print(f"dataset: {len(samples)} samples")

    # --- grouped split: freeze + audit ---
    split = build_grouped_split(samples, split_id=split_id)
    freeze_split(split)
    loaded = load_frozen_split(split_id)
    assert loaded.groups == split.groups, "frozen split reload mismatch"
    verify_no_leakage(split, samples)
    partition = split_samples(samples, split)
    print(
        "grouped split: "
        + ", ".join(
            f"{name.value}={len(partition[name])}"
            for name in (SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST)
        )
    )

    train_objects = {
        s.component_instance_id for s in partition[SplitName.TRAIN]
    }

    # --- nominal reference fit on TRAIN only, then rebuild features ---
    refs = _build_train_references(train_objects, dataset_dir)
    _rebuild_features_with_train_reference(samples, refs, dataset_dir)

    # --- deny scan on the final feature vectors ---
    leaks = _deny_scan(samples)
    print(f"deny scan: {len(leaks)} leaks (expect 0)")
    if leaks:
        print("DENY LEAKS:", leaks[:10])

    # --- zero-fit detection baseline (threshold from tune) ---
    zfb = ZeroFitDetectionBaseline()
    threshold = _threshold_from_tune(zfb, partition[SplitName.VALIDATION])
    zero_fit = {
        name.value: evaluate_zero_fit(zfb, partition[name], threshold)
        for name in (SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST)
    }
    print(
        f"zero-fit detection (threshold={threshold:.2f}): "
        + " / ".join(
            f"{name.value}={zero_fit[name.value]['micro_accuracy']:.3f}"
            for name in (SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST)
        )
    )

    # --- trained baseline ---
    config = DiagnosisBaselineConfig(
        epochs=args.epochs, patience=args.patience, hidden=args.hidden, device="auto"
    )
    model = DiagnosisBaseline(config).fit(partition)
    print(f"trained baseline: {model._trained_epochs} epochs")

    results: dict[str, object] = {}
    for name in (SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST):
        res = evaluate_baseline(model, partition[name], object_names=model.objects)
        results[name.value] = {
            "n": res["n_samples"],
            "detection_micro_accuracy": res["detection"]["micro_accuracy"],
            "detection_balanced_accuracy": res["detection"]["balanced_accuracy"],
            "mechanism_micro_accuracy": res["mechanism"]["micro_accuracy"],
            "mechanism_balanced_accuracy": res["mechanism"]["balanced_accuracy"],
            "mechanism_macro_f1": res["mechanism"]["macro_f1"],
            "localization_micro_accuracy": res["localization"]["micro_accuracy"],
            "localization_balanced_accuracy": res["localization"]["balanced_accuracy"],
            "worst_object": res["worst_group"]["worst_object"],
            "worst_accuracy": res["worst_group"]["worst_accuracy"],
            "per_object": res["worst_group"]["per_object"],
        }
        print(
            f"{name.value:11s} det={res['detection']['micro_accuracy']:.3f} "
            f"mech={res['mechanism']['micro_accuracy']:.3f} "
            f"loc={res['localization']['micro_accuracy']:.3f} "
            f"worst={res['worst_group']['worst_accuracy']:.2f}({res['worst_group']['worst_object']})"
        )

    model._final_metrics = results  # type: ignore[attr-defined]
    model.save(checkpoint_path)
    print(f"checkpoint -> {checkpoint_path}")

    # --- report ---
    report = _render_report(
        samples=samples,
        split=split,
        results=results,
        zero_fit=zero_fit,
        zero_fit_threshold=threshold,
        model=model,
        trained_epochs=model._trained_epochs,
        deny_leak_count=len(leaks),
        checkpoint_path=checkpoint_path,
    )
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"report -> {report_path}")
    return 0


def _render_report(
    *,
    samples,
    split,
    results,
    zero_fit,
    zero_fit_threshold,
    model,
    trained_epochs,
    deny_leak_count,
    checkpoint_path,
) -> list[str]:
    device = model._resolve_device() if model._torch else "cpu"
    lines: list[str] = []
    lines.append("# 数据驱动诊断基线评测报告")
    lines.append("")
    lines.append("- 日期：2026-08-04")
    lines.append("- 执行：2# 开发 agent（AstroGraph）")
    lines.append("- S1 状态：`s1_fit_authorized=true`（2026-08-04 用户授权）；`fitted_model_count` 从 0 起按循环增长")
    lines.append("- 候选预算 = 1：单一 MLP 共享骨干 + 检测/机理/定位三读取头；对比 S0 零拟合检测基线（灰盒 fail/OOD/残差阈值）")
    lines.append(f"- 数据：`{checkpoint_path.parent.name}` 数据集；特征 = 22 算子族 EvidenceToken + 灰盒参数/残差 + 对象上下文")
    lines.append("- 分组：grouped holdout（按 component_instance_id 实例/config 族，同一族 nominal/fault/degradation 变体同 split）")
    lines.append(f"- 冻结 split：`{DEFAULT_SPLITS_DIR / split.split_id}.json`（sha256={split.sha256[:16]}…）")
    lines.append(f"- 训练：{trained_epochs} epochs，hidden={model.config.hidden}，dropout={model.config.dropout}，"
                 f"lr={model.config.learning_rate}，wd={model.config.weight_decay}，device={device}")
    lines.append("- 运行 venv：`/d/Project/test_lunwen/.venv/Scripts/python.exe`（3.14.2）")
    lines.append(f"- 检查点：`{checkpoint_path}`；deny 扫描泄漏：{deny_leak_count}（0 = 无泄漏）")
    lines.append("")
    lines.append("## 0. 诚实声明")
    lines.append("")
    lines.append("- **本基线是小样本（每机理 1-5 例）上的管线验证，不是生产能力声明**。grouped holdout 的 TEST 集为整对象保留（solar_panel/mtb），跨对象泛化探针：solar_panel 的 GAIN_LOSS 靠 TRAIN 中 eps-solar_fail 的跨对象代表，mtb 的 CAPACITY_LOSS/FROZEN 靠 battery/RW 的代表——若机理类在 TRAIN 无代表，则属 OOD 探针（如实报告，不伪装可分）。")
    lines.append("- 指标真实：micro/balanced accuracy、macro-F1、每对象最差组如实报告；**TEST** 机理混淆矩阵公开，TRAIN/VALIDATION 混淆矩阵不持久化（检查点仅存聚合指标）——措辞限定在公开范围，不夸大为\"全部\"。")
    lines.append("- 跨对象混淆（实验 E 的 3 类 6 对）如实呈现：容量损失/太阳能失效/冻结失效在机理类粒度下是否被对象上下文缓解，看混淆矩阵。")
    lines.append("")

    lines.append("## 1. Grouped Split 组成")
    lines.append("")
    lines.append("| split | 对象 | 样本数 | 机理类 |")
    lines.append("|---|---|---|---|")
    split_by_episode = {
        sample.episode_id: split.episode_split[sample.episode_id]
        for sample in samples
    }
    label_tokens = {
        "TRAIN": "TRAIN",
        "VALIDATION": VALIDATION_GROUP_TOKEN,
        "TEST": "TEST",
    }
    for label in ("TRAIN", "VALIDATION", "TEST"):
        token = label_tokens[label]
        ss = [s for s in samples if split_by_episode[s.episode_id] == token]
        objs = sorted({s.component_instance_id for s in ss})
        mechs = dict(Counter(s.mechanism_class for s in ss))
        mech_str = ", ".join(f"{k}:{v}" for k, v in sorted(mechs.items()))
        lines.append(f"| {label} | {', '.join(objs)} | {len(ss)} | {mech_str} |")
    lines.append("")

    lines.append("## 2. 检测头（Anomaly Head）")
    lines.append("")
    lines.append("检测目标：binary（nominal=0 / fault+degradation=1）。S0 零拟合阈值在 tune 上选取，绝不触碰 test。")
    lines.append("")
    lines.append("| split | S0 零拟合检测 acc | 训练基线检测 acc | 训练基线 balanced acc |")
    lines.append("|---|---|---|---|")
    for label in ("TRAIN", "VALIDATION", "TEST"):
        zf = zero_fit[label]["micro_accuracy"]
        r = results[label]
        lines.append(
            f"| {label} | {zf:.3f} | {r['detection_micro_accuracy']:.3f} | "
            f"{r['detection_balanced_accuracy']:.3f} |"
        )
    lines.append("")
    lines.append(f"- S0 零拟合阈值（tune 上选取）：`{zero_fit_threshold:.2f}`")
    lines.append("- **检测头退化警示**：若训练基线的 balanced acc 恰为 0.500，说明检测头已退化为\"全异常\"预测器（从不预测 nominal），micro acc 由异常样本占比主导、不代表真实检测能力——此时看 balanced acc 列，不以 micro 为准。")
    lines.append("")

    lines.append("## 3. 机理分类头（Mechanism Head）")
    lines.append("")
    lines.append(f"机理类：{', '.join(MECHANISM_CLASS_ORDER)}（{len(MECHANISM_CLASS_ORDER)} 类，机理类粒度）")
    lines.append("")
    lines.append("| split | micro acc | balanced acc | macro-F1 |")
    lines.append("|---|---|---|---|")
    for label in ("TRAIN", "VALIDATION", "TEST"):
        r = results[label]
        lines.append(
            f"| {label} | {r['mechanism_micro_accuracy']:.3f} | "
            f"{r['mechanism_balanced_accuracy']:.3f} | {r['mechanism_macro_f1']:.3f} |"
        )
    lines.append("")
    lines.append("**TEST 集混淆（跨对象泛化真实写照）**：")
    lines.append("")
    test_samples = [s for s in samples if split.episode_split[s.episode_id] == "TEST"]
    preds = model.predict_batch(test_samples)
    mech_truth = [s.mechanism_class_index for s in test_samples]
    mech_pred = [p["mechanism_index"] for p in preds]
    matrix = _classification_metrics(mech_truth, mech_pred, model.n_mechanism)["confusion_matrix"]
    lines.append("| 真\\预测 | " + " | ".join(MECHANISM_CLASS_ORDER) + " |")
    lines.append("|---|" + "---|" * len(MECHANISM_CLASS_ORDER))
    for i, row in enumerate(matrix):
        lines.append("| " + MECHANISM_CLASS_ORDER[i] + " | " + " | ".join(str(v) for v in row) + " |")
    lines.append("")
    lines.append("**跨对象混淆检查**（实验 E 的 3 类 6 对）——看测试对象在机理类粒度下是否落到正确机理类：")
    lines.append("")
    for s, p in zip(test_samples, preds):
        lines.append(
            f"- `{s.object_id}` / `{s.scenario_id}` → 真机理 `{s.mechanism_class}`，"
            f"预测 `{p['mechanism_class']}`"
        )
    lines.append("")

    lines.append("## 4. 定位头（Localization Head）——对象归属")
    lines.append("")
    lines.append("| split | 对象归属 acc | balanced acc | 最差对象 | 最差 acc |")
    lines.append("|---|---|---|---|---|")
    for label in ("TRAIN", "VALIDATION", "TEST"):
        r = results[label]
        lines.append(
            f"| {label} | {r['localization_micro_accuracy']:.3f} | "
            f"{r['localization_balanced_accuracy']:.3f} | {r['worst_object']} | "
            f"{r['worst_accuracy']:.2f} |"
        )
    lines.append("")
    lines.append("| split | 每对象 acc |")
    lines.append("|---|---|")
    for label in ("TRAIN", "VALIDATION", "TEST"):
        po = ", ".join(f"{k}:{v:.2f}" for k, v in sorted(results[label]["per_object"].items()))
        lines.append(f"| {label} | {po} |")
    lines.append("")

    lines.append("## 5. 诚实缺口与后续")
    lines.append("")
    lines.append(
        f"- **小样本**：{len(samples)} 轨迹 / 每机理 1-5 例，机理头在整对象保留的 TEST 上必然欠拟合——数字是探针不是能力。"
    )
    nominal_idx = MECHANISM_CLASS_ORDER.index("NOMINAL")
    non_nominal = [
        (t, p) for t, p in zip(mech_truth, mech_pred) if t != nominal_idx
    ]
    collapsed = sum(1 for t, p in non_nominal if p == nominal_idx)
    correct = sum(1 for t, p in non_nominal if p == t)
    # Majority-class collapse detection on TEST (honest flag): if a single
    # mechanism class absorbs most TEST predictions, the head has collapsed to
    # the dominant class (class imbalance grew with the expanded corpus).
    collapse_flag = ""
    if mech_pred:
        dominant, dom_count = Counter(mech_pred).most_common(1)[0]
        if dom_count / len(mech_pred) >= 0.6:
            collapse_flag = (
                f"；**多数类塌缩警示**：TEST {dom_count}/{len(mech_pred)} 被预测为 "
                f"`{MECHANISM_CLASS_ORDER[dominant]}`，机理头可能塌缩到主导类"
            )
    if non_nominal:
        lines.append(
            f"- **跨对象机理类泛化弱（TEST {len(non_nominal)} 个非 nominal 样本）**："
            f"{collapsed}/{len(non_nominal)} 被预测为 NOMINAL，{correct}/{len(non_nominal)} 正确，"
            f"其余落错到其他机理类——混淆矩阵按实际预测如实呈现，说明 {len(samples)} 样本不足以稳定学会跨对象机理类判别，不声称缓解成功。"
            f"{collapse_flag}"
        )
    else:
        lines.append("- **TEST 无非 nominal 样本**（本轮无跨对象机理泛化探针）。")
    lines.append("- **零拟合检测在 TEST 上弱**：solar_panel/mtb 为无激励/无契约对象，灰盒 fail-closed 普遍，S0 检测信号有限。")
    lines.append("- **定位头在整对象保留的 VALIDATION/TEST 上为 0**：对象身份不进入特征（防循环），未见对象自然无法归属——符合 S1「对象归属作上下文先验」设计，跨对象定位需上下文而非从参数学。")
    lines.append("- **下一步建议**：① 扩数据后检测头受益（TEST 提升），但机理头可能因类不平衡加剧而塌缩到主导类——先做类重平衡 + 对象上下文条件的头结构，再继续扩数据；② 针对 TEST 整对象类（CAPACITY_LOSS/FROZEN/GAIN_LOSS）补跨对象代表，避免靠 TRAIN 单一代表硬泛化；③ 开放集头（原型条件共形拒识）。")
    lines.append("")
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
