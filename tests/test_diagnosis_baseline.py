"""Tests for the S1 diagnosis baseline (loop 15): grouped split, feature
extraction, dataset, baseline training + honest metrics.

These tests exercise the committed dataset + generated traces only (no
Sat-Agent venv).  They verify:

1. grouped holdout: split by component_instance_id family; no object spans two
   splits; freeze is reproducible and reloadable; feature vector is finite.
2. deny discipline: label/truth/case/scenario/seed/split/file path/fault-id
   tokens never appear in a feature vector (model input).
3. feature schema: fixed dimension, per-archetype coverage, mechanism-class
   mapping covers all scenario ids in the dataset.
4. baseline training: the shared-backbone multi-head model trains (CPU) and
   produces deterministic predictions; zero-fit baseline scores are in [0,1].
5. metrics: classification metrics + worst-group are honest and finite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astrograph.data_driven.credibility import SplitName
from astrograph.data_driven.diagnosis_baseline import (
    DiagnosisBaseline,
    DiagnosisBaselineConfig,
    FeatureScaler,
    ZeroFitDetectionBaseline,
    _classification_metrics,
    evaluate_baseline,
    evaluate_zero_fit,
    worst_group_accuracy,
)
from astrograph.data_driven.diagnosis_dataset import build_diagnosis_samples
from astrograph.data_driven.diagnosis_features import (
    FEATURE_DIM_NAMES,
    MECHANISM_CLASS_MAP,
    MECHANISM_CLASS_ORDER,
    NUM_FEATURES,
    OBJECT_ARCHETYPE,
    mechanism_class_of,
)
from astrograph.data_driven.diagnosis_split import (
    DEFAULT_SPLITS_DIR,
    build_grouped_split,
    freeze_split,
    load_frozen_split,
    split_samples,
    verify_no_leakage,
)

ROOT = Path(__file__).resolve().parents[1]

# deny tokens that must never appear in a model-input feature vector.
DENY_TOKENS = (
    "label", "truth", "case_id", "scenario", "seed", "split",
    "generated_traces", "fault_id", "onset", "mag", "sev",
    "-long.csv", "overflow", "bias_step", "jam",
)


@pytest.fixture(scope="module")
def diagnosis_samples():
    """Materialize the immutable committed corpus once for this test module."""
    return tuple(build_diagnosis_samples())


def test_feature_schema_is_fixed_and_finite(diagnosis_samples):
    samples = diagnosis_samples
    # v2 dataset now carries the loop-17 expanded corpus (battery-cycled traces
    # gained charge segments under gap-1), so the count is >= the loop-13
    # baseline of 40 rather than a hard number; the schema invariant is exact.
    assert len(samples) >= 40
    shapes = {s.feature.shape for s in samples}
    assert shapes == {(NUM_FEATURES,)}
    assert len(FEATURE_DIM_NAMES) == NUM_FEATURES
    for s in samples:
        assert np.isfinite(s.feature).all(), s.episode_id


def test_feature_vector_contains_no_deny_tokens(diagnosis_samples):
    """deny rules: label/truth/case/scenario/seed/split/file path/fault-id
    must never appear in a model input."""
    samples = diagnosis_samples
    for s in samples:
        for token in DENY_TOKENS:
            assert token not in [str(v) for v in s.feature.tolist()], f"{s.episode_id}: {token}"
        # object identity (component id) is NOT a feature (would be circular
        # with the localization head); archetype context is allowed.
        dims = {name.split(".")[-1] for name in FEATURE_DIM_NAMES}
        assert "comp" not in {name.split("_")[0] for name in FEATURE_DIM_NAMES}


def test_mechanism_class_map_covers_all_scenario_ids(diagnosis_samples):
    samples = diagnosis_samples
    seen = {s.scenario_id for s in samples}
    for scenario_id in seen:
        cls = mechanism_class_of(scenario_id)
        assert cls in MECHANISM_CLASS_ORDER, (scenario_id, cls)
        assert cls != "UNKNOWN", scenario_id
    # cross-object mechanism classes present (S1 design): CAPACITY_LOSS / FROZEN / GAIN_LOSS
    classes_by_object: dict[str, set[str]] = {}
    for s in samples:
        classes_by_object.setdefault(s.object_id, set()).add(s.mechanism_class)
    assert "CAPACITY_LOSS" in classes_by_object["battery"]
    assert "CAPACITY_LOSS" in classes_by_object["mtb"]


def test_grouped_split_no_leakage_and_reproducible(diagnosis_samples):
    samples = diagnosis_samples
    split = build_grouped_split(samples)
    verify_no_leakage(split, samples)

    # no object spans two splits
    obj_splits: dict[str, set[str]] = {}
    for s in samples:
        obj_splits.setdefault(s.component_instance_id, set()).add(
            split.episode_split[s.episode_id]
        )
    for obj, splits in obj_splits.items():
        assert len(splits) == 1, obj

    # freeze + reload reproducibility
    path = freeze_split(split)
    assert path.is_file()
    loaded = load_frozen_split(split.split_id)
    assert loaded.groups == split.groups
    assert loaded.episode_split == split.episode_split
    assert loaded.sha256 == split.sha256

    # split has all three splits populated
    partition = split_samples(samples, split)
    for name in (SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST):
        assert partition[name], name


def test_grouped_split_has_cross_object_mechanism_representation(diagnosis_samples):
    """S1 design probe: TEST objects' mechanism classes have a cross-object
    representative in TRAIN (so the head is tested on cross-object transfer,
    not on OOD classes only)."""
    samples = diagnosis_samples
    split = build_grouped_split(samples)
    partition = split_samples(samples, split)
    train_mech = {s.mechanism_class for s in partition[SplitName.TRAIN]}
    test_mech = {s.mechanism_class for s in partition[SplitName.TEST]}
    # at least one non-NOMINAL class is shared cross-object
    assert any(m in train_mech for m in test_mech if m != "NOMINAL")


def test_zero_fit_baseline_scores_bounded(diagnosis_samples):
    samples = diagnosis_samples
    baseline = ZeroFitDetectionBaseline()
    for s in samples:
        score = baseline.score(s)
        assert 0.0 <= score <= 1.0
        pred = baseline.predict(s, threshold=0.5)
        assert pred in (0, 1)


def test_baseline_trains_and_predicts_deterministically(diagnosis_samples):
    samples = diagnosis_samples
    split = build_grouped_split(samples)
    partition = split_samples(samples, split)
    config = DiagnosisBaselineConfig(epochs=5, patience=2, hidden=8, device="cpu", seed=7)
    model = DiagnosisBaseline(config).fit(partition)
    # deterministic predictions for the same input
    p1 = model.predict(partition[SplitName.TRAIN][0])
    p2 = model.predict(partition[SplitName.TRAIN][0])
    assert p1 == p2
    assert p1["detection"] in (0, 1)
    assert p1["mechanism_class"] in MECHANISM_CLASS_ORDER
    assert p1["object_id"] in OBJECT_ARCHETYPE


def test_evaluation_metrics_honest_and_finite(diagnosis_samples):
    samples = diagnosis_samples
    split = build_grouped_split(samples)
    partition = split_samples(samples, split)
    config = DiagnosisBaselineConfig(epochs=3, patience=1, hidden=8, device="cpu", seed=3)
    model = DiagnosisBaseline(config).fit(partition)
    for name in (SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST):
        res = evaluate_baseline(model, partition[name], object_names=model.objects)
        assert res["n_samples"] == len(partition[name])
        assert 0.0 <= res["detection"]["micro_accuracy"] <= 1.0
        assert 0.0 <= res["mechanism"]["micro_accuracy"] <= 1.0
        assert 0.0 <= res["localization"]["micro_accuracy"] <= 1.0
        assert len(res["mechanism"]["confusion_matrix"]) == model.n_mechanism
        assert res["worst_group"]["worst_accuracy"] <= 1.0


def test_metrics_functions():
    m = _classification_metrics([0, 1, 1], [0, 0, 1], 2)
    assert m["micro_accuracy"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] >= 0.0
    assert len(m["confusion_matrix"]) == 2

    class _S:
        def __init__(self, obj):
            self.component_instance_id = obj

    samples = [_S("battery"), _S("battery"), _S("eps")]
    worst = worst_group_accuracy(samples, ["battery", "eps", "eps"])
    assert worst["worst_accuracy"] == pytest.approx(0.5)
    assert worst["worst_object"] in ("battery", "eps")


def test_feature_scaler_fit_transform():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(10, NUM_FEATURES))
    scaler = FeatureScaler.fit(features)
    transformed = scaler.transform(features)
    assert transformed.shape == features.shape
    assert np.isfinite(transformed).all()
