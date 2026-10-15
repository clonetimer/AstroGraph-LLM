from __future__ import annotations

import numpy as np
import pytest

from astrograph.data_driven.preprocessing import fit_channel_preprocessor
from astrograph.evaluation.data_exposure import dataset_use


@pytest.fixture(autouse=True)
def _synthetic_development_scope():
    with dataset_use("SYNTHETIC_PROTOCOL", "MODEL_TRAIN", "FIT"):
        yield


def test_standard_preprocessor_matches_train_mean_and_std() -> None:
    train = np.asarray([[[1.0, 0.0], [3.0, 1.0]], [[5.0, 0.0], [7.0, 1.0]]])
    state = fit_channel_preprocessor(train, method="standard")
    transformed = state.transform(train)

    assert transformed.reshape(-1, 2).mean(0) == pytest.approx([0.0, 0.0])
    assert transformed.reshape(-1, 2).std(0) == pytest.approx([1.0, 1.0])
    assert state.receipt()["fit_scope"] == "train_only"


def test_robust_preprocessor_clips_unseen_outlier_with_train_bounds() -> None:
    train = np.arange(100, dtype=float).reshape(2, 50, 1)
    state = fit_channel_preprocessor(train, method="robust_clip", clip_quantiles=(0.1, 0.9))
    ordinary = state.transform(np.asarray([[[90.0]]]))[0, 0, 0]
    unseen_outlier = state.transform(np.asarray([[[1_000_000.0]]]))[0, 0, 0]

    assert unseen_outlier == pytest.approx(ordinary)
    assert state.upper[0] == pytest.approx(89.1)


def test_robust_preprocessor_caps_degenerate_normalized_magnitude() -> None:
    train = np.asarray([[[0.0], [0.0], [0.0], [1e-8], [1.0]]])
    state = fit_channel_preprocessor(train, method="robust_clip", clip_quantiles=(0.0, 1.0))

    assert state.transform(np.asarray([[[1.0]]]))[0, 0, 0] == 10.0
    assert state.receipt()["normalized_clip"] == 10.0


def test_robust_preprocessor_handles_binary_constant_and_missing_channels() -> None:
    train = np.asarray([[[0.0, np.nan, 3.0], [1.0, np.inf, 3.0]]])
    state = fit_channel_preprocessor(
        train, method="robust_clip", fallback_impute=np.asarray([0.0, 7.0, 3.0])
    )
    result = state.transform(np.asarray([[[1.0, np.nan, 3.0]]]))

    assert np.isfinite(result).all()
    assert state.impute[1] == 7.0
    assert state.scale[2] == 1.0


def test_preprocessor_rejects_channel_or_method_contract_drift() -> None:
    train = np.ones((2, 3, 2), dtype=float)
    with pytest.raises(ValueError, match="unsupported"):
        fit_channel_preprocessor(train, method="unknown")
    state = fit_channel_preprocessor(train)
    with pytest.raises(ValueError, match="channel count"):
        state.transform(np.ones((1, 3, 3)))
