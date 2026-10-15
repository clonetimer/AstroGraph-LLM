from __future__ import annotations

import numpy as np
import pytest

from astrograph.evaluation.data_exposure import dataset_use

from astrograph.data_driven.lab.aldm_detector import (
    ALDMDetectorConfig,
    IQRScoreCalibration,
    build_aldm_detector,
    fit_aldm_detector,
    score_aldm_detector,
)


@pytest.fixture(autouse=True)
def _synthetic_development_scope():
    with dataset_use("SYNTHETIC_PROTOCOL", "MODEL_TRAIN", "FIT"):
        yield


def _config() -> ALDMDetectorConfig:
    return ALDMDetectorConfig(
        window_size=8,
        channel_count=4,
        latent_size=8,
        context_size=8,
        hidden_size=16,
        flow_layers=2,
    )


def test_aldm_contract_is_detection_only_and_research_only() -> None:
    receipt = _config().receipt()
    assert receipt["task"] == "DETECTION_ONLY"
    assert receipt["runtime_binding"] == "RESEARCH_ONLY_NOT_REGISTERED"
    with pytest.raises(ValueError, match="even"):
        ALDMDetectorConfig(window_size=8, channel_count=2, latent_size=7)


def test_aldm_loss_and_scores_are_finite() -> None:
    import torch

    torch.manual_seed(9)
    model = build_aldm_detector(_config())
    values = torch.randn(6, 8, 4)
    loss, terms = model.training_loss(values)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(terms) == {"negative_log_likelihood", "soft_contrastive"}
    assert any(parameter.grad is not None for parameter in model.parameters())
    scores, adjacency = model.anomaly_score(values)
    assert scores.shape == (6,)
    assert adjacency.shape == (6, 4, 4)
    assert torch.allclose(adjacency.sum(-1), torch.ones(6, 4), atol=1e-5)


def test_train_only_iqr_calibration_maps_paper_threshold_to_half() -> None:
    calibration = IQRScoreCalibration.fit([0.0, 1.0, 2.0, 3.0, 4.0])
    transformed = calibration.transform(
        [calibration.raw_threshold - calibration.scale, calibration.raw_threshold]
    )
    assert transformed[0] < 0.5
    assert transformed[1] == pytest.approx(0.5)
    assert calibration.receipt()["fit_scope"] == "TRAIN_ONLY"


def test_small_cpu_fit_has_no_label_input_and_scores_roundtrip() -> None:
    rng = np.random.default_rng(12)
    windows = rng.normal(size=(12, 8, 4)).astype(np.float32)
    model, receipt = fit_aldm_detector(
        windows,
        _config(),
        epochs=1,
        batch_size=6,
        learning_rate=1e-3,
        device="cpu",
        seed=12,
    )
    scores = score_aldm_detector(model, windows[:5], device="cpu", batch_size=2)
    assert scores.shape == (5,)
    assert np.isfinite(scores).all()
    assert receipt["fit_scope"] == "TRAIN_WINDOWS_ONLY_LABEL_FREE"
