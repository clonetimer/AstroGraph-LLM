from __future__ import annotations

import hashlib

import numpy as np
import pytest

from astrograph.evaluation.data_exposure import dataset_use

from astrograph.data_driven import (
    DataDrivenModelRegistry,
    ModelInferenceRequest,
    TemporalLSTMAutoencoderConfig,
    TemporalLSTMAutoencoderProvider,
    fit_temporal_lstm_autoencoder,
    load_temporal_lstm_autoencoder,
    save_temporal_lstm_autoencoder,
)


@pytest.fixture(autouse=True)
def _synthetic_development_scope():
    with dataset_use("SYNTHETIC_PROTOCOL", "MODEL_TRAIN", "FIT"):
        yield


def _nominal_sequences() -> np.ndarray:
    rng = np.random.default_rng(20260725)
    time = np.linspace(0.0, 2.0 * np.pi, 16, dtype=np.float32)
    rows = []
    for _ in range(24):
        phase = float(rng.normal(0.0, 0.03))
        first = np.sin(time + phase) + rng.normal(0.0, 0.01, size=time.shape)
        second = np.cos(time + phase) + rng.normal(0.0, 0.01, size=time.shape)
        rows.append(np.stack([first, second], axis=1))
    return np.asarray(rows, dtype=np.float32)


def _config() -> TemporalLSTMAutoencoderConfig:
    return TemporalLSTMAutoencoderConfig(
        model_id="test-lstm-ae",
        model_version="0.1.0",
        channel_order=["sensor.a", "sensor.b"],
        fault_channel_weights={
            "FAULT_A": {"sensor.a": 2.0},
            "FAULT_B": {"sensor.b": 2.0},
        },
        hidden_size=6,
        latent_size=3,
        epochs=3,
        batch_size=8,
        learning_rate=0.01,
        threshold_quantile=0.9,
        random_seed=7,
    )


def _request(config: TemporalLSTMAutoencoderConfig, sequence: np.ndarray, episode: str) -> ModelInferenceRequest:
    digest = hashlib.sha256(sequence.tobytes()).hexdigest()
    return ModelInferenceRequest(
        episode_id=episode,
        model_id=config.model_id,
        model_version=config.model_version,
        feature_schema_id=config.feature_schema_id,
        input_data_sha256=digest,
        registered_fault_ids=config.supported_fault_ids,
        payload={"sequence": sequence.tolist()},
    )


def test_temporal_autoencoder_detects_large_channel_anomaly_and_roundtrips(tmp_path) -> None:
    nominal = _nominal_sequences()
    config = _config()
    model = fit_temporal_lstm_autoencoder(nominal, config)
    registry = DataDrivenModelRegistry.for_research()
    registry.register(TemporalLSTMAutoencoderProvider(model))

    nominal_prediction = registry.predict(_request(config, nominal[0], "episode-nominal"))
    anomaly = nominal[0].copy()
    anomaly[:, 0] += np.linspace(0.0, 8.0, anomaly.shape[0], dtype=np.float32)
    anomaly_prediction = registry.predict(_request(config, anomaly, "episode-anomaly"))

    assert (1.0 - anomaly_prediction.normal_probability) > (1.0 - nominal_prediction.normal_probability)
    assert anomaly_prediction.classes[0].class_id == "FAULT_A"
    assert anomaly_prediction.model_family == "PYTORCH"
    assert anomaly_prediction.metadata["raw_output_metadata"]["provider"] == "TemporalLSTMAutoencoderProvider"

    save_temporal_lstm_autoencoder(model, tmp_path)
    loaded = load_temporal_lstm_autoencoder(tmp_path)
    loaded_registry = DataDrivenModelRegistry.for_research()
    loaded_registry.register(TemporalLSTMAutoencoderProvider(loaded))
    loaded_prediction = loaded_registry.predict(_request(config, anomaly, "episode-loaded"))
    original_probs = {item.class_id: item.probability for item in anomaly_prediction.classes}
    loaded_probs = {item.class_id: item.probability for item in loaded_prediction.classes}
    assert loaded_probs == original_probs
