"""Nominal-data temporal autoencoder (engineering candidate).

STATUS (recorded 2026-08-04, ADR-004): DEVELOPMENT / NOT WIRED.  This module is
an offline engineering candidate and is NOT part of the S1 data-driven mainline
(diagnosis_baseline) nor reachable from unified/CLI.  It is kept for research
but must not be cited as a current capability.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from astrograph.evidence.executable import stable_id

from .contracts import (
    DataDrivenModelManifest,
    DataDrivenTaskType,
    ModelCapability,
    ModelInferenceRequest,
    ModelOutputSemantics,
    ModelRuntime,
    RawModelOutput,
)


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("Install AstroGraph with the 'deep-learning' optional dependency") from exc
    return torch, nn, (DataLoader, TensorDataset)


class TemporalLSTMAutoencoderConfig(BaseModel):
    """Configuration for a nominal-data LSTM reconstruction model.

    Fault attribution remains an explicit engineering mapping from telemetry channels to
    registered fault modes. The model detects temporal reconstruction anomalies; it does
    not autonomously invent causal relationships.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-temporal-lstm-autoencoder-config.v1"
    model_id: str = Field(default="temporal-lstm-autoencoder", min_length=1)
    model_version: str = Field(default="temporal-autoencoder-unvalidated-v1", min_length=1)
    display_name: str = Field(default="Temporal LSTM autoencoder engineering candidate", min_length=1)
    channel_order: list[str] = Field(min_length=1)
    fault_channel_weights: dict[str, dict[str, float]] = Field(min_length=1)
    hidden_size: int = Field(default=16, ge=2, le=1024)
    latent_size: int = Field(default=8, ge=2, le=1024)
    num_layers: int = Field(default=1, ge=1, le=8)
    epochs: int = Field(default=20, ge=1, le=10000)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    threshold_quantile: float = Field(default=0.995, gt=0.5, lt=1.0)
    random_seed: int = 20260725
    training_dataset_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("channel_order", "training_dataset_ids")
    @classmethod
    def unique_list_values(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("configuration lists must contain unique values")
        return value

    @model_validator(mode="after")
    def validate_fault_channel_weights(self) -> "TemporalLSTMAutoencoderConfig":
        channels = set(self.channel_order)
        for fault_id, weights in self.fault_channel_weights.items():
            if not fault_id:
                raise ValueError("fault IDs must not be empty")
            if not weights:
                raise ValueError(f"fault {fault_id} must map to at least one telemetry channel")
            unknown = set(weights) - channels
            if unknown:
                raise ValueError(f"fault {fault_id} references unknown channels: {sorted(unknown)}")
            if any((not math.isfinite(float(weight))) or float(weight) <= 0.0 for weight in weights.values()):
                raise ValueError(f"fault {fault_id} channel weights must be finite and positive")
        return self

    @property
    def supported_fault_ids(self) -> list[str]:
        return sorted(self.fault_channel_weights)

    @property
    def feature_schema_id(self) -> str:
        return stable_id(
            "FEATURE-SCHEMA",
            {
                "kind": "TEMPORAL_SEQUENCE",
                "channel_order": self.channel_order,
                "normalization": "TRAINING_MEAN_STD",
            },
        )


class TemporalAutoencoderTrainingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-temporal-autoencoder-training-summary.v1"
    sample_count: int = Field(ge=1)
    sequence_length: int = Field(ge=2)
    channel_count: int = Field(ge=1)
    epochs: int = Field(ge=1)
    final_training_loss: float = Field(ge=0.0)
    threshold_mse: float = Field(gt=0.0)
    threshold_quantile: float = Field(gt=0.5, lt=1.0)
    training_error_mean: float = Field(ge=0.0)
    training_error_max: float = Field(ge=0.0)
    channel_threshold_mse: dict[str, float] = Field(default_factory=dict)
    claim_boundary: str = (
        "Training summary is an engineering artifact. Independent-run benchmark and release gates are required "
        "before any fault-model binding can be enabled."
    )


@dataclass
class TemporalLSTMAutoencoderModel:
    config: TemporalLSTMAutoencoderConfig
    network: Any
    channel_mean: np.ndarray
    channel_scale: np.ndarray
    threshold_mse: float
    channel_threshold_mse: np.ndarray
    training_summary: TemporalAutoencoderTrainingSummary
    manifest: DataDrivenModelManifest


class _NetworkFactory:
    @staticmethod
    def build(config: TemporalLSTMAutoencoderConfig) -> Any:
        torch, nn, _ = _require_torch()

        class LSTMAutoencoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                channels = len(config.channel_order)
                self.encoder = nn.LSTM(
                    input_size=channels,
                    hidden_size=config.hidden_size,
                    num_layers=config.num_layers,
                    batch_first=True,
                )
                self.to_latent = nn.Linear(config.hidden_size, config.latent_size)
                self.to_hidden = nn.Linear(
                    config.latent_size,
                    config.hidden_size * config.num_layers,
                )
                self.to_cell = nn.Linear(
                    config.latent_size,
                    config.hidden_size * config.num_layers,
                )
                self.decoder = nn.LSTM(
                    input_size=channels,
                    hidden_size=config.hidden_size,
                    num_layers=config.num_layers,
                    batch_first=True,
                )
                self.output = nn.Linear(config.hidden_size, channels)

            def forward(self, values: Any) -> Any:
                _, (hidden, _) = self.encoder(values)
                latent = self.to_latent(hidden[-1])
                batch = values.shape[0]
                hidden0 = self.to_hidden(latent).reshape(
                    batch,
                    config.num_layers,
                    config.hidden_size,
                ).permute(1, 0, 2).contiguous()
                cell0 = self.to_cell(latent).reshape(
                    batch,
                    config.num_layers,
                    config.hidden_size,
                ).permute(1, 0, 2).contiguous()
                decoder_input = torch.zeros_like(values)
                decoded, _ = self.decoder(decoder_input, (hidden0, cell0))
                return self.output(decoded)

        return LSTMAutoencoder()


def _as_sequence_batch(values: Any, channel_count: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, :, :]
    if array.ndim != 3:
        raise ValueError("temporal sequence must have shape [time,channel] or [batch,time,channel]")
    if array.shape[1] < 2:
        raise ValueError("temporal sequence requires at least two time samples")
    if array.shape[2] != channel_count:
        raise ValueError(
            f"temporal sequence channel count mismatch: expected {channel_count}, got {array.shape[2]}"
        )
    if not np.isfinite(array).all():
        raise ValueError("temporal sequence contains non-finite values")
    return array


def _normalise(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values - mean.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)


def _manifest(config: TemporalLSTMAutoencoderConfig) -> DataDrivenModelManifest:
    return DataDrivenModelManifest(
        model_id=config.model_id,
        model_version=config.model_version,
        display_name=config.display_name,
        task_type=DataDrivenTaskType.ANOMALY_ATTRIBUTION,
        runtime=ModelRuntime.PYTORCH,
        output_semantics=ModelOutputSemantics.ANOMALY_ATTRIBUTION,
        capabilities=[
            ModelCapability.FAULT_RANKING,
            ModelCapability.NORMAL_CLASS,
            ModelCapability.TEMPORAL_CONTEXT,
        ],
        supported_fault_ids=config.supported_fault_ids,
        normal_class_id="NORMAL",
        feature_schema_ids=[config.feature_schema_id],
        input_contract={
            "payload.sequence": "float[time,channel]",
            "channel_order": config.channel_order,
            "training_regime": "nominal-only reconstruction",
            "fault_attribution": "configured weighted channel reconstruction error",
        },
        training_dataset_ids=config.training_dataset_ids,
        claim_boundary=(
            "Unvalidated LSTM reconstruction candidate. Independent-run benchmark, fault attribution review, "
            "open-set policy, graph evidence, and diagnosis release gates remain mandatory."
        ),
        metadata={
            **config.metadata,
            "architecture": "LSTM_AUTOENCODER",
            "template_only": False,
            "online_registration_allowed": False,
            "fault_channel_weights": config.fault_channel_weights,
        },
    )


def fit_temporal_lstm_autoencoder(
    nominal_sequences: Any,
    config: TemporalLSTMAutoencoderConfig,
) -> TemporalLSTMAutoencoderModel:
    """Train an engineering candidate on nominal sequences only.

    This function intentionally does not create or enable a fault-model binding. That
    decision belongs to the independent benchmark comparison and expert-review path.
    """
    from astrograph.evaluation.data_exposure import guard_development
    guard_development('FIT')

    torch, nn, loader_types = _require_torch()
    DataLoader, TensorDataset = loader_types
    sequences = _as_sequence_batch(nominal_sequences, len(config.channel_order))
    mean = sequences.mean(axis=(0, 1)).astype(np.float32)
    scale = sequences.std(axis=(0, 1)).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    normalised = _normalise(sequences, mean, scale).astype(np.float32)

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    network = _NetworkFactory.build(config)
    network.train()
    optimiser = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    tensor = torch.from_numpy(normalised)
    generator = torch.Generator().manual_seed(config.random_seed)
    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=min(config.batch_size, len(tensor)),
        shuffle=True,
        generator=generator,
    )
    final_loss = 0.0
    for _ in range(config.epochs):
        running = 0.0
        count = 0
        for (batch,) in loader:
            optimiser.zero_grad(set_to_none=True)
            reconstruction = network(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimiser.step()
            running += float(loss.detach()) * len(batch)
            count += len(batch)
        final_loss = running / max(1, count)

    network.eval()
    with torch.no_grad():
        reconstruction = network(tensor)
        squared = (reconstruction - tensor) ** 2
        per_sequence = torch.mean(squared, dim=(1, 2)).cpu().numpy()
        per_sequence_channel = torch.mean(squared, dim=1).cpu().numpy()
    threshold = float(np.quantile(per_sequence, config.threshold_quantile))
    threshold = max(threshold, float(np.finfo(np.float32).eps))
    channel_threshold = np.quantile(
        per_sequence_channel,
        config.threshold_quantile,
        axis=0,
    ).astype(np.float32)
    channel_threshold = np.maximum(channel_threshold, np.finfo(np.float32).eps)
    summary = TemporalAutoencoderTrainingSummary(
        sample_count=int(sequences.shape[0]),
        sequence_length=int(sequences.shape[1]),
        channel_count=int(sequences.shape[2]),
        epochs=config.epochs,
        final_training_loss=float(final_loss),
        threshold_mse=threshold,
        threshold_quantile=config.threshold_quantile,
        training_error_mean=float(np.mean(per_sequence)),
        training_error_max=float(np.max(per_sequence)),
        channel_threshold_mse={
            channel: float(channel_threshold[index])
            for index, channel in enumerate(config.channel_order)
        },
    )
    return TemporalLSTMAutoencoderModel(
        config=config,
        network=network,
        channel_mean=mean,
        channel_scale=scale,
        threshold_mse=threshold,
        channel_threshold_mse=channel_threshold,
        training_summary=summary,
        manifest=_manifest(config),
    )


@dataclass
class TemporalLSTMAutoencoderProvider:
    model: TemporalLSTMAutoencoderModel

    @property
    def manifest(self) -> DataDrivenModelManifest:
        return self.model.manifest

    def predict_raw(self, request: ModelInferenceRequest) -> RawModelOutput:
        torch, _, _ = _require_torch()
        sequence = request.payload.get("sequence")
        if sequence is None:
            raise ValueError("temporal autoencoder expects payload.sequence")
        values = _as_sequence_batch(sequence, len(self.model.config.channel_order))
        if values.shape[0] != 1:
            raise ValueError("online inference accepts exactly one temporal sequence")
        normalised = _normalise(values, self.model.channel_mean, self.model.channel_scale).astype(np.float32)
        tensor = torch.from_numpy(normalised)
        self.model.network.eval()
        with torch.no_grad():
            reconstruction = self.model.network(tensor)
            squared_error = ((reconstruction - tensor) ** 2).cpu().numpy()[0]
        reconstruction_mse = float(np.mean(squared_error))
        channel_mse = np.mean(squared_error, axis=0)
        channel_exceedance = np.maximum(
            channel_mse / self.model.channel_threshold_mse - 1.0,
            0.0,
        )
        exceedance = max(0.0, reconstruction_mse / self.model.threshold_mse - 1.0)
        anomaly_score = float(1.0 - math.exp(-exceedance)) if exceedance > 0.0 else 0.0
        anomaly_score = min(max(anomaly_score, 0.0), 1.0)

        attribution: dict[str, float] = {}
        if anomaly_score > 0.0:
            index = {channel: idx for idx, channel in enumerate(self.model.config.channel_order)}
            for fault_id, weights in self.model.config.fault_channel_weights.items():
                attribution[fault_id] = float(
                    sum(
                        float(weight) * float(channel_exceedance[index[channel]])
                        for channel, weight in weights.items()
                    )
                )
            if not any(value > 0.0 for value in attribution.values()):
                for fault_id, weights in self.model.config.fault_channel_weights.items():
                    attribution[fault_id] = float(
                        sum(
                            float(weight) * float(channel_mse[index[channel]])
                            for channel, weight in weights.items()
                        )
                    )
            if not any(value > 0.0 for value in attribution.values()):
                attribution = {fault_id: 1.0 for fault_id in self.model.config.supported_fault_ids}

        return RawModelOutput(
            episode_id=request.episode_id,
            model_id=self.model.config.model_id,
            model_version=self.model.config.model_version,
            output_semantics=ModelOutputSemantics.ANOMALY_ATTRIBUTION,
            anomaly_score=anomaly_score,
            fault_attribution=attribution,
            metadata={
                "provider": "TemporalLSTMAutoencoderProvider",
                "reconstruction_mse": reconstruction_mse,
                "threshold_mse": self.model.threshold_mse,
                "channel_mse": {
                    channel: float(channel_mse[idx])
                    for idx, channel in enumerate(self.model.config.channel_order)
                },
                "channel_exceedance": {
                    channel: float(channel_exceedance[idx])
                    for idx, channel in enumerate(self.model.config.channel_order)
                },
                "claim_boundary": self.model.manifest.claim_boundary,
            },
        )


def save_temporal_lstm_autoencoder(
    model: TemporalLSTMAutoencoderModel,
    output_dir: str | Path,
) -> dict[str, str]:
    torch, _, _ = _require_torch()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "config": root / "config.json",
        "manifest": root / "manifest.json",
        "training_summary": root / "training_summary.json",
        "preprocessing": root / "preprocessing.npz",
        "weights": root / "weights.pt",
    }
    files["config"].write_text(
        json.dumps(model.config.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    files["manifest"].write_text(
        json.dumps(model.manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    files["training_summary"].write_text(
        json.dumps(model.training_summary.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez(
        files["preprocessing"],
        channel_mean=model.channel_mean,
        channel_scale=model.channel_scale,
        threshold_mse=np.asarray([model.threshold_mse], dtype=np.float64),
        channel_threshold_mse=model.channel_threshold_mse,
    )
    torch.save(model.network.state_dict(), files["weights"])
    result: dict[str, str] = {}
    for name, path in files.items():
        result[name] = str(path)
        result[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def load_temporal_lstm_autoencoder(model_dir: str | Path) -> TemporalLSTMAutoencoderModel:
    torch, _, _ = _require_torch()
    root = Path(model_dir)
    config = TemporalLSTMAutoencoderConfig.model_validate(
        json.loads((root / "config.json").read_text(encoding="utf-8"))
    )
    manifest = DataDrivenModelManifest.model_validate(
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    )
    summary = TemporalAutoencoderTrainingSummary.model_validate(
        json.loads((root / "training_summary.json").read_text(encoding="utf-8"))
    )
    preprocessing = np.load(root / "preprocessing.npz", allow_pickle=False)
    network = _NetworkFactory.build(config)
    state = torch.load(root / "weights.pt", map_location="cpu", weights_only=True)
    network.load_state_dict(state)
    network.eval()
    return TemporalLSTMAutoencoderModel(
        config=config,
        network=network,
        channel_mean=np.asarray(preprocessing["channel_mean"], dtype=np.float32),
        channel_scale=np.asarray(preprocessing["channel_scale"], dtype=np.float32),
        threshold_mse=float(np.asarray(preprocessing["threshold_mse"])[0]),
        channel_threshold_mse=np.asarray(preprocessing["channel_threshold_mse"], dtype=np.float32),
        training_summary=summary,
        manifest=manifest,
    )
