"""ALDM-inspired, detection-only research candidate.

This is an independent implementation of the architectural ideas in Liu et
al. (T-ASE 2026), not a vendored copy or a reproduction claim.  It is kept in
the lab namespace and deliberately has no Runtime/provider registration.

The model combines a causal-window context encoder, a positive-semidefinite
adaptive metric, soft contrastive targets, and a conditional RealNVP density
model.  Training consumes telemetry windows only; labels are evaluator-side.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ALDMDetectorConfig:
    window_size: int
    channel_count: int
    latent_size: int = 32
    context_size: int = 32
    hidden_size: int = 64
    flow_layers: int = 4
    contrastive_weight: float = 0.5
    contrastive_temperature: float = 0.2
    assignment_temperature: float = 1.0
    mask_fraction: float = 0.15

    def __post_init__(self) -> None:
        if self.window_size < 4 or self.channel_count < 1:
            raise ValueError("ALDM requires at least four timesteps and one channel")
        if self.latent_size < 4 or self.latent_size % 2:
            raise ValueError("latent_size must be an even integer of at least four")
        if self.context_size < 4 or self.hidden_size < 4 or self.flow_layers < 2:
            raise ValueError("context, hidden and flow dimensions are too small")
        if not 0.0 <= self.contrastive_weight <= 1.0:
            raise ValueError("contrastive_weight must lie in [0, 1]")
        if self.contrastive_temperature <= 0.0 or self.assignment_temperature <= 0.0:
            raise ValueError("contrastive temperatures must be positive")
        if not 0.0 < self.mask_fraction < 1.0:
            raise ValueError("mask_fraction must lie in (0, 1)")

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": "astrograph-aldm-inspired-detector-config.v1",
            **asdict(self),
            "task": "DETECTION_ONLY",
            "runtime_binding": "RESEARCH_ONLY_NOT_REGISTERED",
            "implementation_boundary": (
                "Independent ALDM-inspired implementation; not the authors' official code "
                "and not an exact reproduction claim."
            ),
        }


@dataclass(frozen=True)
class IQRScoreCalibration:
    q1: float
    q3: float
    raw_threshold: float
    scale: float
    fit_sample_count: int

    @classmethod
    def fit(cls, train_scores: Sequence[float]) -> "IQRScoreCalibration":
        from astrograph.evaluation.data_exposure import guard_development
        guard_development('CALIBRATE')
        values = np.asarray(train_scores, dtype=np.float64)
        if values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all():
            raise ValueError("calibration requires at least four finite Train scores")
        q1, q3 = np.quantile(values, (0.25, 0.75))
        spread = float(q3 - q1)
        scale = max(spread, float(values.std()) * 0.1, 1e-6)
        return cls(
            q1=float(q1),
            q3=float(q3),
            raw_threshold=float(q3 + 1.5 * spread),
            scale=scale,
            fit_sample_count=len(values),
        )

    def transform(self, scores: Sequence[float]) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("anomaly scores must be finite")
        logits = np.clip((values - self.raw_threshold) / self.scale, -60.0, 60.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float64)

    def receipt(self) -> dict[str, float | int | str]:
        return {
            "schema_version": "astrograph-aldm-iqr-score-calibration.v1",
            "q1": self.q1,
            "q3": self.q3,
            "raw_threshold": self.raw_threshold,
            "scale": self.scale,
            "fit_sample_count": self.fit_sample_count,
            "fit_scope": "TRAIN_ONLY",
            "normalized_iqr_threshold": 0.5,
        }


def build_aldm_detector(config: ALDMDetectorConfig):
    """Create the Torch network without introducing a package-level Torch import."""
    import torch
    from torch import nn
    import torch.nn.functional as functional

    class ResidualTemporalBlock(nn.Module):
        def __init__(self, dilation: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(
                    config.hidden_size,
                    config.hidden_size,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                ),
                nn.GroupNorm(4 if config.hidden_size % 4 == 0 else 1, config.hidden_size),
                nn.GELU(),
                nn.Conv1d(
                    config.hidden_size,
                    config.hidden_size,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                ),
                nn.GroupNorm(4 if config.hidden_size % 4 == 0 else 1, config.hidden_size),
                nn.GELU(),
            )

        def forward(self, values):
            return values + self.net(values)

    class ContextEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.temporal = nn.GRU(
                config.channel_count, config.context_size, batch_first=True
            )
            self.sensor_projection = nn.Linear(config.window_size, config.context_size)
            self.query = nn.Linear(config.context_size, config.context_size, bias=False)
            self.key = nn.Linear(config.context_size, config.context_size, bias=False)
            self.output = nn.Sequential(
                nn.Linear(config.context_size * 2, config.context_size),
                nn.GELU(),
                nn.LayerNorm(config.context_size),
            )

        def forward(self, values):
            _, temporal_state = self.temporal(values)
            sensors = self.sensor_projection(values.transpose(1, 2))
            affinity = torch.matmul(self.query(sensors), self.key(sensors).transpose(1, 2))
            adjacency = torch.softmax(affinity / math.sqrt(config.context_size), dim=-1)
            graph_state = torch.matmul(adjacency, sensors).mean(dim=1)
            context = self.output(torch.cat((temporal_state[-1], graph_state), dim=1))
            return context, adjacency

    class LatentEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Conv1d(config.channel_count, config.hidden_size, 1)
            self.blocks = nn.Sequential(
                ResidualTemporalBlock(1),
                ResidualTemporalBlock(2),
                ResidualTemporalBlock(4),
            )
            self.output = nn.Sequential(
                nn.Linear(config.hidden_size * 3, config.hidden_size),
                nn.GELU(),
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size, config.latent_size),
            )

        def forward(self, values):
            hidden = self.blocks(self.input(values.transpose(1, 2)))
            pooled = torch.cat(
                (hidden.mean(dim=-1), hidden.amax(dim=-1), hidden[:, :, -1]), dim=1
            )
            return self.output(pooled)

    class AdaptiveMetric(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.raw_lower = nn.Parameter(torch.eye(config.context_size))

        def factor(self):
            lower = torch.tril(self.raw_lower, diagonal=-1)
            diagonal = functional.softplus(torch.diagonal(self.raw_lower)) + 1e-4
            return lower + torch.diag(diagonal)

        def distances(self, values):
            projected = values @ self.factor().transpose(0, 1)
            return torch.cdist(projected, projected, p=2)

    class ConditionalCoupling(nn.Module):
        def __init__(self, parity: int) -> None:
            super().__init__()
            mask = torch.tensor(
                [(index + parity) % 2 for index in range(config.latent_size)],
                dtype=torch.float32,
            )
            self.register_buffer("mask", mask)
            self.network = nn.Sequential(
                nn.Linear(config.latent_size + config.context_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.latent_size * 2),
            )

        def forward(self, values, context):
            retained = values * self.mask
            shift, log_scale = self.network(torch.cat((retained, context), dim=1)).chunk(2, 1)
            active = 1.0 - self.mask
            log_scale = 1.5 * torch.tanh(log_scale) * active
            shift = shift * active
            transformed = retained + active * (values * torch.exp(log_scale) + shift)
            return transformed, log_scale.sum(dim=1)

    class ConditionalFlow(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                ConditionalCoupling(index % 2) for index in range(config.flow_layers)
            )

        def log_prob(self, values, context):
            transformed = values
            log_determinant = torch.zeros(len(values), device=values.device)
            for layer in self.layers:
                transformed, contribution = layer(transformed, context)
                log_determinant = log_determinant + contribution
            base = -0.5 * (
                transformed.square() + math.log(2.0 * math.pi)
            ).sum(dim=1)
            return base + log_determinant

    class ALDMInspiredDetector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.context_encoder = ContextEncoder()
            self.latent_encoder = LatentEncoder()
            self.metric = AdaptiveMetric()
            self.flow = ConditionalFlow()
            self.config = config

        def _masked_view(self, values):
            keep = torch.rand(
                values.shape[0], values.shape[1], 1, device=values.device
            ) >= config.mask_fraction
            return values * keep

        def soft_contrastive_loss(self, first, second, context):
            distance = self.metric.distances(context)
            targets = 2.0 * torch.sigmoid(-config.assignment_temperature * distance)
            targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-12)
            first = functional.normalize(first, dim=1)
            second = functional.normalize(second, dim=1)
            logits = first @ second.transpose(0, 1) / config.contrastive_temperature
            return -(targets * functional.log_softmax(logits, dim=1)).sum(dim=1).mean()

        def training_loss(self, values):
            context, _ = self.context_encoder(values)
            latent = self.latent_encoder(values)
            first = self.latent_encoder(self._masked_view(values))
            second = self.latent_encoder(self._masked_view(values))
            nll = -self.flow.log_prob(latent, context).mean() / config.latent_size
            contrastive = self.soft_contrastive_loss(first, second, context)
            total = (
                (1.0 - config.contrastive_weight) * nll
                + config.contrastive_weight * contrastive
            )
            return total, {
                "negative_log_likelihood": nll.detach(),
                "soft_contrastive": contrastive.detach(),
            }

        def anomaly_score(self, values):
            context, adjacency = self.context_encoder(values)
            latent = self.latent_encoder(values)
            score = -self.flow.log_prob(latent, context) / config.latent_size
            return score, adjacency

        def forward(self, values):
            return self.anomaly_score(values)[0]

    return ALDMInspiredDetector()


def fit_aldm_detector(
    train_windows: np.ndarray,
    config: ALDMDetectorConfig,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Fit from Train telemetry windows only; no label argument exists."""
    from astrograph.evaluation.data_exposure import guard_development
    guard_development('FIT')
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    values = np.asarray(train_windows, dtype=np.float32)
    expected = (config.window_size, config.channel_count)
    if values.ndim != 3 or tuple(values.shape[1:]) != expected:
        raise ValueError(f"train_windows must have shape [batch,{expected[0]},{expected[1]}]")
    if not np.isfinite(values).all() or len(values) < 2:
        raise ValueError("training windows must be finite and contain at least two samples")
    if epochs < 1 or batch_size < 2 or learning_rate <= 0.0:
        raise ValueError("invalid training hyperparameters")

    torch.manual_seed(seed)
    np.random.seed(seed)
    target = torch.device(device)
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")
        torch.cuda.set_device(target)
        torch.cuda.manual_seed_all(seed)
        torch.empty(0, device=target)
        torch.cuda.reset_peak_memory_stats(target)
    model = build_aldm_detector(config).to(target)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values)),
        batch_size=min(batch_size, len(values)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=target.type == "cuda",
        drop_last=len(values) > batch_size,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=epochs * len(loader),
        pct_start=0.2,
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "negative_log_likelihood": 0.0, "soft_contrastive": 0.0}
        seen = 0
        for (batch,) in loader:
            batch = batch.to(target, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = model.training_loss(batch)
            if not torch.isfinite(loss):
                raise RuntimeError("ALDM training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            count = len(batch)
            seen += count
            totals["loss"] += float(loss.detach()) * count
            for name in ("negative_log_likelihood", "soft_contrastive"):
                totals[name] += float(terms[name]) * count
        entry = {"epoch": epoch, **{name: value / seen for name, value in totals.items()}}
        history.append(entry)
        print(
            f"[aldm] epoch={epoch}/{epochs} loss={entry['loss']:.6f} "
            f"nll={entry['negative_log_likelihood']:.6f} "
            f"contrastive={entry['soft_contrastive']:.6f}",
            flush=True,
        )
    return model, {
        "schema_version": "astrograph-aldm-inspired-training.v1",
        "fit_scope": "TRAIN_WINDOWS_ONLY_LABEL_FREE",
        "sample_count": len(values),
        "epochs": epochs,
        "batch_size": min(batch_size, len(values)),
        "learning_rate": learning_rate,
        "seed": seed,
        "device": device,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(target)) if target.type == "cuda" else None
        ),
        "history": history,
        "claim_boundary": (
            "E0 research training receipt; no Test access, provider registration, "
            "diagnosis-performance or production claim."
        ),
    }


def score_aldm_detector(
    model: Any,
    windows: np.ndarray,
    *,
    device: str,
    batch_size: int = 2048,
) -> np.ndarray:
    import torch

    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("scoring windows must be a finite rank-three tensor")
    target = torch.device(device)
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(target)
            score, _ = model.anomaly_score(batch)
            parts.append(score.cpu().numpy())
    result = np.concatenate(parts).astype(np.float64)
    if not np.isfinite(result).all():
        raise RuntimeError("ALDM scoring produced non-finite values")
    return result


__all__ = [
    "ALDMDetectorConfig",
    "IQRScoreCalibration",
    "build_aldm_detector",
    "fit_aldm_detector",
    "score_aldm_detector",
]
