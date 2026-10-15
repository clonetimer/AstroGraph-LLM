"""Strong time-series classification backbones for governed diagnosis comparisons.

The implementations preserve the defining mechanism of InceptionTime,
TimesNet and ModernTCN while exposing the same three heads and input contract.
They are local, dependency-free adaptations for AstroGraph's short causal
windows; they are not represented as byte-identical reproductions of the
authors' repositories.
"""

from __future__ import annotations

import math
from typing import Any


SOTA_MODEL_FAMILIES = ("inception_time", "timesnet", "modern_tcn")


def make_sota_multitask_model(
    family: str,
    channels: int,
    components: int,
    classes: int,
    *,
    window_size: int,
) -> Any:
    """Return a strong time-series backbone with common diagnosis heads."""
    import torch
    from torch import nn

    if family not in SOTA_MODEL_FAMILIES:
        raise ValueError(f"unknown SOTA model family: {family}")
    if min(channels, components, classes, window_size) < 1:
        raise ValueError("SOTA model dimensions must be positive")

    class InceptionModule(nn.Module):
        def __init__(self, width_in: int, branch_width: int = 32) -> None:
            super().__init__()
            bottleneck = min(32, width_in)
            self.reduce = nn.Conv1d(width_in, bottleneck, 1, bias=False)
            self.temporal = nn.ModuleList(
                nn.Conv1d(bottleneck, branch_width, kernel, padding=kernel // 2, bias=False)
                for kernel in (9, 19, 39)
            )
            self.pool_branch = nn.Sequential(
                nn.MaxPool1d(3, stride=1, padding=1),
                nn.Conv1d(width_in, branch_width, 1, bias=False),
            )
            self.norm = nn.BatchNorm1d(branch_width * 4)
            self.activation = nn.GELU()

        def forward(self, value):
            reduced = self.reduce(value)
            merged = torch.cat(
                [*(layer(reduced) for layer in self.temporal), self.pool_branch(value)], dim=1
            )
            return self.activation(self.norm(merged))

    class InceptionTimeBackbone(nn.Module):
        output_width = 128

        def __init__(self) -> None:
            super().__init__()
            modules = []
            projections = []
            width = channels
            for index in range(6):
                modules.append(InceptionModule(width))
                if index in {2, 5}:
                    projections.append(nn.Sequential(
                        nn.Conv1d(channels if index == 2 else 128, 128, 1, bias=False),
                        nn.BatchNorm1d(128),
                    ))
                width = 128
            self.modules_ = nn.ModuleList(modules)
            self.projections = nn.ModuleList(projections)
            self.activation = nn.GELU()

        def forward(self, value):
            state = value.transpose(1, 2)
            residual = state
            projection_index = 0
            for index, layer in enumerate(self.modules_):
                state = layer(state)
                if index in {2, 5}:
                    state = self.activation(state + self.projections[projection_index](residual))
                    residual = state
                    projection_index += 1
            return state.mean(-1)

    class TimesInception(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            hidden = width * 2
            self.expand = nn.ModuleList(
                nn.Conv2d(width, hidden, kernel, padding=kernel // 2)
                for kernel in (1, 3, 5)
            )
            self.project = nn.ModuleList(
                nn.Conv2d(hidden, width, kernel, padding=kernel // 2)
                for kernel in (1, 3, 5)
            )
            self.activation = nn.GELU()

        def forward(self, value):
            expanded = sum(layer(value) for layer in self.expand) / len(self.expand)
            expanded = self.activation(expanded)
            return sum(layer(expanded) for layer in self.project) / len(self.project)

    class TimesBlock(nn.Module):
        def __init__(self, width: int, top_k: int = 3) -> None:
            super().__init__()
            self.top_k = top_k
            self.inception = TimesInception(width)
            self.norm = nn.LayerNorm(width)

        def forward(self, value):
            batch, length, width = value.shape
            spectrum = torch.fft.rfft(value, dim=1)
            amplitude = spectrum.abs().mean(dim=(0, 2))
            candidates = amplitude[1:]
            count = min(self.top_k, int(candidates.numel()))
            if count == 0:
                return self.norm(value)
            frequency = torch.topk(candidates, count).indices + 1
            sample_amplitude = spectrum.abs().mean(dim=2)[:, frequency]
            outputs = []
            for item in frequency.tolist():
                period = max(1, math.ceil(length / int(item)))
                padded_length = math.ceil(length / period) * period
                padded = value
                if padded_length > length:
                    padded = torch.cat([
                        value,
                        value.new_zeros(batch, padded_length - length, width),
                    ], dim=1)
                image = padded.reshape(batch, padded_length // period, period, width)
                image = image.permute(0, 3, 1, 2).contiguous()
                restored = self.inception(image).permute(0, 2, 3, 1)
                outputs.append(restored.reshape(batch, padded_length, width)[:, :length])
            weights_ = torch.softmax(sample_amplitude, dim=1)
            combined = sum(
                item * weights_[:, index].reshape(batch, 1, 1)
                for index, item in enumerate(outputs)
            )
            return self.norm(value + combined)

    class TimesNetBackbone(nn.Module):
        output_width = 64

        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Linear(channels, 64)
            self.position = nn.Parameter(torch.zeros(1, window_size, 64))
            self.blocks = nn.ModuleList([TimesBlock(64), TimesBlock(64)])
            self.norm = nn.LayerNorm(64)

        def forward(self, value):
            state = self.embedding(value) + self.position[:, : value.shape[1]]
            for block in self.blocks:
                state = block(state)
            state = self.norm(state)
            return 0.5 * (state.mean(1) + state[:, -1])

    class ModernTCNBlock(nn.Module):
        def __init__(self, width: int, kernel: int) -> None:
            super().__init__()
            self.depthwise = nn.Conv1d(
                width, width, kernel, padding=kernel // 2, groups=width, bias=False
            )
            self.norm = nn.BatchNorm1d(width)
            self.channel_mixer = nn.Sequential(
                nn.Conv1d(width, width * 4, 1), nn.GELU(), nn.Dropout(0.1),
                nn.Conv1d(width * 4, width, 1),
            )

        def forward(self, value):
            temporal = value + self.norm(self.depthwise(value))
            return temporal + self.channel_mixer(temporal)

    class ModernTCNBackbone(nn.Module):
        output_width = 96

        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Conv1d(channels, 96, 3, padding=1)
            self.blocks = nn.Sequential(
                ModernTCNBlock(96, 7), ModernTCNBlock(96, 13),
                ModernTCNBlock(96, 19), ModernTCNBlock(96, 25),
            )
            self.norm = nn.LayerNorm(96)

        def forward(self, value):
            state = self.blocks(self.stem(value.transpose(1, 2)))
            return self.norm(0.5 * (state.mean(-1) + state[:, :, -1]))

    backbone_types = {
        "inception_time": InceptionTimeBackbone,
        "timesnet": TimesNetBackbone,
        "modern_tcn": ModernTCNBackbone,
    }
    backbone = backbone_types[family]()

    class MultiTaskModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = backbone
            width = int(backbone.output_width)
            self.detector = nn.Linear(width, 2)
            self.localizer = nn.Linear(width, components)
            self.classifier = nn.Linear(width, classes)
            self.comparison_family = family
            self.reference_adaptation = "ASTROGRAPH_SHORT_CAUSAL_WINDOW"

        def forward(self, value):
            state = self.encoder(value)
            return self.detector(state), self.localizer(state), self.classifier(state)

    return MultiTaskModel()


__all__ = ["SOTA_MODEL_FAMILIES", "make_sota_multitask_model"]
