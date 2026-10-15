"""Unified domain model skeleton for the AI-Native roadmap Phase 3.

One shared dilated-residual backbone with per-dataset input projections and
per-dataset task heads (detection / localization / classification). The shared
parameters are the generalization slot; dataset adapters carry only the
input/output surface differences (channels, component/class counts).

This module is a bounded architecture candidate. It performs no training and
is not bound to any production Registry; training follows the preregistered
protocol in docs/CURRENT_ENGINEERING_POLICY.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from .three_task_tcn import receptive_field


@dataclass(frozen=True)
class DatasetTaskSpec:
    """Input/output surface of one dataset under the unified backbone."""

    channels: int
    components: int
    classes: int
    window_size: int

    def __post_init__(self) -> None:
        for name, value in (
            ("channels", self.channels),
            ("components", self.components),
            ("classes", self.classes),
            ("window_size", self.window_size),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive: {value}")


def make_unified_platform_model(
    dataset_specs: dict[str, DatasetTaskSpec],
    *,
    width: int = 64,
    consistency_weight: float = 0.1,
):
    """Build the unified backbone with per-dataset adapters.

    Returns an nn.Module whose forward(x, dataset_id) routes a window tensor
    through the matching input projection -> shared blocks -> shared pooling,
    then through the dataset task heads. Shared parameters are exactly the
    blocks and pooling projection; adapters hold input projections and heads.
    """
    import torch
    from torch import nn

    if not dataset_specs:
        raise ValueError("at least one dataset spec is required")
    if width <= 0:
        raise ValueError(f"width must be positive: {width}")
    windows = {spec.window_size for spec in dataset_specs.values()}
    max_window = max(windows)
    dilations = (1, 2, 4)
    if receptive_field(dilations) < max_window:
        dilations = (*dilations, 8)

    class ResidualBlock(nn.Module):
        def __init__(self, dilation: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(4 if width % 4 == 0 else 1, width),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(4 if width % 4 == 0 else 1, width),
                nn.GELU(),
            )

        def forward(self, value):
            return value + self.net(value)

    class DatasetAdapter(nn.Module):
        """Input projection plus task heads for one dataset."""

        def __init__(self, spec: DatasetTaskSpec) -> None:
            super().__init__()
            self.channels = spec.channels
            self.input_projection = nn.Conv1d(spec.channels, width, 1)
            self.detector = nn.Linear(width, 2)
            self.localizer = nn.Linear(width, spec.components)
            self.classifier = nn.Linear(width, spec.classes)

        def forward(self, state: torch.Tensor):
            return (
                self.detector(state),
                self.localizer(state),
                self.classifier(state),
            )

    class UnifiedPlatformModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.Sequential(*(ResidualBlock(value) for value in dilations))
            self.pool_projection = nn.Sequential(
                nn.Linear(width * 3, width), nn.GELU(), nn.LayerNorm(width),
            )
            self.adapters = nn.ModuleDict(
                {name: DatasetAdapter(spec) for name, spec in dataset_specs.items()}
            )
            self.dilations = dilations
            self.receptive_field = receptive_field(dilations)
            self.consistency_weight = float(consistency_weight)
            self.dataset_specs = {
                name: {
                    "channels": spec.channels,
                    "components": spec.components,
                    "classes": spec.classes,
                    "window_size": spec.window_size,
                }
                for name, spec in dataset_specs.items()
            }

        def shared_parameter_count(self) -> int:
            blocks_total = sum(
                parameter.numel() for parameter in self.blocks.parameters()
            )
            pool_total = sum(
                parameter.numel() for parameter in self.pool_projection.parameters()
            )
            return blocks_total + pool_total

        def forward(self, value: torch.Tensor, dataset_id: str):
            if dataset_id not in self.adapters:
                raise KeyError(f"unknown dataset adapter: {dataset_id}")
            adapter = self.adapters[dataset_id]
            if value.shape[-1] != adapter.channels:
                raise ValueError(
                    f"{dataset_id} expects {adapter.channels} channels, got {value.shape[-1]}",
                )
            hidden = self.blocks(adapter.input_projection(value.transpose(1, 2)))
            state = self.pool_projection(
                torch.cat((
                    hidden.mean(-1),
                    hidden.amax(-1),
                    hidden[:, :, -1],
                ), dim=1),
            )
            return adapter.forward(state)

    return UnifiedPlatformModel()


__all__ = ["DatasetTaskSpec", "make_unified_platform_model"]
