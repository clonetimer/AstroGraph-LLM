"""Shared experimental TCN components for three-task satellite diagnosis.

This module is intentionally independent of dataset loaders.  Dataset-specific
scripts provide the frozen class-to-component mapping and remain responsible
for split, scaling, checkpoint and evaluation policy.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence


def prepare_torch_device(device: str, seed: int):
    """Initialize a requested Torch device before querying CUDA memory state."""
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    target = torch.device("cuda:0" if device == "cuda" else device)
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")
        torch.cuda.set_device(target)
        torch.cuda.manual_seed_all(seed)
        torch.empty(0, device=target)  # force lazy context initialization
        torch.cuda.reset_peak_memory_stats(target)
    return target


def receptive_field(dilations: Sequence[int], *, kernel_size: int = 3, convolutions_per_block: int = 2) -> int:
    """Return the receptive field of stride-one dilated convolution blocks."""
    if kernel_size < 1 or convolutions_per_block < 1 or any(value < 1 for value in dilations):
        raise ValueError("kernel size, convolution count and dilations must be positive")
    return 1 + (kernel_size - 1) * convolutions_per_block * sum(dilations)


def trajectory_uniform_weights(
    sample_ids: Sequence[str], *, nominal_multiplier: float = 1.0,
) -> list[float]:
    """Give every Train trajectory equal base mass, with optional nominal emphasis."""
    if nominal_multiplier <= 0.0:
        raise ValueError("nominal trajectory multiplier must be positive")
    counts: dict[str, int] = {}
    for sample_id in sample_ids:
        counts[str(sample_id)] = counts.get(str(sample_id), 0) + 1
    return [
        (nominal_multiplier if ":nominal:" in str(sample_id) else 1.0)
        / counts[str(sample_id)]
        for sample_id in sample_ids
    ]


def mean_plus_cvar(losses, *, cvar_weight: float = 0.0, cvar_fraction: float = 0.2):
    """Combine mean risk with upper-tail CVaR for Train-only hard examples."""
    import torch

    if losses.ndim != 1 or losses.numel() == 0:
        raise ValueError("CVaR losses must be a non-empty vector")
    if cvar_weight < 0.0 or not 0.0 < cvar_fraction <= 1.0:
        raise ValueError("CVaR weight/fraction are outside their valid range")
    mean = losses.mean()
    if cvar_weight == 0.0:
        return mean
    count = max(1, int(math.ceil(losses.numel() * cvar_fraction)))
    tail = torch.topk(losses, count, largest=True, sorted=False).values.mean()
    return mean + cvar_weight * tail


def hierarchy_js_loss(component_logits, class_logits, class_component, active):
    """Symmetric JS loss between direct and class-derived component probabilities."""
    import torch
    import torch.nn.functional as functional

    if not bool(active.any()):
        return component_logits.sum() * 0.0
    direct = functional.softmax(component_logits[active], dim=1)
    classes = functional.softmax(class_logits[active], dim=1)
    mapping = torch.as_tensor(class_component, dtype=torch.long, device=classes.device)
    if mapping.ndim != 1 or mapping.numel() != classes.shape[1]:
        raise ValueError("class_component must provide one component index per class")
    if mapping.numel() and (int(mapping.min()) < 0 or int(mapping.max()) >= direct.shape[1]):
        raise ValueError("class_component index outside component logits")
    derived = torch.zeros_like(direct)
    derived.scatter_add_(1, mapping.unsqueeze(0).expand_as(classes), classes)
    midpoint = 0.5 * (direct + derived)
    epsilon = torch.finfo(direct.dtype).eps
    direct_kl = functional.kl_div((midpoint + epsilon).log(), direct, reduction="batchmean")
    derived_kl = functional.kl_div((midpoint + epsilon).log(), derived, reduction="batchmean")
    return 0.5 * (direct_kl + derived_kl)


def make_shared_residual_tcn(
    channels: int,
    components: int,
    classes: int,
    *,
    width: int = 64,
):
    """Build the frozen fully-shared Residual TCN used by the ADCS P2 baseline."""
    from torch import nn

    if min(channels, components, classes, width) < 1:
        raise ValueError("model dimensions must be positive")

    class ResidualBlock(nn.Module):
        def __init__(self, dilation: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(4 if width % 4 == 0 else 1, width), nn.GELU(), nn.Dropout(0.1),
                nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(4 if width % 4 == 0 else 1, width), nn.GELU(),
            )

        def forward(self, value):
            return value + self.net(value)

    class TCNEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Conv1d(channels, width, 1)
            self.blocks = nn.Sequential(*(ResidualBlock(value) for value in (1, 2, 4)))

        def forward(self, value):
            hidden = self.blocks(self.input(value.transpose(1, 2)))
            return 0.5 * (hidden.mean(-1) + hidden[:, :, -1])

    class SharedResidualTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = TCNEncoder()
            self.detector = nn.Linear(width, 2)
            self.localizer = nn.Linear(width, components)
            self.classifier = nn.Linear(width, classes)
            self.branching_mode = "fully_shared"
            self.receptive_field = receptive_field((1, 2, 4))

        def forward(self, value):
            state = self.encoder(value)
            return self.detector(state), self.localizer(state), self.classifier(state)

    return SharedResidualTCN()


def _task_gradient_vectors(losses, parameters):
    import torch

    params = tuple(parameters)
    vectors = []
    for loss in losses:
        gradients = torch.autograd.grad(
            loss, params, retain_graph=True, allow_unused=True
        )
        vectors.append(torch.cat([
            (gradient if gradient is not None else torch.zeros_like(parameter)).reshape(-1)
            for parameter, gradient in zip(params, gradients, strict=True)
        ]))
    return params, vectors


def task_gradient_statistics(losses, parameters) -> dict[str, object]:
    """Measure pairwise task-gradient cosine similarities without changing grads."""
    import torch
    import torch.nn.functional as functional

    _, vectors = _task_gradient_vectors(losses, parameters)
    pairs = {}
    conflicts = 0
    total = 0
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            cosine = float(functional.cosine_similarity(
                vectors[left].unsqueeze(0), vectors[right].unsqueeze(0), dim=1, eps=1e-12
            ).detach())
            pairs[f"{left}-{right}"] = cosine
            conflicts += int(cosine < 0.0)
            total += 1
    return {
        "pairwise_cosine": pairs,
        "conflict_fraction": conflicts / total if total else 0.0,
        "gradient_norms": [float(value.norm().detach()) for value in vectors],
    }


def make_hierarchy_event_tcn(
    channels: int,
    components: int,
    classes: int,
    *,
    window_size: int,
    width: int = 64,
    consistency_weight: float = 0.1,
):
    """Build the bounded C1 candidate without binding it to a dataset."""
    import torch
    from torch import nn

    dilations = (1, 2, 4)
    if receptive_field(dilations) < window_size:
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

    class HierarchyEventTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Conv1d(channels, width, 1)
            self.blocks = nn.Sequential(*(ResidualBlock(value) for value in dilations))
            self.pool_projection = nn.Sequential(
                nn.Linear(width * 3, width), nn.GELU(), nn.LayerNorm(width)
            )
            self.detector = nn.Linear(width, 2)
            self.localizer = nn.Linear(width, components)
            self.classifier = nn.Linear(width, classes)
            self.dilations = dilations
            self.receptive_field = receptive_field(dilations)
            self.consistency_weight = float(consistency_weight)

        def forward(self, value):
            hidden = self.blocks(self.input(value.transpose(1, 2)))
            state = self.pool_projection(
                torch.cat((hidden.mean(-1), hidden.amax(-1), hidden[:, :, -1]), dim=1)
            )
            return self.detector(state), self.localizer(state), self.classifier(state)

    return HierarchyEventTCN()


def make_causal_multiview_detector_tcn(
    channels: int,
    components: int,
    classes: int,
    *,
    window_size: int,
    raw_width: int = 48,
    change_width: int = 32,
    fused_width: int = 64,
):
    """Build a detector specialized for operating-point shift and fault onset.

    The four causal endpoint views retain complementary information: absolute
    level, first difference, change relative to the early-window anchor and
    explicit distribution/change statistics.  Localization/classification
    heads only preserve the common research interface; detection-only training
    never optimizes them.
    """
    import torch
    from torch import nn

    if channels < 1 or components < 1 or classes < 1 or window_size < 4:
        raise ValueError("positive dimensions and a window of at least four samples are required")

    class Block(nn.Module):
        def __init__(self, width: int, dilation: int) -> None:
            super().__init__()
            groups = 4 if width % 4 == 0 else 1
            self.net = nn.Sequential(
                nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(groups, width), nn.GELU(), nn.Dropout(0.1),
                nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(groups, width), nn.GELU(),
            )

        def forward(self, value):
            return value + self.net(value)

    class TemporalView(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.input = nn.Conv1d(channels, width, 1)
            self.blocks = nn.Sequential(*(Block(width, dilation) for dilation in (1, 2, 4)))

        def forward(self, value):
            hidden = self.blocks(self.input(value.transpose(1, 2)))
            return 0.5 * (hidden.mean(-1) + hidden[:, :, -1])

    class CausalMultiviewDetectorTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.raw_view = TemporalView(raw_width)
            self.delta_view = TemporalView(change_width)
            self.relative_view = TemporalView(change_width)
            self.statistics = nn.Sequential(
                nn.Linear(channels * 5, change_width), nn.GELU(), nn.LayerNorm(change_width)
            )
            total_width = raw_width + 3 * change_width
            self.fusion = nn.Sequential(
                nn.Linear(total_width, fused_width), nn.GELU(), nn.LayerNorm(fused_width),
                nn.Dropout(0.1),
            )
            self.detector = nn.Linear(fused_width, 2)
            self.localizer = nn.Linear(fused_width, components)
            self.classifier = nn.Linear(fused_width, classes)
            self.branching_mode = "detection_specialist_absolute_delta_anchor_statistics"
            self.receptive_field = receptive_field((1, 2, 4))
            self.anchor_length = max(4, window_size // 4)

        def forward(self, value):
            delta = torch.diff(value, dim=1, prepend=value[:, :1])
            anchor = value[:, : self.anchor_length]
            center = anchor.mean(dim=1, keepdim=True)
            scale = anchor.std(dim=1, keepdim=True, unbiased=False).clamp_min(0.25)
            relative = (value - center) / scale
            statistics = torch.cat((
                value.mean(dim=1),
                value.std(dim=1, unbiased=False),
                value[:, -1] - value[:, 0],
                delta.abs().mean(dim=1),
                relative[:, -1],
            ), dim=1)
            state = self.fusion(torch.cat((
                self.raw_view(value),
                self.delta_view(delta),
                self.relative_view(relative),
                self.statistics(statistics),
            ), dim=1))
            return self.detector(state), self.localizer(state), self.classifier(state)

    return CausalMultiviewDetectorTCN()


def make_decoupled_tcn(
    kind: str,
    channels: int,
    components: int,
    classes: int,
    *,
    width: int = 64,
    adapter_width: int = 16,
):
    """Build bounded task-specialization variants with the standard three heads."""
    import torch
    from torch import nn

    allowed = {
        "pooled_task_adapter_tcn",
        "late_fork_tcn",
        "tied_late_fork_tcn",
        "capacity_matched_shared_tcn",
    }
    if kind not in allowed:
        raise ValueError(f"unknown decoupled TCN kind: {kind}")

    class Block(nn.Module):
        def __init__(self, block_width: int, dilation: int) -> None:
            super().__init__()
            groups = 4 if block_width % 4 == 0 else 1
            self.net = nn.Sequential(
                nn.Conv1d(block_width, block_width, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(groups, block_width), nn.GELU(), nn.Dropout(0.1),
                nn.Conv1d(block_width, block_width, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(groups, block_width), nn.GELU(),
            )

        def forward(self, value):
            return value + self.net(value)

    class Adapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down = nn.Linear(width, adapter_width)
            self.activation = nn.GELU()
            self.up = nn.Linear(adapter_width, width)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

        def forward(self, value):
            return value + self.up(self.activation(self.down(value)))

    def pool(value):
        return 0.5 * (value.mean(-1) + value[:, :, -1])

    class PooledTaskAdapterTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Conv1d(channels, width, 1)
            self.shared_blocks = nn.Sequential(*(Block(width, value) for value in (1, 2, 4)))
            self.detection_adapter = Adapter()
            self.diagnosis_adapter = Adapter()
            self.detector = nn.Linear(width, 2)
            self.localizer = nn.Linear(width, components)
            self.classifier = nn.Linear(width, classes)
            self.branching_mode = "pooled_task_adapters"
            self.receptive_field = receptive_field((1, 2, 4))

        def forward(self, value):
            shared = pool(self.shared_blocks(self.input(value.transpose(1, 2))))
            detection_state = self.detection_adapter(shared)
            diagnosis_state = self.diagnosis_adapter(shared)
            return (
                self.detector(detection_state),
                self.localizer(diagnosis_state),
                self.classifier(diagnosis_state),
            )

    class LateForkTCN(nn.Module):
        def __init__(self, *, tied: bool = False) -> None:
            super().__init__()
            self.input = nn.Conv1d(channels, width, 1)
            self.shared_blocks = nn.Sequential(Block(width, 1), Block(width, 2))
            self.detection_branch = Block(width, 4)
            self.diagnosis_branch = self.detection_branch if tied else Block(width, 4)
            self.detector = nn.Linear(width, 2)
            self.localizer = nn.Linear(width, components)
            self.classifier = nn.Linear(width, classes)
            self.branching_mode = "tied_late_fork" if tied else "independent_late_fork"
            self.receptive_field = receptive_field((1, 2, 4))

        def forward(self, value):
            shared = self.shared_blocks(self.input(value.transpose(1, 2)))
            detection_state = pool(self.detection_branch(shared))
            diagnosis_state = pool(self.diagnosis_branch(shared))
            return (
                self.detector(detection_state),
                self.localizer(diagnosis_state),
                self.classifier(diagnosis_state),
            )

    class CapacityMatchedSharedTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            matched_width = 72
            self.input = nn.Conv1d(channels, matched_width, 1)
            self.shared_blocks = nn.Sequential(*(Block(matched_width, value) for value in (1, 2, 4)))
            self.detector = nn.Linear(matched_width, 2)
            self.localizer = nn.Linear(matched_width, components)
            self.classifier = nn.Linear(matched_width, classes)
            self.branching_mode = "capacity_matched_fully_shared"
            self.receptive_field = receptive_field((1, 2, 4))

        def forward(self, value):
            shared = pool(self.shared_blocks(self.input(value.transpose(1, 2))))
            return self.detector(shared), self.localizer(shared), self.classifier(shared)

    if kind == "pooled_task_adapter_tcn":
        return PooledTaskAdapterTCN()
    if kind == "late_fork_tcn":
        return LateForkTCN()
    if kind == "tied_late_fork_tcn":
        return LateForkTCN(tied=True)
    return CapacityMatchedSharedTCN()


def make_mechanism_gated_gat_tcn(
    channels: int,
    components: int,
    classes: int,
    *,
    subsystem_groups: Sequence[Sequence[int]],
    component_group: Sequence[int],
    class_component: Sequence[int],
    graph_edges: Sequence[tuple[int, int]],
    temporal_width: int = 64,
    node_width: int = 16,
    attention_heads: int = 4,
    temporal_norm_groups: int = 4,
    initial_message_gate: float = 0.1,
    gate_regularization_weight: float = 0.01,
):
    """Build a bounded GAT challenger for localization and classification.

    Detection logits retain a graph-free global causal TCN path. Diagnosis uses
    the shared temporal context plus subsystem-local graph attention. Thus the
    ordinary multi-task representation benefit remains, while graph messages
    can never flow directly into the detector.
    """
    import torch
    from torch import nn

    groups = tuple(tuple(int(value) for value in group) for group in subsystem_groups)
    component_binding = tuple(int(value) for value in component_group)
    class_binding = tuple(int(value) for value in class_component)
    edges = tuple((int(source), int(target)) for source, target in graph_edges)
    node_count = len(groups)
    if not groups or any(not group for group in groups):
        raise ValueError("every graph node requires at least one telemetry channel")
    flattened = [value for group in groups for value in group]
    if len(set(flattened)) != len(flattened):
        raise ValueError("telemetry channels must not be assigned to multiple graph nodes")
    if min(flattened) < 0 or max(flattened) >= channels:
        raise ValueError("graph channel index outside model input")
    if len(component_binding) != components:
        raise ValueError("component_group must provide one graph node per component")
    if len(class_binding) != classes:
        raise ValueError("class_component must provide one component per class")
    if component_binding and (min(component_binding) < 0 or max(component_binding) >= node_count):
        raise ValueError("component_group index outside graph nodes")
    if class_binding and (min(class_binding) < 0 or max(class_binding) >= components):
        raise ValueError("class_component index outside components")
    if edges and any(min(edge) < 0 or max(edge) >= node_count for edge in edges):
        raise ValueError("graph edge index outside graph nodes")
    if not 0.0 < initial_message_gate < 1.0:
        raise ValueError("initial_message_gate must be strictly between zero and one")
    if node_width % attention_heads:
        raise ValueError("node_width must be divisible by attention_heads")
    if temporal_width % temporal_norm_groups:
        raise ValueError("temporal_width must be divisible by temporal_norm_groups")

    class ResidualBlock(nn.Module):
        def __init__(self, width: int, dilation: int, *, dropout: float = 0.1) -> None:
            super().__init__()
            group_count = temporal_norm_groups if width == temporal_width else (4 if width % 4 == 0 else 1)
            self.net = nn.Sequential(
                nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(group_count, width), nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(group_count, width), nn.GELU(),
            )

        def forward(self, value):
            return value + self.net(value)

    class MaskedGraphAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            head_width = node_width // attention_heads
            self.head_width = head_width
            self.projection = nn.Linear(node_width, node_width, bias=False)
            self.source_attention = nn.Parameter(torch.empty(attention_heads, head_width))
            self.target_attention = nn.Parameter(torch.empty(attention_heads, head_width))
            self.output = nn.Linear(node_width, node_width, bias=False)
            self.dropout = nn.Identity()
            self.norm = nn.LayerNorm(node_width)
            gate_logit = math.log(initial_message_gate / (1.0 - initial_message_gate))
            self.message_gate_logit = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))
            mask = torch.eye(node_count, dtype=torch.bool)
            for source, target in edges:
                mask[target, source] = True
            self.register_buffer("attention_mask", mask)
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.xavier_uniform_(self.output.weight)
            nn.init.xavier_uniform_(self.source_attention)
            nn.init.xavier_uniform_(self.target_attention)

        def forward(self, value):
            batch = value.shape[0]
            projected = self.projection(value).view(
                batch, node_count, attention_heads, self.head_width
            ).transpose(1, 2)
            source_score = (projected * self.source_attention[None, :, None, :]).sum(-1)
            target_score = (projected * self.target_attention[None, :, None, :]).sum(-1)
            logits = torch.nn.functional.leaky_relu(
                target_score.unsqueeze(-1) + source_score.unsqueeze(-2), negative_slope=0.2
            )
            logits = logits.masked_fill(~self.attention_mask[None, None, :, :], -torch.inf)
            attention = self.dropout(torch.softmax(logits, dim=-1))
            messages = torch.matmul(attention, projected).transpose(1, 2).reshape(
                batch, node_count, node_width
            )
            gate = torch.sigmoid(self.message_gate_logit)
            return self.norm(value + gate * self.output(messages)), attention

    class MechanismGatedGATTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.global_input = nn.Conv1d(channels, temporal_width, 1)
            self.global_blocks = nn.Sequential(*(
                ResidualBlock(temporal_width, dilation) for dilation in (1, 2, 4)
            ))
            # Construct this head immediately after the global encoder so its
            # seeded initialization matches the flat Residual TCN baseline.
            self.detector = nn.Linear(temporal_width, 2)
            self.local_inputs = nn.ModuleList(
                nn.Conv1d(len(indices), node_width, 1) for indices in groups
            )
            self.local_blocks = nn.ModuleList(
                nn.Sequential(*(
                    ResidualBlock(node_width, dilation, dropout=0.0)
                    for dilation in (1, 2, 4)
                ))
                for _ in groups
            )
            for index, values in enumerate(groups):
                self.register_buffer(
                    f"subsystem_index_{index}", torch.tensor(values, dtype=torch.long)
                )
            self.graph_attention = MaskedGraphAttention()
            diagnosis_width = temporal_width + node_width
            self.localizer = nn.ModuleList(nn.Linear(diagnosis_width, 1) for _ in component_binding)
            self.classifier = nn.ModuleList(nn.Linear(diagnosis_width, 1) for _ in class_binding)
            self.component_group = component_binding
            self.class_component = class_binding
            self.graph_edges = edges
            self.branching_mode = "shared_temporal_graph_free_detection_plus_gated_gat_diagnosis"
            self.receptive_field = receptive_field((1, 2, 4))
            self.graph_gate_regularization_weight = float(gate_regularization_weight)
            self.last_attention = None

        @staticmethod
        def pool(value):
            return 0.5 * (value.mean(-1) + value[:, :, -1])

        def graph_regularization_loss(self):
            return (
                self.graph_gate_regularization_weight
                * torch.sigmoid(self.graph_attention.message_gate_logit).square()
            )

        def graph_diagnostics(self) -> dict[str, object]:
            return {
                "message_gate": float(torch.sigmoid(
                    self.graph_attention.message_gate_logit.detach()
                ).cpu()),
                "directed_edges_excluding_self": [list(edge) for edge in self.graph_edges],
                "self_loops": node_count,
            }

        def forward(self, value):
            temporal = value.transpose(1, 2)
            global_state = self.pool(self.global_blocks(self.global_input(temporal)))
            local_states = []
            for index, (projection, blocks) in enumerate(zip(
                self.local_inputs, self.local_blocks, strict=True
            )):
                channel_index = getattr(self, f"subsystem_index_{index}")
                local_states.append(self.pool(blocks(projection(
                    temporal.index_select(1, channel_index)
                ))))
            graph_state, attention = self.graph_attention(torch.stack(local_states, dim=1))
            self.last_attention = attention.detach()
            diagnosis_context = global_state
            localization = torch.cat([
                head(torch.cat([diagnosis_context, graph_state[:, graph_index]], dim=1))
                for graph_index, head in zip(component_binding, self.localizer, strict=True)
            ], dim=1)
            classification = torch.cat([
                head(torch.cat([
                    diagnosis_context,
                    graph_state[:, component_binding[class_binding[index]]],
                ], dim=1))
                for index, head in enumerate(self.classifier)
            ], dim=1)
            return self.detector(global_state), localization, classification

    return MechanismGatedGATTCN()


def make_hierarchical_mechanism_gat_tcn(
    channels: int,
    components: int,
    classes: int,
    *,
    component_groups: Sequence[Sequence[int]],
    subsystem_groups: Sequence[Sequence[int]],
    component_subsystem: Sequence[int],
    class_component: Sequence[int],
    subsystem_edges: Sequence[tuple[int, int]],
    temporal_width: int = 64,
    node_width: int = 16,
    attention_heads: int = 4,
    temporal_norm_groups: int = 8,
    initial_message_gate: float = 0.1,
    gate_regularization_weight: float = 0.01,
):
    """Component -> subsystem -> spacecraft temporal GAT challenger.

    The detector deliberately retains a graph-free global TCN. Diagnosis uses
    explicit component observations, owned-subsystem aggregation, reviewed
    cross-subsystem edges and a whole-spacecraft context. This prevents a
    subsystem node from standing in for several distinct localization labels.
    """
    import torch
    from torch import nn

    component_channels = tuple(tuple(int(value) for value in group) for group in component_groups)
    subsystem_channels = tuple(tuple(int(value) for value in group) for group in subsystem_groups)
    component_owner = tuple(int(value) for value in component_subsystem)
    class_owner = tuple(int(value) for value in class_component)
    edges = tuple((int(source), int(target)) for source, target in subsystem_edges)
    subsystem_count = len(subsystem_channels)
    if len(component_channels) != components or any(not group for group in component_channels):
        raise ValueError("one non-empty component channel group is required per component")
    if not subsystem_channels or any(not group for group in subsystem_channels):
        raise ValueError("every subsystem requires observable channels")
    if len(component_owner) != components or any(
        value < 0 or value >= subsystem_count for value in component_owner
    ):
        raise ValueError("component_subsystem must bind every component to a subsystem")
    if len(class_owner) != classes or any(value < 0 or value >= components for value in class_owner):
        raise ValueError("class_component must bind every class to a component")
    all_indices = [
        value for group in (*component_channels, *subsystem_channels) for value in group
    ]
    if min(all_indices) < 0 or max(all_indices) >= channels:
        raise ValueError("hierarchical channel index outside model input")
    if edges and any(min(edge) < 0 or max(edge) >= subsystem_count for edge in edges):
        raise ValueError("subsystem edge index outside hierarchy")
    if node_width % attention_heads:
        raise ValueError("node_width must be divisible by attention_heads")
    if temporal_width % temporal_norm_groups:
        raise ValueError("temporal_width must be divisible by temporal_norm_groups")
    if not 0.0 < initial_message_gate < 1.0:
        raise ValueError("initial_message_gate must be strictly between zero and one")

    gate_logit = math.log(initial_message_gate / (1.0 - initial_message_gate))

    class ResidualBlock(nn.Module):
        def __init__(self, width: int, dilation: int, *, dropout: float = 0.1) -> None:
            super().__init__()
            groups = temporal_norm_groups if width == temporal_width else (
                4 if width % 4 == 0 else 1
            )
            self.net = nn.Sequential(
                nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(groups, width), nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(groups, width), nn.GELU(),
            )

        def forward(self, value):
            return value + self.net(value)

    class MaskedSubsystemAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = nn.MultiheadAttention(
                node_width, attention_heads, batch_first=True, dropout=0.0
            )
            mask = torch.ones((subsystem_count, subsystem_count), dtype=torch.bool)
            mask.fill_diagonal_(False)
            for source, target in edges:
                mask[target, source] = False
            self.register_buffer("blocked_attention", mask)
            self.message_gate_logit = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))
            self.norm = nn.LayerNorm(node_width)

        def forward(self, value):
            message, attention = self.attention(
                value, value, value, attn_mask=self.blocked_attention,
                need_weights=True, average_attn_weights=False,
            )
            gate = torch.sigmoid(self.message_gate_logit)
            return self.norm(value + gate * message), attention

    owned_components = tuple(
        tuple(index for index, owner in enumerate(component_owner) if owner == subsystem)
        for subsystem in range(subsystem_count)
    )

    class HierarchicalMechanismGATTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.global_input = nn.Conv1d(channels, temporal_width, 1)
            self.global_blocks = nn.Sequential(*(
                ResidualBlock(temporal_width, dilation) for dilation in (1, 2, 4)
            ))
            # Keep seeded graph-free detection comparable with Residual TCN.
            self.detector = nn.Linear(temporal_width, 2)
            self.component_inputs = nn.ModuleList(
                nn.Conv1d(len(indices), node_width, 1) for indices in component_channels
            )
            self.component_blocks = nn.ModuleList(
                nn.Sequential(*(ResidualBlock(node_width, dilation, dropout=0.0) for dilation in (1, 2)))
                for _ in component_channels
            )
            self.subsystem_inputs = nn.ModuleList(
                nn.Conv1d(len(indices), node_width, 1) for indices in subsystem_channels
            )
            self.subsystem_blocks = nn.ModuleList(
                nn.Sequential(*(ResidualBlock(node_width, dilation, dropout=0.0) for dilation in (1, 2)))
                for _ in subsystem_channels
            )
            for index, values in enumerate(component_channels):
                self.register_buffer(
                    f"component_index_{index}", torch.tensor(values, dtype=torch.long)
                )
            for index, values in enumerate(subsystem_channels):
                self.register_buffer(
                    f"subsystem_index_{index}", torch.tensor(values, dtype=torch.long)
                )
            self.bottom_up_attention = nn.ModuleList(
                nn.MultiheadAttention(node_width, attention_heads, batch_first=True, dropout=0.0)
                for _ in subsystem_channels
            )
            self.bottom_up_gate_logits = nn.Parameter(
                torch.full((subsystem_count,), gate_logit, dtype=torch.float32)
            )
            self.bottom_up_norms = nn.ModuleList(nn.LayerNorm(node_width) for _ in subsystem_channels)
            self.subsystem_attention = MaskedSubsystemAttention()
            self.spacecraft_projection = nn.Linear(temporal_width, node_width)
            self.spacecraft_norm = nn.LayerNorm(node_width)
            self.top_down_gate_logit = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))
            self.top_down_projection = nn.Linear(node_width, node_width, bias=False)
            self.top_down_norms = nn.ModuleList(nn.LayerNorm(node_width) for _ in subsystem_channels)
            diagnosis_width = temporal_width + 2 * node_width
            self.localizer = nn.ModuleList(nn.Linear(diagnosis_width, 1) for _ in component_owner)
            self.classifier = nn.ModuleList(nn.Linear(diagnosis_width, 1) for _ in class_owner)
            self.subsystem_localizer = nn.ModuleList(
                nn.Linear(temporal_width + node_width, 1) for _ in subsystem_channels
            )
            self.component_subsystem = component_owner
            self.class_component = class_owner
            self.subsystem_edges = edges
            self.branching_mode = (
                "graph_free_detection_plus_component_subsystem_spacecraft_gat_diagnosis"
            )
            self.receptive_field = receptive_field((1, 2, 4))
            self.graph_gate_regularization_weight = float(gate_regularization_weight)
            self.last_subsystem_logits = None
            self.last_attention = None

        @staticmethod
        def pool(value):
            return 0.5 * (value.mean(-1) + value[:, :, -1])

        def graph_regularization_loss(self):
            gates = torch.cat((
                torch.sigmoid(self.bottom_up_gate_logits),
                torch.sigmoid(self.subsystem_attention.message_gate_logit).reshape(1),
                torch.sigmoid(self.top_down_gate_logit).reshape(1),
            ))
            return self.graph_gate_regularization_weight * gates.square().mean()

        def graph_diagnostics(self) -> dict[str, object]:
            return {
                "hierarchy_levels": ["component", "subsystem", "whole_spacecraft"],
                "component_count": components,
                "subsystem_count": subsystem_count,
                "bottom_up_gates": torch.sigmoid(
                    self.bottom_up_gate_logits.detach()
                ).cpu().tolist(),
                "cross_subsystem_gate": float(torch.sigmoid(
                    self.subsystem_attention.message_gate_logit.detach()
                ).cpu()),
                "top_down_gate": float(torch.sigmoid(self.top_down_gate_logit.detach()).cpu()),
                "directed_subsystem_edges_excluding_self": [list(edge) for edge in edges],
            }

        def forward(self, value):
            temporal = value.transpose(1, 2)
            global_state = self.pool(self.global_blocks(self.global_input(temporal)))
            component_states = []
            for index, (projection, blocks) in enumerate(zip(
                self.component_inputs, self.component_blocks, strict=True
            )):
                indices = getattr(self, f"component_index_{index}")
                component_states.append(self.pool(blocks(projection(
                    temporal.index_select(1, indices)
                ))))
            component_state = torch.stack(component_states, dim=1)
            subsystem_states = []
            for index, (projection, blocks, attention, norm) in enumerate(zip(
                self.subsystem_inputs, self.subsystem_blocks,
                self.bottom_up_attention, self.bottom_up_norms, strict=True,
            )):
                indices = getattr(self, f"subsystem_index_{index}")
                local = self.pool(blocks(projection(temporal.index_select(1, indices))))
                if owned_components[index]:
                    owned = component_state[:, owned_components[index]]
                    message, _ = attention(local.unsqueeze(1), owned, owned, need_weights=False)
                    gate = torch.sigmoid(self.bottom_up_gate_logits[index])
                    local = local + gate * message[:, 0]
                subsystem_states.append(norm(local))
            subsystem_state, attention = self.subsystem_attention(
                torch.stack(subsystem_states, dim=1)
            )
            spacecraft_state = self.spacecraft_norm(
                self.spacecraft_projection(global_state) + subsystem_state.mean(1)
            )
            top_gate = torch.sigmoid(self.top_down_gate_logit)
            top_message = self.top_down_projection(spacecraft_state)
            final_subsystems = torch.stack([
                norm(subsystem_state[:, index] + top_gate * top_message)
                for index, norm in enumerate(self.top_down_norms)
            ], dim=1)
            self.last_attention = attention.detach()
            self.last_subsystem_logits = torch.cat([
                head(torch.cat([global_state, final_subsystems[:, index]], dim=1))
                for index, head in enumerate(self.subsystem_localizer)
            ], dim=1)
            localization = torch.cat([
                head(torch.cat([
                    global_state,
                    component_state[:, component_index],
                    final_subsystems[:, subsystem_index],
                ], dim=1))
                for component_index, (subsystem_index, head) in enumerate(zip(
                    component_owner, self.localizer, strict=True
                ))
            ], dim=1)
            classification = torch.cat([
                head(torch.cat([
                    global_state,
                    component_state[:, component_index],
                    final_subsystems[:, component_owner[component_index]],
                ], dim=1))
                for component_index, head in zip(class_owner, self.classifier, strict=True)
            ], dim=1)
            return self.detector(global_state), localization, classification

    return HierarchicalMechanismGATTCN()
