"""Progressive parameter-to-spacecraft graph attention for diagnosis research.

The topology is a reviewed engineering prior.  Detection deliberately uses a
separate global temporal path; graph messages may affect localization and
classification only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def make_progressive_hierarchical_gat_tcn(
    channels: int,
    components: int,
    classes: int,
    *,
    parameter_groups: Sequence[Sequence[int]],
    parameter_part: Sequence[int],
    part_component: Sequence[int],
    component_module: Sequence[int],
    module_subsystem: Sequence[int],
    class_component: Sequence[int],
    temporal_width: int = 64,
    node_width: int = 16,
    attention_heads: int = 4,
    initial_message_gate: float = 0.1,
    gate_regularization_weight: float = 0.01,
):
    """Build a six-level, bottom-up/top-down diagnosis network.

    Levels are parameter, part, component, functional module, subsystem and
    whole spacecraft.  Each parent attends only to its declared children.
    """
    import torch
    from torch import nn

    groups = tuple(tuple(int(index) for index in group) for group in parameter_groups)
    bindings = tuple(
        tuple(int(index) for index in values)
        for values in (parameter_part, part_component, component_module, module_subsystem)
    )
    class_owner = tuple(int(index) for index in class_component)
    if not groups or any(not group for group in groups):
        raise ValueError("every parameter node requires at least one telemetry channel")
    if len(bindings[0]) != len(groups):
        raise ValueError("parameter_part must bind every parameter node")
    expected_children = (len(groups),) + tuple(max(values) + 1 for values in bindings[:-1])
    for level, (values, expected) in enumerate(zip(bindings, expected_children, strict=True)):
        if len(values) != expected or min(values, default=-1) < 0:
            raise ValueError(f"invalid hierarchy binding at level {level}")
        parent_count = max(values) + 1
        if set(values) != set(range(parent_count)):
            raise ValueError(f"hierarchy level {level} contains an empty parent")
    if max(bindings[1]) + 1 != components:
        raise ValueError("part_component must resolve exactly to the localization components")
    if len(class_owner) != classes or any(value < 0 or value >= components for value in class_owner):
        raise ValueError("class_component must bind every class to a component")
    flattened = [index for group in groups for index in group]
    if min(flattened) < 0 or max(flattened) >= channels:
        raise ValueError("parameter channel index outside model input")
    if node_width % attention_heads:
        raise ValueError("node_width must be divisible by attention_heads")
    if temporal_width % 8:
        raise ValueError("temporal_width must be divisible by eight")
    if not 0.0 < initial_message_gate < 1.0:
        raise ValueError("initial_message_gate must be strictly between zero and one")

    counts = (
        len(groups),
        max(bindings[0]) + 1,
        components,
        max(bindings[2]) + 1,
        max(bindings[3]) + 1,
        1,
    )
    gate_logit = math.log(initial_message_gate / (1.0 - initial_message_gate))

    class ResidualBlock(nn.Module):
        def __init__(self, width: int, dilation: int, dropout: float) -> None:
            super().__init__()
            groups_count = 8 if width == temporal_width else (4 if width % 4 == 0 else 1)
            self.net = nn.Sequential(
                nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(groups_count, width), nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(groups_count, width), nn.GELU(),
            )

        def forward(self, value):
            return value + self.net(value)

    class ChildAttentionLayer(nn.Module):
        def __init__(self, owners: tuple[int, ...]) -> None:
            super().__init__()
            self.owners = owners
            parent_count = max(owners) + 1
            self.queries = nn.Parameter(torch.empty(parent_count, node_width))
            nn.init.xavier_uniform_(self.queries)
            self.attention = nn.ModuleList(
                nn.MultiheadAttention(node_width, attention_heads, batch_first=True, dropout=0.0)
                for _ in range(parent_count)
            )
            self.norms = nn.ModuleList(nn.LayerNorm(node_width) for _ in range(parent_count))

        def forward(self, children):
            parents = []
            weights = []
            for parent, (attention, norm) in enumerate(zip(self.attention, self.norms, strict=True)):
                indices = [index for index, owner in enumerate(self.owners) if owner == parent]
                owned = children[:, indices]
                query = self.queries[parent].expand(children.shape[0], 1, -1)
                message, weight = attention(query, owned, owned, need_weights=True)
                parents.append(norm(query[:, 0] + message[:, 0]))
                weights.append(weight.detach())
            return torch.stack(parents, dim=1), weights

    class ProgressiveHierarchicalGATTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.global_input = nn.Conv1d(channels, temporal_width, 1)
            self.global_blocks = nn.Sequential(*(
                ResidualBlock(temporal_width, dilation, 0.1) for dilation in (1, 2, 4)
            ))
            self.detector = nn.Linear(temporal_width, 2)
            self.parameter_inputs = nn.ModuleList(
                nn.Conv1d(len(indices), node_width, 1) for indices in groups
            )
            self.parameter_blocks = nn.ModuleList(
                nn.Sequential(*(ResidualBlock(node_width, dilation, 0.0) for dilation in (1, 2)))
                for _ in groups
            )
            for index, indices in enumerate(groups):
                self.register_buffer(
                    f"parameter_index_{index}", torch.tensor(indices, dtype=torch.long)
                )
            upward_owners = (*bindings, tuple(0 for _ in range(counts[-2])))
            self.upward = nn.ModuleList(ChildAttentionLayer(owner) for owner in upward_owners)
            self.downward_projections = nn.ModuleList(
                nn.Linear(node_width, node_width, bias=False) for _ in bindings[1:]
            )
            self.downward_norms = nn.ModuleList(
                nn.LayerNorm(node_width) for _ in bindings[1:]
            )
            self.downward_gate_logits = nn.Parameter(
                torch.full((len(bindings) - 1,), gate_logit, dtype=torch.float32)
            )
            diagnosis_width = temporal_width + node_width
            self.localizer = nn.ModuleList(nn.Linear(diagnosis_width, 1) for _ in range(components))
            self.classifier = nn.ModuleList(nn.Linear(diagnosis_width, 1) for _ in class_owner)
            self.class_component = class_owner
            self.level_counts = counts
            self.branching_mode = "graph_free_detection_plus_six_level_progressive_gat_diagnosis"
            self.receptive_field = 29
            self.graph_gate_regularization_weight = float(gate_regularization_weight)
            self.last_attention = None

        @staticmethod
        def pool(value):
            return 0.5 * (value.mean(-1) + value[:, :, -1])

        def graph_regularization_loss(self):
            gates = torch.sigmoid(self.downward_gate_logits)
            return self.graph_gate_regularization_weight * gates.square().mean()

        def graph_diagnostics(self) -> dict[str, object]:
            return {
                "hierarchy_levels": [
                    "parameter", "part", "component", "functional_module",
                    "subsystem", "whole_spacecraft",
                ],
                "level_counts": list(self.level_counts),
                "top_down_gates": torch.sigmoid(
                    self.downward_gate_logits.detach()
                ).cpu().tolist(),
                "detection_uses_graph": False,
                "topology_fit_scope": "reviewed configuration only; no telemetry values or labels",
            }

        def forward(self, value):
            temporal = value.transpose(1, 2)
            global_state = self.pool(self.global_blocks(self.global_input(temporal)))
            parameters = []
            for index, (projection, blocks) in enumerate(zip(
                self.parameter_inputs, self.parameter_blocks, strict=True
            )):
                channel_index = getattr(self, f"parameter_index_{index}")
                parameters.append(self.pool(blocks(projection(
                    temporal.index_select(1, channel_index)
                ))))
            levels = [torch.stack(parameters, dim=1)]
            attentions = []
            for layer in self.upward:
                state, weights = layer(levels[-1])
                levels.append(state)
                attentions.append(weights)
            self.last_attention = attentions
            # Spacecraft -> subsystem -> functional module -> component.
            enriched = levels[-2]
            parent_levels = (levels[-1], levels[-2], levels[-3])
            owner_maps = (
                tuple(0 for _ in range(counts[-2])),
                bindings[3],
                bindings[2],
            )
            for step, (parents, owners, projection, norm) in enumerate(zip(
                parent_levels, owner_maps, self.downward_projections, self.downward_norms,
                strict=True,
            )):
                children = levels[-2 - step]
                parent_context = parents[:, list(owners)]
                gate = torch.sigmoid(self.downward_gate_logits[step])
                enriched = norm(children + gate * projection(parent_context))
                if step < 2:
                    levels[-2 - step] = enriched
            component_state = enriched
            localization = torch.cat([
                head(torch.cat([global_state, component_state[:, index]], dim=1))
                for index, head in enumerate(self.localizer)
            ], dim=1)
            classification = torch.cat([
                head(torch.cat([global_state, component_state[:, owner]], dim=1))
                for owner, head in zip(self.class_component, self.classifier, strict=True)
            ], dim=1)
            return self.detector(global_state), localization, classification

    return ProgressiveHierarchicalGATTCN()


__all__ = ["make_progressive_hierarchical_gat_tcn"]
