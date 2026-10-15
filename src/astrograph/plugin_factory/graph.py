from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np

from astrograph.operators import BUILTIN_OPERATOR_REGISTRY, DeterministicOperatorRegistry, OperatorInput
from astrograph.plugin_factory.models import (
    OperatorDataRef,
    OperatorDataSourceKind,
    OperatorExecutionRecord,
    OperatorGraphNodeSpec,
    OperatorGraphSpec,
)


class OperatorGraphCompiler:
    """Validates and topologically orders a white-listed deterministic DAG."""

    def __init__(self, operator_registry: DeterministicOperatorRegistry | None = None) -> None:
        self.operators = operator_registry or BUILTIN_OPERATOR_REGISTRY

    def compile(self, graph: OperatorGraphSpec) -> list[OperatorGraphNodeSpec]:
        nodes = {node.node_id: node for node in graph.nodes}
        indegree = {node_id: 0 for node_id in nodes}
        outgoing: dict[str, list[str]] = defaultdict(list)

        for node in graph.nodes:
            if not self.operators.has(node.operator_id):
                raise ValueError(
                    f"operator graph node {node.node_id} references unknown operator "
                    f"{node.operator_id}"
                )
            definition = self.operators.get(node.operator_id)
            if definition.spec.requires_baseline and not node.baseline_inputs:
                raise ValueError(
                    f"operator graph node {node.node_id} uses {node.operator_id}, which requires "
                    "baseline_inputs"
                )
            if definition.spec.requires_time and node.time_input is None:
                raise ValueError(
                    f"operator graph node {node.node_id} uses {node.operator_id}, which requires "
                    "time_input"
                )
            dependencies = {
                ref.ref
                for ref in [*node.current_inputs, *node.baseline_inputs]
                if ref.source == OperatorDataSourceKind.NODE_OUTPUT
            }
            for dependency in dependencies:
                if dependency not in nodes:
                    raise ValueError(
                        f"operator graph node {node.node_id} references unknown node {dependency}"
                    )
                outgoing[str(dependency)].append(node.node_id)
                indegree[node.node_id] += 1

        ready = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
        ordered: list[OperatorGraphNodeSpec] = []
        while ready:
            node_id = ready.popleft()
            ordered.append(nodes[node_id])
            for target in sorted(outgoing.get(node_id, [])):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)

        if len(ordered) != len(nodes):
            cyclic = sorted(node_id for node_id, degree in indegree.items() if degree > 0)
            raise ValueError(f"operator graph contains a cycle: {cyclic}")
        return ordered


class OperatorGraphExecutor:
    """Executes a compiled operator graph without evaluating arbitrary code."""

    def __init__(self, operator_registry: DeterministicOperatorRegistry | None = None) -> None:
        self.operators = operator_registry or BUILTIN_OPERATOR_REGISTRY
        self.compiler = OperatorGraphCompiler(self.operators)

    def execute(
        self,
        graph: OperatorGraphSpec,
        *,
        current_signals: dict[str, Any],
        baseline_signals: dict[str, Any] | None = None,
        time_s: Any | None = None,
    ) -> tuple[float | None, list[OperatorExecutionRecord], list[str]]:
        ordered = self.compiler.compile(graph)
        values: dict[str, float | None] = {}
        trace: list[OperatorExecutionRecord] = []
        overall_flags: list[str] = []
        for node in ordered:
            current = self._resolve_many(
                node.current_inputs,
                current_signals=current_signals,
                baseline_signals=baseline_signals,
                node_values=values,
                time_s=time_s,
            )
            baseline = None
            if node.baseline_inputs:
                baseline = self._resolve_many(
                    node.baseline_inputs,
                    current_signals=current_signals,
                    baseline_signals=baseline_signals,
                    node_values=values,
                    time_s=time_s,
                )
            node_time = None if node.time_input is None else np.asarray(time_s, dtype=float)
            quality_flags: list[str] = []
            if not np.isfinite(current).all():
                quality_flags.append("NONFINITE_CURRENT_INPUT")
            if baseline is not None and not np.isfinite(baseline).all():
                quality_flags.append("NONFINITE_BASELINE_INPUT")
            value = self.operators.compute(
                node.operator_id,
                OperatorInput(
                    current=current,
                    baseline=baseline,
                    time_s=node_time,
                    parameters=node.parameters,
                ),
            )
            if value is None:
                quality_flags.append("NO_FINITE_RESULT")
            values[node.node_id] = value
            trace.append(
                OperatorExecutionRecord(
                    node_id=node.node_id,
                    operator_id=node.operator_id,
                    value=value,
                    quality_flags=quality_flags,
                )
            )
            for flag in quality_flags:
                qualified = f"{node.node_id}:{flag}"
                if qualified not in overall_flags:
                    overall_flags.append(qualified)
        return values[graph.output_node_id], trace, overall_flags

    @staticmethod
    def _resolve_many(
        refs: list[OperatorDataRef],
        *,
        current_signals: dict[str, Any],
        baseline_signals: dict[str, Any] | None,
        node_values: dict[str, float | None],
        time_s: Any | None,
    ) -> np.ndarray:
        arrays = [
            OperatorGraphExecutor._resolve_one(
                ref,
                current_signals=current_signals,
                baseline_signals=baseline_signals,
                node_values=node_values,
                time_s=time_s,
            )
            for ref in refs
        ]
        if len(arrays) == 1:
            return arrays[0]
        return np.concatenate([array.reshape(-1) for array in arrays])

    @staticmethod
    def _resolve_one(
        ref: OperatorDataRef,
        *,
        current_signals: dict[str, Any],
        baseline_signals: dict[str, Any] | None,
        node_values: dict[str, float | None],
        time_s: Any | None,
    ) -> np.ndarray:
        if ref.source == OperatorDataSourceKind.CURRENT_SIGNAL:
            if ref.ref not in current_signals:
                raise KeyError(f"missing current signal binding: {ref.ref}")
            return np.asarray(current_signals[str(ref.ref)], dtype=float).reshape(-1)
        if ref.source == OperatorDataSourceKind.BASELINE_SIGNAL:
            if baseline_signals is None or ref.ref not in baseline_signals:
                raise KeyError(f"missing baseline signal binding: {ref.ref}")
            return np.asarray(baseline_signals[str(ref.ref)], dtype=float).reshape(-1)
        if ref.source == OperatorDataSourceKind.NODE_OUTPUT:
            if ref.ref not in node_values:
                raise RuntimeError(f"operator graph dependency has not executed: {ref.ref}")
            value = node_values[str(ref.ref)]
            return np.asarray([np.nan if value is None else value], dtype=float)
        if ref.source == OperatorDataSourceKind.TIME:
            if time_s is None:
                raise ValueError("operator graph requires time_s")
            return np.asarray(time_s, dtype=float).reshape(-1)
        raise ValueError(f"unsupported operator graph data source: {ref.source}")
