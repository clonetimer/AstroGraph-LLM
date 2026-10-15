from __future__ import annotations

from typing import Any

import numpy as np

from astrograph.operators import BUILTIN_OPERATOR_REGISTRY, DeterministicOperatorRegistry, OperatorInput
from astrograph.plugin_factory.graph import OperatorGraphExecutor
from astrograph.plugin_factory.models import (
    ComputedFeatureValue,
    FeatureImplementationMode,
    FeatureServiceSpec,
    OperatorDataRef,
    OperatorDataSourceKind,
    OperatorExecutionRecord,
    OperatorGraphNodeSpec,
    OperatorGraphSpec,
)


class ConfigDrivenFeatureService:
    """Object-independent executor for direct and composed approved operators."""

    def __init__(self, operator_registry: DeterministicOperatorRegistry | None = None) -> None:
        self.operators = operator_registry or BUILTIN_OPERATOR_REGISTRY
        self.graph_executor = OperatorGraphExecutor(self.operators)

    def compute(
        self,
        spec: FeatureServiceSpec,
        *,
        current_signals: dict[str, Any],
        baseline_signals: dict[str, Any] | None = None,
        time_s: Any | None = None,
    ) -> ComputedFeatureValue:
        if spec.operator_id is not None:
            return self._compute_direct(
                spec,
                current_signals=current_signals,
                baseline_signals=baseline_signals,
                time_s=time_s,
            )
        if spec.operator_graph is not None:
            value, trace, flags = self.graph_executor.execute(
                spec.operator_graph,
                current_signals=current_signals,
                baseline_signals=baseline_signals,
                time_s=time_s,
            )
            output_node = next(
                node for node in spec.operator_graph.nodes if node.node_id == spec.operator_graph.output_node_id
            )
            return ComputedFeatureValue(
                feature_id=spec.feature_id,
                value=value,
                unit=spec.output_unit,
                operator_id=f"DAG:{output_node.operator_id}",
                implementation_mode=FeatureImplementationMode.OPERATOR_COMPOSITION,
                quality_flags=flags,
                operator_trace=trace,
            )
        if spec.operator_pipeline:
            graph = self._compile_legacy_pipeline(spec)
            value, trace, flags = self.graph_executor.execute(
                graph,
                current_signals=current_signals,
                baseline_signals=baseline_signals,
                time_s=time_s,
            )
            return ComputedFeatureValue(
                feature_id=spec.feature_id,
                value=value,
                unit=spec.output_unit,
                operator_id=f"PIPELINE:{spec.operator_pipeline[-1]}",
                implementation_mode=FeatureImplementationMode.OPERATOR_COMPOSITION,
                quality_flags=flags,
                operator_trace=trace,
            )
        raise ValueError("custom-code feature cannot execute in the configuration runtime")

    def _compute_direct(
        self,
        spec: FeatureServiceSpec,
        *,
        current_signals: dict[str, Any],
        baseline_signals: dict[str, Any] | None,
        time_s: Any | None,
    ) -> ComputedFeatureValue:
        aliases = dict(spec.input_bindings)
        current_binding = aliases.get("current") or aliases.get("signal")
        if current_binding is None:
            raise ValueError(f"feature {spec.feature_id} requires a current or signal input alias")
        if current_binding not in current_signals:
            raise KeyError(f"missing current signal binding: {current_binding}")

        baseline = None
        baseline_binding = aliases.get("baseline", current_binding)
        if baseline_signals is not None and baseline_binding in baseline_signals:
            baseline = baseline_signals[baseline_binding]

        current = np.asarray(current_signals[current_binding], dtype=float)
        quality_flags: list[str] = []
        if not np.isfinite(current).all():
            quality_flags.append("NONFINITE_CURRENT_INPUT")
        if baseline is not None and not np.isfinite(np.asarray(baseline, dtype=float)).all():
            quality_flags.append("NONFINITE_BASELINE_INPUT")
        value = self.operators.compute(
            str(spec.operator_id),
            OperatorInput(
                current=current,
                baseline=baseline,
                time_s=time_s,
                parameters=spec.operator_parameters,
            ),
        )
        if value is None:
            quality_flags.append("NO_FINITE_RESULT")
        record = OperatorExecutionRecord(
            node_id="direct",
            operator_id=str(spec.operator_id),
            value=value,
            quality_flags=quality_flags,
        )
        return ComputedFeatureValue(
            feature_id=spec.feature_id,
            value=value,
            unit=spec.output_unit,
            operator_id=str(spec.operator_id),
            implementation_mode=FeatureImplementationMode.CONFIGURATION_ONLY,
            quality_flags=quality_flags,
            operator_trace=[record],
        )

    def _compile_legacy_pipeline(self, spec: FeatureServiceSpec) -> OperatorGraphSpec:
        current_binding = spec.input_bindings.get("current") or spec.input_bindings.get("signal")
        if current_binding is None:
            raise ValueError(f"feature {spec.feature_id} requires current/signal for operator_pipeline")
        baseline_binding = spec.input_bindings.get("baseline", current_binding)
        nodes: list[OperatorGraphNodeSpec] = []
        previous: str | None = None
        for index, operator_id in enumerate(spec.operator_pipeline):
            definition = self.operators.get(operator_id)
            current_ref = (
                OperatorDataRef(source=OperatorDataSourceKind.CURRENT_SIGNAL, ref=current_binding)
                if previous is None
                else OperatorDataRef(source=OperatorDataSourceKind.NODE_OUTPUT, ref=previous)
            )
            baseline_refs: list[OperatorDataRef] = []
            if definition.spec.requires_baseline:
                if previous is not None:
                    raise ValueError(
                        f"legacy pipeline operator {operator_id} requires a baseline after an "
                        "intermediate scalar; use explicit operator_graph"
                    )
                baseline_refs = [
                    OperatorDataRef(
                        source=OperatorDataSourceKind.BASELINE_SIGNAL,
                        ref=baseline_binding,
                    )
                ]
            node_id = f"step_{index + 1}"
            nodes.append(
                OperatorGraphNodeSpec(
                    node_id=node_id,
                    operator_id=operator_id,
                    current_inputs=[current_ref],
                    baseline_inputs=baseline_refs,
                    time_input=(
                        OperatorDataRef(source=OperatorDataSourceKind.TIME)
                        if definition.spec.requires_time
                        else None
                    ),
                    parameters=spec.operator_parameters if index == 0 else {},
                )
            )
            previous = node_id
        return OperatorGraphSpec(nodes=nodes, output_node_id=str(previous))
