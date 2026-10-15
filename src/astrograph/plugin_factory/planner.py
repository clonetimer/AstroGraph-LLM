from __future__ import annotations

from astrograph.operators import BUILTIN_OPERATOR_REGISTRY, DeterministicOperatorRegistry
from astrograph.plugin_factory.graph import OperatorGraphCompiler
from astrograph.plugin_factory.models import (
    FeatureGenerationDecision,
    FeatureImplementationMode,
    ObjectPluginBuildSpec,
    PluginGenerationPlan,
)


class PluginGenerationPlanner:
    """Converts an Agent-authored build specification into a deterministic plan."""

    def __init__(self, operator_registry: DeterministicOperatorRegistry | None = None) -> None:
        self.operators = operator_registry or BUILTIN_OPERATOR_REGISTRY
        self.graph_compiler = OperatorGraphCompiler(self.operators)

    def plan(self, spec: ObjectPluginBuildSpec) -> PluginGenerationPlan:
        decisions: list[FeatureGenerationDecision] = []
        required: set[str] = set()
        for feature in spec.features:
            if feature.operator_id is not None:
                if self.operators.has(feature.operator_id):
                    required.add(feature.operator_id)
                    decisions.append(
                        FeatureGenerationDecision(
                            feature_id=feature.feature_id,
                            mode=FeatureImplementationMode.CONFIGURATION_ONLY,
                            resolved_operator_ids=[feature.operator_id],
                            reason="Existing deterministic operator can implement the feature directly.",
                        )
                    )
                else:
                    decisions.append(
                        FeatureGenerationDecision(
                            feature_id=feature.feature_id,
                            mode=FeatureImplementationMode.BLOCKED,
                            resolved_operator_ids=[feature.operator_id],
                            reason="Requested operator is not present in the approved operator registry.",
                            requires_human_review=True,
                        )
                    )
            elif feature.operator_pipeline or feature.operator_graph is not None:
                operator_ids = (
                    list(feature.operator_pipeline)
                    if feature.operator_pipeline
                    else [node.operator_id for node in feature.operator_graph.nodes]
                )
                missing = [item for item in operator_ids if not self.operators.has(item)]
                graph_error = None
                if not missing and feature.operator_graph is not None:
                    try:
                        self.graph_compiler.compile(feature.operator_graph)
                    except ValueError as exc:
                        graph_error = str(exc)
                if missing or graph_error:
                    reason = (
                        f"Operator composition references unregistered operators: {sorted(missing)}."
                        if missing
                        else f"Operator graph failed deterministic validation: {graph_error}."
                    )
                    decisions.append(
                        FeatureGenerationDecision(
                            feature_id=feature.feature_id,
                            mode=FeatureImplementationMode.BLOCKED,
                            resolved_operator_ids=operator_ids,
                            reason=reason,
                            requires_human_review=True,
                        )
                    )
                else:
                    required.update(operator_ids)
                    decisions.append(
                        FeatureGenerationDecision(
                            feature_id=feature.feature_id,
                            mode=FeatureImplementationMode.OPERATOR_COMPOSITION,
                            resolved_operator_ids=operator_ids,
                            reason="Feature can be expressed as an approved deterministic operator composition.",
                        )
                    )
            else:
                decisions.append(
                    FeatureGenerationDecision(
                        feature_id=feature.feature_id,
                        mode=FeatureImplementationMode.CUSTOM_CODE_REQUIRED,
                        reason="No approved operator implementation exists; a reviewed object-specific operator is required.",
                        requires_human_review=True,
                    )
                )

        counts = {
            mode: sum(item.mode == mode for item in decisions)
            for mode in FeatureImplementationMode
        }
        requires_review = any(item.requires_human_review for item in decisions)
        return PluginGenerationPlan(
            plugin_id=spec.plugin_id,
            plugin_version=spec.plugin_version,
            decisions=decisions,
            required_operator_ids=sorted(required),
            configuration_only_feature_count=counts[FeatureImplementationMode.CONFIGURATION_ONLY],
            composition_feature_count=counts[FeatureImplementationMode.OPERATOR_COMPOSITION],
            custom_code_feature_count=counts[FeatureImplementationMode.CUSTOM_CODE_REQUIRED],
            blocked_feature_count=counts[FeatureImplementationMode.BLOCKED],
            requires_human_review=requires_review,
            online_registration_allowed=False,
        )
