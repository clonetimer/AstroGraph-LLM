"""V3 five-layer graph projection for episodes and evidence tokens.

This projects an EpisodeManifest (ABox + Evidence) and its EvidenceTokens /
StateEstimateResult (Evidence layer) into the versioned V3 knowledge graph,
then validates it with GraphShapeValidator. It is deliberately read-only over
the input data and emits only schema-validated summaries — never raw telemetry.
"""

from __future__ import annotations

import hashlib

from astrograph.semantics.contracts import SemanticTelemetryFrame
from astrograph.windowing.episode_models import EpisodeManifest

from .schema_v3 import (
    GraphEdgeV3,
    GraphNodeV3,
    KnowledgeGraphV3,
    KnowledgeNodeTypeV3,
    KnowledgeRelationTypeV3,
    LifecycleStatus,
)


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _node(
    node_id: str,
    node_type: KnowledgeNodeTypeV3,
    source_id: str,
    properties: dict,
    source_payload: str = "",
) -> GraphNodeV3:
    return GraphNodeV3(
        node_id=node_id,
        node_type=node_type,
        source_id=source_id,
        source_sha256=_sha256(source_payload or node_id),
        lifecycle_status=LifecycleStatus.ACTIVE,
        properties=properties,
    )


def _edge(
    edge_id: str,
    source_id: str,
    relation_type: KnowledgeRelationTypeV3,
    target_id: str,
    provenance_id: str,
    properties: dict | None = None,
) -> GraphEdgeV3:
    return GraphEdgeV3(
        edge_id=edge_id,
        source_id=source_id,
        relation_type=relation_type,
        target_id=target_id,
        provenance_id=provenance_id,
        provenance_sha256=_sha256(f"{source_id}|{relation_type.value}|{target_id}"),
        lifecycle_status=LifecycleStatus.ACTIVE,
        properties=properties or {},
    )


def _stable_component_id(asset_id: str, component_instance_id: str) -> str:
    return f"comp:{asset_id}:{component_instance_id}"


def project_episode_to_graph(
    episode: EpisodeManifest,
    *,
    source_id: str = "astrograph:episode-projection",
    tokens: list | None = None,
    state_estimate: object | None = None,
    quantity_kinds: dict[str, dict] | None = None,
    contract: object | None = None,
) -> KnowledgeGraphV3:
    """Project an episode manifest into the V3 five-layer graph.

    ABox: asset instance + component instances (INSTANCE_OF their archetype,
    PART_OF their asset) + per-channel ChannelBinding (with BINDS_COMPONENT /
    BINDS_PORT / BINDS_QUANTITY edges to a PortInstance and QuantityKind).
    Model: optional DynamicsContract + StateEstimatorArtifact nodes that anchor
    the evidence (CONSTRAINT_VIOLATION --VIOLATES--> DYNAMICS_CONTRACT).
    Evidence: one ObservationEpisode node, plus optional StateEstimate /
    Residual / ConstraintViolation / EvidenceFact nodes from the state estimate
    and evidence tokens.

    Only schema-safe summaries are stored; raw time series stay external
    (residual ``values`` arrays and parameter confidence arrays are never copied
    into node properties).
    """
    nodes: list[GraphNodeV3] = []
    edges: list[GraphEdgeV3] = []
    episode_payload = episode.model_dump_json()

    # --- ABox: AssetInstance ---
    asset_id = f"asset:{episode.asset_instance_id}"
    nodes.append(
        _node(
            asset_id,
            KnowledgeNodeTypeV3.ASSET_INSTANCE,
            source_id,
            {
                "asset_instance_id": episode.asset_instance_id,
                "whole_spacecraft_id": episode.whole_spacecraft_id,
                "subsystem_ids": episode.subsystem_ids,
            },
            source_payload=episode_payload,
        )
    )

    # --- ABox: ComponentInstances + ChannelBindings ---
    for comp in episode.components:
        comp_id = _stable_component_id(episode.asset_instance_id, comp.component_instance_id)
        archetype_node = f"archetype:{comp.component_archetype_id}"
        nodes.append(
            _node(
                comp_id,
                KnowledgeNodeTypeV3.COMPONENT_INSTANCE,
                source_id,
                {
                    "component_instance_id": comp.component_instance_id,
                    "component_archetype_id": comp.component_archetype_id,
                    "start_s": comp.start_s,
                    "end_s": comp.end_s,
                    "row_count": comp.row_count,
                    "data_quality_status": comp.data_quality.status if comp.data_quality else None,
                },
            )
        )
        # ComponentInstance --PART_OF--> AssetInstance
        edges.append(
            _edge(
                f"e:{comp_id}:part_of:{asset_id}",
                comp_id,
                KnowledgeRelationTypeV3.PART_OF,
                asset_id,
                source_id,
            )
        )

        # Per-channel binding: ChannelBinding + PortInstance + QuantityKind.
        for channel_id in comp.semantic_channel_ids:
            binding_id = f"bind:{episode.asset_instance_id}:{comp.component_instance_id}:{channel_id}"
            port_id = f"port:{episode.asset_instance_id}:{comp.component_instance_id}:{channel_id}"

            nodes.append(
                _node(
                    binding_id,
                    KnowledgeNodeTypeV3.CHANNEL_BINDING,
                    source_id,
                    {"semantic_channel_id": channel_id, "source_field": ""},
                )
            )
            nodes.append(
                _node(
                    port_id,
                    KnowledgeNodeTypeV3.PORT_INSTANCE,
                    source_id,
                    {"semantic_channel_id": channel_id},
                )
            )
            # PortInstance --PART_OF--> ComponentInstance
            edges.append(
                _edge(
                    f"e:{port_id}:part_of:{comp_id}",
                    port_id,
                    KnowledgeRelationTypeV3.PART_OF,
                    comp_id,
                    source_id,
                )
            )
            # ChannelBinding --BINDS_COMPONENT/PORT--> instances.
            edges.append(
                _edge(
                    f"e:{binding_id}:binds_comp:{comp_id}",
                    binding_id,
                    KnowledgeRelationTypeV3.BINDS_COMPONENT,
                    comp_id,
                    source_id,
                )
            )
            edges.append(
                _edge(
                    f"e:{binding_id}:binds_port:{port_id}",
                    binding_id,
                    KnowledgeRelationTypeV3.BINDS_PORT,
                    port_id,
                    source_id,
                )
            )
            # ChannelBinding --BINDS_QUANTITY--> QuantityKind (from schema map).
            qk_info = (quantity_kinds or {}).get(channel_id)
            if qk_info is not None:
                qk_id = f"qk:{qk_info['quantity_kind_id']}"
                if not any(n.node_id == qk_id for n in nodes):
                    nodes.append(
                        _node(
                            qk_id,
                            KnowledgeNodeTypeV3.QUANTITY_KIND,
                            source_id,
                            {
                                "quantity_kind_id": qk_info["quantity_kind_id"],
                                "canonical_unit": qk_info["canonical_unit"],
                                "dimension": qk_info["dimension"],
                            },
                        )
                    )
                edges.append(
                    _edge(
                        f"e:{binding_id}:binds_qk:{qk_id}",
                        binding_id,
                        KnowledgeRelationTypeV3.BINDS_QUANTITY,
                        qk_id,
                        source_id,
                    )
                )

    # --- Evidence: ObservationEpisode ---
    episode_id = f"ep:{episode.episode_id}"
    nodes.append(
        _node(
            episode_id,
            KnowledgeNodeTypeV3.OBSERVATION_EPISODE,
            source_id,
            {
                "start_s": episode.start_s,
                "end_s": episode.end_s,
                "overall_quality": episode.overall_data_quality.status if episode.overall_data_quality else None,
                "component_count": len(episode.components),
            },
            source_payload=episode_payload,
        )
    )

    # --- Evidence: tokens as EvidenceFact ---
    if tokens:
        for token_index, token in enumerate(tokens):
            # A single operator primitive may emit multiple OperatorResults (one
            # per input channel), all sharing the combined port-role string.  The
            # fact id must stay unique per token, so the digest mixes the token
            # index (deterministic for a fixed token order).
            digest = hashlib.sha256(
                f"{token.operator_id}|{token.port_role_id}|{token.quantity_kind_id}|{token_index}".encode("utf-8")
            ).hexdigest()[:8]
            fact_id = f"fact:{episode.episode_id}:{token.operator_id}:{digest}"
            nodes.append(
                _node(
                    fact_id,
                    KnowledgeNodeTypeV3.EVIDENCE_FACT,
                    source_id,
                    {
                        "operator_id": token.operator_id,
                        "quantity_kind_id": token.quantity_kind_id,
                        "port_role_id": token.port_role_id,
                        "value": token.value,
                        "uncertainty": token.uncertainty,
                        "availability": token.availability,
                        "quality_flags": token.quality_flags,
                    },
                    source_payload=token.model_dump_json(),
                )
            )
            edges.append(
                _edge(
                    f"e:{fact_id}:in_episode:{episode_id}",
                    fact_id,
                    KnowledgeRelationTypeV3.IN_EPISODE,
                    episode_id,
                    source_id,
                )
            )

    # --- Evidence: StateEstimate + Residual + ConstraintViolation ---
    if state_estimate is not None:
        est_id = f"estimate:{episode.episode_id}"
        # Model layer: the dynamics contract + estimator artifact anchor the
        # evidence (used by CONSTRAINT_VIOLATION --VIOLATES--> contract and
        # STATE_ESTIMATE --DERIVED_FROM--> artifact).
        receipt = getattr(state_estimate, "receipt", None)
        contract_id = getattr(receipt, "dynamics_contract_id", "") if receipt is not None else ""
        if contract is not None:
            contract_id = getattr(contract, "contract_id", "") or contract_id
        contract_node_id = f"contract:{contract_id}" if contract_id else f"contract:{episode.episode_id}"
        nodes.append(
            _node(
                contract_node_id,
                KnowledgeNodeTypeV3.DYNAMICS_CONTRACT,
                source_id,
                {"dynamics_contract_id": contract_id, "equation_family": getattr(contract, "equation_family", "")},
            )
        )
        artifact_id = f"estimator:{episode.episode_id}"
        estimator_id = getattr(receipt, "estimator_id", "") if receipt is not None else ""
        nodes.append(
            _node(
                artifact_id,
                KnowledgeNodeTypeV3.STATE_ESTIMATOR_ARTIFACT,
                source_id,
                {"estimator_id": estimator_id, "estimator_version": getattr(receipt, "estimator_version", "")},
            )
        )
        edges.append(
            _edge(
                f"e:{artifact_id}:implements:{contract_node_id}",
                artifact_id,
                KnowledgeRelationTypeV3.IMPLEMENTS,
                contract_node_id,
                source_id,
            )
        )

        parameter_summary = [
            {
                "parameter_id": p.parameter_id,
                "value": p.value,
                "canonical_unit": p.canonical_unit,
            }
            for p in getattr(state_estimate, "parameters", [])
        ]
        nodes.append(
            _node(
                est_id,
                KnowledgeNodeTypeV3.STATE_ESTIMATE,
                source_id,
                {
                    "component_archetype_id": getattr(state_estimate, "component_archetype_id", ""),
                    "ood_score": getattr(state_estimate, "ood_score", None),
                    "residual_count": len(getattr(state_estimate, "residuals", [])),
                    "parameter_count": len(getattr(state_estimate, "parameters", [])),
                    "parameters": parameter_summary,
                },
                source_payload=(
                    state_estimate.model_dump_json()
                    if hasattr(state_estimate, "model_dump_json")
                    else ""
                ),
            )
        )
        edges.append(
            _edge(
                f"e:{est_id}:in_episode:{episode_id}",
                est_id,
                KnowledgeRelationTypeV3.IN_EPISODE,
                episode_id,
                source_id,
            )
        )
        edges.append(
            _edge(
                f"e:{est_id}:derived_from:{artifact_id}",
                est_id,
                KnowledgeRelationTypeV3.DERIVED_FROM,
                artifact_id,
                source_id,
            )
        )

        # Residual series -> Evidence Residual nodes (summary statistics only;
        # the raw residual value array is never copied into the graph).
        for residual_index, residual in enumerate(getattr(state_estimate, "residuals", [])):
            residual_id = getattr(residual, "residual_id", f"residual_{residual_index}")
            residual_node_id = f"residual:{episode.episode_id}:{residual_index}:{_sha256(str(residual_id))[:8]}"
            nodes.append(
                _node(
                    residual_node_id,
                    KnowledgeNodeTypeV3.RESIDUAL,
                    source_id,
                    {
                        "residual_id": residual_id,
                        "canonical_unit": getattr(residual, "canonical_unit", ""),
                        "root_mean_square": getattr(residual, "root_mean_square", None),
                        "mean": getattr(residual, "mean", None),
                    },
                )
            )
            edges.append(
                _edge(
                    f"e:{residual_node_id}:in_episode:{episode_id}",
                    residual_node_id,
                    KnowledgeRelationTypeV3.IN_EPISODE,
                    episode_id,
                    source_id,
                )
            )
            edges.append(
                _edge(
                    f"e:{residual_node_id}:derived_from:{est_id}",
                    residual_node_id,
                    KnowledgeRelationTypeV3.DERIVED_FROM,
                    est_id,
                    source_id,
                )
            )

        # Constraint violations -> one Evidence ConstraintViolation node per code
        # (aggregated: count + max magnitude).  Avoids node explosion on long
        # traces where thousands of pointwise residuals would otherwise appear.
        violations_by_code: dict[str, dict] = {}
        for violation in getattr(state_estimate, "constraint_violations", []):
            code = getattr(violation, "code", "CONSERVATION_RESIDUAL_EXCEEDED")
            magnitude = float(getattr(violation, "magnitude", 0.0))
            entry = violations_by_code.setdefault(
                code, {"code": code, "count": 0, "max_magnitude": 0.0, "limit": getattr(violation, "limit", 0.0)}
            )
            entry["count"] += 1
            entry["max_magnitude"] = max(entry["max_magnitude"], magnitude)
        for code, summary in violations_by_code.items():
            violation_node_id = f"violation:{episode.episode_id}:{_sha256(code)[:8]}"
            nodes.append(
                _node(
                    violation_node_id,
                    KnowledgeNodeTypeV3.CONSTRAINT_VIOLATION,
                    source_id,
                    summary,
                )
            )
            edges.append(
                _edge(
                    f"e:{violation_node_id}:in_episode:{episode_id}",
                    violation_node_id,
                    KnowledgeRelationTypeV3.IN_EPISODE,
                    episode_id,
                    source_id,
                )
            )
            edges.append(
                _edge(
                    f"e:{violation_node_id}:violates:{contract_node_id}",
                    violation_node_id,
                    KnowledgeRelationTypeV3.VIOLATES,
                    contract_node_id,
                    source_id,
                )
            )
            edges.append(
                _edge(
                    f"e:{violation_node_id}:derived_from:{est_id}",
                    violation_node_id,
                    KnowledgeRelationTypeV3.DERIVED_FROM,
                    est_id,
                    source_id,
                )
            )

    graph_id = f"graph:{episode.episode_id}"
    return KnowledgeGraphV3(graph_id=graph_id, nodes=nodes, edges=edges)


def _channel_quantity_kind(semantic_channel_id: str) -> str:
    """Map a semantic channel id to a plausible quantity-kind node id.

    The projection cannot infer physics from a channel id, so this returns a
    stable placeholder that the caller may overwrite with real QuantityKind
    nodes. It stays in TBox namespace.
    """
    digest = hashlib.sha256(semantic_channel_id.encode("utf-8")).hexdigest()[:12]
    return f"qk:{digest}"


__all__ = ["project_episode_to_graph"]
